import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from utils.dataloader import FcosDataset
from utils.utils import get_lr_scheduler, set_optimizer_lr
from utils.utils import show_config
import torch.backends.cudnn as cudnn
from utils.callbacks import LossHistory
import datetime
from models.model import Model
from utils.loss import Fcos_Loss
import yaml
from pathlib import Path
from utils.utils_fit import fit_one_epoch
from utils.dataloader import fcos_dataset_collate
import os
from utils.callbacks import EvalCallback
import argparse
import sys
import torch.distributed as dist

FILE = Path(__file__).resolve()
ROOT = FILE.parents[0]
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

ROOT = Path(os.path.relpath(ROOT, Path.cwd()))


def run(opt):
    Min_lr = opt.Init_lr * 0.01
    input_shape = [opt.input_shape, opt.input_shape] if type(opt.input_shape) is int else opt.input_shape

    # 设置用到的显卡
    ngpus_per_node = torch.cuda.device_count()
    if opt.distributed:
        dist.init_process_group(backend="nccl")
        local_rank = int(os.environ["LOCAL_RANK"])
        rank = int(os.environ["RANK"])
        device = torch.device("cuda", local_rank)
        if local_rank == 0:
            print(f"[{os.getpid()}] (rank = {rank}, local_rank = {local_rank}) training...")
            print("Gpu Device Count : ", ngpus_per_node)
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        local_rank = 0

    # 获取 classes
    with open(opt.data, encoding="utf-8", errors="ignore") as f:
        data = yaml.safe_load(f)
        f.close()

    train_path = data['train']
    val_path = data['val']
    class_names = data['names']
    num_classes = data['nc']

    model = Model(cfg=opt.cfg)
    if opt.model_path != "":
        if local_rank == 0:
            print('Load weights {}.'.format(opt.model_path))

        model_dict = model.state_dict()
        pretrained_dict = torch.load(opt.model_path, map_location=device)
        load_key, no_load_key, temp_dict = [], [], {}
        for k, v in pretrained_dict.items():
            if k in model_dict.keys() and np.shape(model_dict[k] == np.shape(v)):
                temp_dict[k] = v
                load_key.append(k)
            else:
                no_load_key.append(k)
        model_dict.update(temp_dict)
        model.load_state_dict(model_dict)

        if local_rank == 0:
            print("\nSuccessful Load Key:", str(load_key)[:500], "……\nSuccessful Load Key Num:", len(load_key))
            print("\nFail To Load Key:", str(no_load_key)[:500], "……\nFail To Load Key num:", len(no_load_key))
            print("\n\033[1;33;44m温馨提示，head部分没有载入是正常现象，Backbone部分没有载入是错误的。\033[0m")
    fcos_loss = Fcos_Loss(model)

    if local_rank == 0:
        time_str = datetime.datetime.strftime(datetime.datetime.now(), '%Y_%m_%d_%H_%M_%S')
        log_dir = os.path.join(opt.save_dir, "loss_" + str(time_str))
        loss_history = LossHistory(log_dir, model, input_shape=input_shape)
    else:
        loss_history = None

    if opt.fp16:
        from torch.cuda.amp import GradScaler as GradScaler
        scaler = GradScaler()
    else:
        scaler = None

    model_train = model.train()

    if opt.sync_bn and ngpus_per_node > 1 and opt.distributed:
        model_train = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model_train)
    elif opt.sync_bn:
        print("Sync_bn is not support in one gpu or not distributed.")
    if opt.cuda:
        if opt.distributed:
            model_train = model_train.cuda(local_rank)
            model_train = torch.nn.parallel.DistributedDataParallel(model_train, device_ids=[local_rank],
                                                                    find_unused_parameters=True)
        else:
            model_train = torch.nn.DataParallel(model)
            cudnn.benchmark = True
            model_train = model_train.cuda()
    with open(train_path, encoding="utf-8") as f:
        train_lines = f.readlines()
        f.close()
    with open(val_path, encoding="utf-8") as f:
        val_lines = f.readlines()
        f.close()

    num_train = len(train_lines)
    num_val = len(val_lines)

    if local_rank == 0:
        show_config(
            input_shape=input_shape, \
            Init_Epoch=opt.Init_Epoch, Freeze_Epoch=opt.Freeze_Epoch, UnFreeze_Epoch=opt.UnFreeze_Epoch,
            Freeze_batch_size=opt.Freeze_batch_size, Unfreeze_batch_size=opt.Unfreeze_batch_size,
            Freeze_Train=opt.Freeze_Train, \
            Init_lr=opt.Init_lr, Min_lr=Min_lr, optimizer_type=opt.optimizer, momentum=opt.momentum,
            lr_decay_type=opt.lr_decay_type, \
            save_period=opt.save_period, save_dir=opt.save_dir, num_workers=opt.num_workers, num_train=num_train,
            num_val=num_val
        )
        # ---------------------------------------------------------#
        #   总训练世代指的是遍历全部数据的总次数
        #   总训练步长指的是梯度下降的总次数
        #   每个训练世代包含若干训练步长，每个训练步长进行一次梯度下降。
        #   此处仅建议最低训练世代，上不封顶，计算时只考虑了解冻部分
        # ----------------------------------------------------------#
        wanted_step = 5e4 if opt.optimizer == "sgd" else 1.5e4
        total_step = num_train // opt.Unfreeze_batch_size * opt.UnFreeze_Epoch
        if total_step <= wanted_step:
            if num_train // opt.Unfreeze_batch_size == 0:
                raise ValueError('数据集过小，无法进行训练，请扩充数据集。')
            wanted_epoch = wanted_step // (num_train // opt.Unfreeze_batch_size) + 1
            print("\n\033[1;33;44m[Warning] 使用%s优化器时，建议将训练总步长设置到%d以上。\033[0m" % (opt.optimizer, wanted_step))
            print("\033[1;33;44m[Warning] 本次运行的总训练数据量为%d，Unfreeze_batch_size为%d，共训练%d个Epoch，计算出总训练步长为%d。\033[0m" % (
                num_train, opt.Unfreeze_batch_size, opt.UnFreeze_Epoch, total_step))
            print("\033[1;33;44m[Warning] 由于总训练步长为%d，小于建议总步长%d，建议设置总世代为%d。\033[0m" % (
                total_step, wanted_step, wanted_epoch))
    if True:
        UnFreeze_flag = False
        # ------------------------------------#
        #   冻结一定部分训练
        # ------------------------------------#
        if opt.Freeze_Train:
            for param in model.backbone.parameters():
                param.requires_grad = False

        # -------------------------------------------------------------------#
        #   如果不冻结训练的话，直接设置batch_size为Unfreeze_batch_size
        # -------------------------------------------------------------------#
        batch_size = opt.Freeze_batch_size if opt.Freeze_Train else opt.Unfreeze_batch_size

        # -------------------------------------------------------------------#
        #   判断当前batch_size，自适应调整学习率
        # -------------------------------------------------------------------#
        nbs = 16
        lr_limit_max = 5e-4 if opt.optimizer == 'adam' else 5e-2
        lr_limit_min = 3e-4 if opt.optimizer == 'adam' else 5e-4
        Init_lr_fit = min(max(batch_size / nbs * opt.Init_lr, lr_limit_min), lr_limit_max)
        Min_lr_fit = min(max(batch_size / nbs * Min_lr, lr_limit_min * 1e-2), lr_limit_max * 1e-2)

        # ---------------------------------------#
        #   根据optimizer_type选择优化器
        # ---------------------------------------#
        pg0, pg1, pg2 = [], [], []
        for k, v in model.named_modules():
            if hasattr(v, "bias") and isinstance(v.bias, nn.Parameter):
                pg2.append(v.bias)
            if isinstance(v, nn.BatchNorm2d) or "bn" in k:
                pg0.append(v.weight)
            elif hasattr(v, "weight") and isinstance(v.weight, nn.Parameter):
                pg1.append(v.weight)
        optimizer = {
            'adam': optim.Adam(pg0, Init_lr_fit, betas=(opt.momentum, 0.999)),
            'sgd': optim.SGD(pg0, Init_lr_fit, momentum=opt.momentum, nesterov=True)
        }[opt.optimizer]
        optimizer.add_param_group({"params": pg1, "weight_decay": opt.weight_decay})
        optimizer.add_param_group({"params": pg2})

        # ---------------------------------------#
        #   获得学习率下降的公式
        # ---------------------------------------#
        lr_scheduler_func = get_lr_scheduler(opt.lr_decay_type, Init_lr_fit, Min_lr_fit, opt.UnFreeze_Epoch)

        # ---------------------------------------#
        #   判断每一个世代的长度
        # ---------------------------------------#
        epoch_step = num_train // batch_size
        epoch_step_val = num_val // batch_size

        if epoch_step == 0 or epoch_step_val == 0:
            raise ValueError("数据集过小，无法继续进行训练，请扩充数据集。")

        train_dataset = FcosDataset(train_lines, input_shape, train=True)
        val_dataset = FcosDataset(val_lines, input_shape, train=False)

        if opt.distributed:
            train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset, shuffle=True, )
            val_sampler = torch.utils.data.distributed.DistributedSampler(val_dataset, shuffle=False, )
            batch_size = batch_size // ngpus_per_node
            shuffle = False
        else:
            train_sampler = None
            val_sampler = None
            shuffle = True

        gen = DataLoader(train_dataset, shuffle=shuffle, batch_size=batch_size, num_workers=opt.num_workers,
                         pin_memory=True,
                         drop_last=True, collate_fn=fcos_dataset_collate, sampler=train_sampler)
        gen_val = DataLoader(val_dataset, shuffle=shuffle, batch_size=batch_size, num_workers=opt.num_workers,
                             pin_memory=True,
                             drop_last=True, collate_fn=fcos_dataset_collate, sampler=val_sampler)

        # ----------------------#
        #   记录eval的map曲线
        # ----------------------#
        if local_rank == 0:
            eval_callback = EvalCallback(model, input_shape, fcos_loss.strides, class_names, num_classes, val_lines,
                                         log_dir,
                                         opt.cuda, \
                                         eval_flag=opt.eval_flag, period=opt.eval_period)
        else:
            eval_callback = None

        # ---------------------------------------#
        #   开始模型训练
        # ---------------------------------------#
        for epoch in range(opt.Init_Epoch, opt.UnFreeze_Epoch):
            # ---------------------------------------#
            #   如果模型有冻结学习部分
            #   则解冻，并设置参数
            # ---------------------------------------#
            if epoch >= opt.Freeze_Epoch and not UnFreeze_flag and opt.Freeze_Train:
                batch_size = opt.Unfreeze_batch_size

                # -------------------------------------------------------------------#
                #   判断当前batch_size，自适应调整学习率
                # -------------------------------------------------------------------#
                nbs = 16
                lr_limit_max = 5e-4 if opt.optimizer == 'adam' else 5e-2
                lr_limit_min = 3e-4 if opt.optimizer == 'adam' else 5e-4
                Init_lr_fit = min(max(batch_size / nbs * opt.Init_lr, lr_limit_min), lr_limit_max)
                Min_lr_fit = min(max(batch_size / nbs * Min_lr, lr_limit_min * 1e-2), lr_limit_max * 1e-2)
                # ---------------------------------------#
                #   获得学习率下降的公式
                # ---------------------------------------#
                lr_scheduler_func = get_lr_scheduler(opt.lr_decay_type, Init_lr_fit, Min_lr_fit, opt.UnFreeze_Epoch)

                for param in model.backbone.parameters():
                    param.requires_grad = True

                epoch_step = num_train // batch_size
                epoch_step_val = num_val // batch_size

                if epoch_step == 0 or epoch_step_val == 0:
                    raise ValueError("数据集过小，无法继续进行训练，请扩充数据集。")

                gen = DataLoader(train_dataset, shuffle=shuffle, batch_size=batch_size, num_workers=opt.num_workers,
                                 pin_memory=True,
                                 drop_last=True, collate_fn=fcos_dataset_collate)
                gen_val = DataLoader(val_dataset, shuffle=shuffle, batch_size=batch_size, num_workers=opt.num_workers,
                                     pin_memory=True,
                                     drop_last=True, collate_fn=fcos_dataset_collate)

                UnFreeze_flag = True

            if opt.distributed:
                train_sampler.set_epoch(epoch)

            set_optimizer_lr(optimizer, lr_scheduler_func, epoch)

            fit_one_epoch(model_train, model, fcos_loss, loss_history, eval_callback, optimizer, epoch,
                          epoch_step, epoch_step_val, gen, gen_val, opt.UnFreeze_Epoch, opt.cuda, opt.fp16, scaler,
                          opt.save_period,
                          opt.save_dir, local_rank)

            if opt.distributed:
                dist.barrier()

        if local_rank == 0:
            loss_history.writer.close()


def parse_opt():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str, default="", help="initil weights path")
    parser.add_argument("--cfg", type=str, default=ROOT / "configs/fcos.yaml", help="model.yaml path")
    parser.add_argument("--data", type=str, default=ROOT / "data/fcos.yaml", help="dataset.yaml path")
    parser.add_argument("--cuda", type=bool, default=False, help="cuda or cpu")
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--distributed", type=bool, default=False)
    parser.add_argument("--sync_bn", type=bool, default=False)
    parser.add_argument("--fp16", type=bool, default=False)
    parser.add_argument("--input_shape", type=int, default=640)
    parser.add_argument("--Init_Epoch", type=int, default=0, help="start epoch")
    parser.add_argument("--Freeze_Epoch", type=int, default=50, help="freeze epoch")
    parser.add_argument("--Freeze_batch_size", type=int, default=8, help="batch size for freeze epoch")
    parser.add_argument("--UnFreeze_Epoch", type=int, default=100, help="total epoch")
    parser.add_argument("--Unfreeze_batch_size", type=int, default=4, help="batch size for unfreeze epoch")
    parser.add_argument("--Freeze_Train", type=bool, default=True)
    parser.add_argument("--optimizer", type=str, default="adam", help="optimizer Adam or SGD")
    parser.add_argument("--momentum", type=float, default=0.9)
    parser.add_argument("--weight_decay", type=int, default=0)
    parser.add_argument("--lr_decay_type", type=str, default="cos")
    parser.add_argument("--save_period", type=int, default=5, help="save weights for periods")
    parser.add_argument("--save_dir", type=str, default="logs")
    parser.add_argument("--eval_flag", type=bool, default=True)
    parser.add_argument("--eval_period", type=int, default=5)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--Init_lr", type=float, default=3e-4)

    opt = parser.parse_args()
    return opt


if __name__ == "__main__":
    opt = parse_opt()
    run(opt)
