import torch
import torch.nn as nn
from backbone import build_backbone
from head import build_head
from neck import build_neck


class Model(nn.Module):
    def __init__(self, cfg='fcos.yaml'):
        super(Model, self).__init__()
        # load config
        if type(cfg) is dict:
            self.yaml = cfg
        else:
            import yaml
            with open(cfg, encoding="utf-8", errors="ignore") as f:
                self.yaml = yaml.safe_load(f)
        # build backbone
        backbone_type = self.yaml['backbone']["backbone_name"]
        self.backbone = build_backbone(backbone_type, **self.yaml["backbone"]['backbone_dict'])

        # build neck
        neck_type = self.yaml["neck"]["neck_name"]
        self.neck = build_neck(neck_type, **self.yaml["neck"]["neck_dict"])

        # build head
        head_type = self.yaml["head"]["head_name"]
        self.head = build_head(head_type, **self.yaml["head"]["head_dict"])

    def forward(self, x):
        # --------------------------------------------------- #
        # 80, 80, 512
        # 40, 40, 512
        # 20, 20, 2048
        # --------------------------------------------------- #
        C3, C4, C5 = self.backbone(x)

        # --------------------------------------------------- #
        # 80, 80, 256
        # 40, 40, 256
        # 20, 20, 256
        # 10, 10, 256
        # 5, 5, 256
        # --------------------------------------------------- #
        P3, P4, P5, P6, P7 = self.neck([C3, C4, C5])

        cls_logits, cnt_logits, reg_preds = self.head([P3, P4, P5, P6, P7])
        return [cls_logits, cnt_logits, reg_preds]


if __name__ == "__main__":
    cfg = "../configs/fcos.yaml"
    img = torch.rand(1, 3, 640, 640)
    model = Model(cfg=cfg)
    cls_logits, cnt_logits, reg_preds = model(img)
    print("cls_logits shape", cls_logits[0].shape)
    print("cnt_logits shape", cnt_logits[0].shape)
    print("reg_preds shape", reg_preds[0].shape)
