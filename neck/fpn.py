import torch
import torch.nn.functional as F
import torch.nn as nn


class FPN(nn.Module):
    def __init__(self, features=256):
        super(FPN, self).__init__()
        self.prj_3 = nn.Conv2d(512, features, kernel_size=1)
        self.prj_4 = nn.Conv2d(1024, features, kernel_size=1)
        self.prj_5 = nn.Conv2d(2048, features, kernel_size=1)

        self.conv_3 = nn.Conv2d(features, features, kernel_size=3, padding=1)
        self.conv_4 = nn.Conv2d(features, features, kernel_size=3, padding=1)
        self.conv_5 = nn.Conv2d(features, features, kernel_size=3, padding=1)

        self.conv_out6 = nn.Conv2d(features, features, kernel_size=3, padding=1, stride=2)
        self.conv_out7 = nn.Conv2d(features, features, kernel_size=3, padding=1, stride=2)

        self.apply(self.init_conv_kaiming)

    def upsamplelike(self, inputs):
        src, target = inputs
        return F.interpolate(src, size=(target.shape[2], target.shape[3]), mode='nearest')

    def init_conv_kaiming(self, module):
        if isinstance(module, nn.Conv2d):
            nn.init.kaiming_uniform_(module.weight, a=1)
            if module.bias is not None:
                nn.init.constant_(module.bias, 0)

    def forward(self, x):
        C3, C4, C5 = x
        # -------------------------------------#
        #   80, 80, 512 -> 80, 80, 256
        #   40, 40, 1024 -> 40, 40, 256
        #   20, 20, 2048 -> 20, 20, 256
        # -------------------------------------#
        P3 = self.prj_3(C3)
        P4 = self.prj_4(C4)
        P5 = self.prj_5(C5)

        # ------------------------------------------------#
        #   20, 20, 256 -> 40, 40, 256 -> 40, 40, 256
        # ------------------------------------------------#
        P4 = P4 + self.upsamplelike([P5, C4])
        # ------------------------------------------------#
        #   40, 40, 256 -> 80, 80, 256 -> 80, 80, 256
        # ------------------------------------------------#
        P3 = P3 + self.upsamplelike([P4, C3])

        # 80, 80, 256
        P3 = self.conv_3(P3)
        # 40, 40, 256
        P4 = self.conv_4(P4)
        # 20, 20, 256
        P5 = self.conv_5(P5)

        # 10, 10, 256
        P6 = self.conv_out6(P5)
        # 5, 5, 256
        P7 = self.conv_out7(F.relu(P6))
        return [P3, P4, P5, P6, P7]
