import torch
from torch import nn

# User defined imports
from models.model_utils import ACT, POOL
from models.cbam import CBAM
from models.model_utils import LS_ReLU, SWISH


class R_Conv(nn.Module):
    def __init__(
        self,
        inC,
        outC,
        c_ks=(3, 3),
        c_stride=(1, 1),
        c_padding="same",
        dropout=0,
        act=ACT.RELU,
    ):
        super(R_Conv, self).__init__()
        self.conv_x2 = nn.Sequential(
            nn.Conv2d(
                in_channels=inC,
                out_channels=outC,
                kernel_size=c_ks,
                stride=c_stride,
                padding=c_padding,
            ),
            self.act_func(act),
            nn.BatchNorm2d(outC),
            nn.Dropout(p=dropout),
            nn.Conv2d(
                in_channels=outC,
                out_channels=outC,
                kernel_size=c_ks,
                stride=c_stride,
                padding=c_padding,
            ),
            self.act_func(act),
            nn.BatchNorm2d(outC),
            nn.Dropout(p=dropout),
        )
        self.res_conv = nn.Conv2d(
            in_channels=inC,
            out_channels=outC,
            kernel_size=c_ks,
            stride=c_stride,
            padding=c_padding,
        )

    def act_func(self, act=ACT.GLU):
        if act == ACT.GLU:
            return nn.GLU(dim=2)
        elif act == ACT.RELU:
            return nn.ReLU()
        elif act == ACT.LS_RELU:
            return LS_ReLU()
        elif act == ACT.SWISH:
            return SWISH()
        elif act == ACT.LS_RELU_TR:
            return LS_ReLU(trainable=True)
        elif act == ACT.SWISH_TR:
            return SWISH(trainable=True)

    def __call__(self, input: torch.Tensor):
        layer1 = self.conv_x2(input)
        layer2 = self.res_conv(input)
        out = layer1 + layer2
        return out


class R_Conv_cbam_avg_pool(nn.Module):
    def __init__(
        self,
        inC,
        outC,
        c_ks=(3, 3),
        c_stride=(1, 1),
        c_padding="same",
        p_ks=(3, 1),
        p_stride=(2, 1),
        p_padding=0,
        pad_pooling=(0, 0, 1, 0),
        dropout=0,
        act=ACT.RELU,
        pool=POOL.AVG,
    ):
        super(R_Conv_cbam_avg_pool, self).__init__()
        self.r_conv = R_Conv(
            inC=inC,
            outC=outC,
            c_ks=c_ks,
            c_stride=c_stride,
            c_padding=c_padding,
            dropout=dropout,
            act=act,
        )
        self.cbam = CBAM(inC=outC)
        self.pad = nn.ConstantPad2d(pad_pooling, 0)
        self.pool = self.pool_func(pool, p_ks, p_stride, p_padding)

    def pool_func(self, pool, p_ks, p_stride, p_pad):
        if pool == POOL.AVG:
            return nn.AvgPool2d(kernel_size=p_ks, stride=p_stride, padding=p_pad)
        elif pool == POOL.MAX:
            return nn.MaxPool2d(kernel_size=p_ks, stride=p_stride, padding=p_pad)
        elif pool == POOL.LP:
            return nn.LPPool2d(norm_type=4, kernel_size=p_ks, stride=p_stride)

    def __call__(self, input: torch.Tensor):
        conv = self.r_conv(input)
        cbam = self.cbam(conv)
        pad = self.pad(cbam)
        pool = self.pool(pad)
        return pool
