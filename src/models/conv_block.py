import torch
from torch import nn

# User defined imports
from models.model_utils import ACT, POOL


class ConvBlock(nn.Module):
    def __init__(
        self,
        inC,
        outC,
        c_ks=(3, 3),
        c_stride=(1, 1),
        c_padding="same",
        pool=POOL.AVG,
        p_ks=(2, 2),
        p_stride=(1, 1),
        p_pad=0,
        act=ACT.GLU,
        pad_pooling=(0, 1, 1, 0),
        dropout=0,
    ):
        super(ConvBlock, self).__init__()

        self.conv = nn.Conv2d(
            in_channels=inC,
            out_channels=outC,
            kernel_size=c_ks,
            stride=c_stride,
            padding=c_padding,
        )
        self.drop = nn.Dropout(p=dropout)
        self.pad = nn.ConstantPad2d(pad_pooling, 0)
        self.pool = self.pool_func(pool, p_ks, p_stride, p_pad)
        self.act = self.act_func(act)
        self.bn = nn.BatchNorm2d(outC)

    def pool_func(self, pool, p_ks, p_stride, p_pad):
        if pool == POOL.AVG:
            return nn.AvgPool2d(kernel_size=p_ks, stride=p_stride, padding=p_pad)
        elif pool == POOL.MAX:
            return nn.MaxPool2d(kernel_size=p_ks, stride=p_stride, padding=p_pad)
        elif pool == POOL.MAX:
            return nn.LPPool2d(norm_type=4, kernel_size=p_ks, stride=p_stride)

    def act_func(self, act=ACT.GLU):
        if act == ACT.GLU:
            return nn.GLU(dim=2)
        elif act == ACT.RELU:
            return nn.ReLU()

    def __call__(self, input: torch.Tensor):
        conv = self.conv(input)
        drop = self.drop(conv)
        pad = self.pad(drop)
        pool = self.pool(pad)
        act = self.act(pool)
        bn = self.bn(act)
        return bn
