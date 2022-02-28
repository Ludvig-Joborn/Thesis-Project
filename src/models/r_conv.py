import torch
from torch import nn

from models.cbam import CBAM


class R_Conv(nn.Module):
    def __init__(
        self, inC, outC, c_ks=(3, 3), c_stride=(1, 1), c_padding="same", dropout=0
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
            nn.ReLU(),
            nn.BatchNorm2d(outC),
            nn.Dropout(p=dropout),
            nn.Conv2d(
                in_channels=outC,
                out_channels=outC,
                kernel_size=c_ks,
                stride=c_stride,
                padding=c_padding,
            ),
            nn.ReLU(),
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
        p_ks=(3, 1),
        p_stride=(2, 1),
        p_padding=0,
        pad_pooling=(0, 0, 1, 0),
        dropout=0,
    ):
        super(R_Conv_cbam_avg_pool, self).__init__()
        self.r_conv = R_Conv(inC=inC, outC=outC, dropout=dropout)
        self.cbam = CBAM(inC=outC)
        self.pad = nn.ConstantPad2d(pad_pooling, 0)
        self.pool = nn.AvgPool2d(kernel_size=p_ks, stride=p_stride, padding=p_padding)

    def __call__(self, input: torch.Tensor):
        conv = self.r_conv(input)
        cbam = self.cbam(conv)
        pad = self.pad(cbam)
        pool = self.pool(pad)
        return pool
