import torch
from torch import nn


"""
Channel and Spatial-Attention (CBAM) taken (and modified) from:
https://github.com/xmu-xiaoma666/External-Attention-pytorch/blob/master/model/attention/CBAM.py
"""


class ChannelAttention(nn.Module):
    def __init__(self, inC, reduction=16):
        super(ChannelAttention, self).__init__()
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.avg_pool = nn.AdaptiveAvgPool2d(1)

        self.se = nn.Sequential(
            nn.Conv2d(
                in_channels=inC,
                out_channels=inC // reduction,
                kernel_size=1,
                bias=False,
            ),
            nn.ReLU(),
            nn.Conv2d(
                in_channels=inC // reduction,
                out_channels=inC,
                kernel_size=1,
                bias=False,
            ),
        )
        self.sigmoid = nn.Sigmoid()

    def __call__(self, x):
        max_res = self.se(self.max_pool(x))
        avg_res = self.se(self.avg_pool(x))
        return self.sigmoid(max_res + avg_res)


class SpatialAttention(nn.Module):
    def __init__(self, c_ks=7):
        super(SpatialAttention, self).__init__()

        self.conv = nn.Conv2d(
            in_channels=2, out_channels=1, kernel_size=c_ks, padding=c_ks // 2
        )
        self.sigmoid = nn.Sigmoid()

    def __call__(self, x):
        max_res, _ = torch.max(x, dim=1, keepdim=True)
        avg_res = torch.mean(x, dim=1, keepdim=True)
        res = torch.cat([max_res, avg_res], dim=1)
        out = self.conv(res)
        return self.sigmoid(out)


class CBAM(nn.Module):
    def __init__(self, inC, reduction=16, c_ks=7):
        super(CBAM, self).__init__()

        self.CA = ChannelAttention(inC, reduction)
        self.SA = SpatialAttention(c_ks)

    def __call__(self, x: torch.Tensor):
        residual = x
        out = x * self.CA(x)
        out = out * self.SA(out)
        out += residual
        return out
