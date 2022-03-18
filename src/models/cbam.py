import torch
from torch import nn


"""
Channel and Spatial-Attention (CBAM) taken (and modified) from:
https://github.com/xmu-xiaoma666/External-Attention-pytorch/blob/master/model/attention/CBAM.py
"""


class ChannelAttention(nn.Module):
    def __init__(self, inC, reduction=16):
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)

        self.fc = nn.Sequential(
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
        avg_out = self.fc(self.avg_pool(x))
        max_out = self.fc(self.max_pool(x))
        out = avg_out + max_out
        return self.sigmoid(out)


class SpatialAttention(nn.Module):
    def __init__(self, c_ks=7):
        super(SpatialAttention, self).__init__()

        self.conv1 = nn.Conv2d(
            in_channels=2,
            out_channels=1,
            kernel_size=c_ks,
            padding=c_ks // 2,
            bias=False,
        )
        self.sigmoid = nn.Sigmoid()

    def __call__(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        res = torch.cat([avg_out, max_out], dim=1)
        out = self.conv1(res)
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
