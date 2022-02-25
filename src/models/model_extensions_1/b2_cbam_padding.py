import torch
from torch import nn
from nnAudio.features.mel import MelSpectrogram
import julius
from enum import Enum

# User defined imports
from config import PARAMS_TO_MELSPEC, SAMPLE_RATE


### Melspectrogram ###
class PreProcess(nn.Module):
    def __init__(self, input_sample_rate: int, output_sample_rate: int = SAMPLE_RATE):
        super(PreProcess, self).__init__()

        # layer that downsamples the waveform to lower sample rate
        self.resampler = julius.resample.ResampleFrac(
            input_sample_rate, output_sample_rate
        )

        # layer that converts waveforms to log mel spectrograms
        self.spec_layer = MelSpectrogram(**PARAMS_TO_MELSPEC)

    def forward(self, waveform):
        waveform_ds = self.resampler(waveform)
        mel_spec = self.spec_layer(waveform_ds)
        return torch.unsqueeze(input=mel_spec, dim=1)


### Residual Conv block with CBAM ###
"""
Channel and Spatial-Attention (CBAM) taken from:
https://github.com/luuuyi/CBAM.PyTorch/blob/master/model/resnet_cbam.py
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

    def forward(self, x):
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

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        x = torch.cat([avg_out, max_out], dim=1)
        x = self.conv1(x)
        return self.sigmoid(x)


class CBAM(nn.Module):
    def __init__(self, inC, reduction=16, c_ks=7):
        super(CBAM, self).__init__()

        self.CA = ChannelAttention(inC, reduction)
        self.SA = SpatialAttention(c_ks)

    def forward(self, x: torch.Tensor):
        residual = x
        out = x * self.CA(x)
        out = out * self.SA(out)
        out += residual
        return out


class R_Conv(nn.Module):
    def __init__(self, inC, outC, c_ks=(3, 3), c_stride=(1, 1), c_padding="same"):
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
            nn.Conv2d(
                in_channels=outC,
                out_channels=outC,
                kernel_size=c_ks,
                stride=c_stride,
                padding=c_padding,
            ),
            nn.ReLU(),
            nn.BatchNorm2d(outC),
        )
        self.res_conv = nn.Conv2d(
            in_channels=inC,
            out_channels=outC,
            kernel_size=c_ks,
            stride=c_stride,
            padding=c_padding,
        )

    def forward(self, input: torch.Tensor):
        layer1 = self.conv_x2(input)
        layer2 = self.res_conv(input)
        out = layer1 + layer2
        return out


class R_Conv_cbam_avg_pool(nn.Module):
    def __init__(
        self, inC, outC, p_ks=(3, 1), p_stride=(2, 1), p_padding=0, pad_pooling=False
    ):
        super(R_Conv_cbam_avg_pool, self).__init__()
        self.pad_pooling = pad_pooling

        self.r_conv = R_Conv(inC=inC, outC=outC)
        self.cbam = CBAM(inC=outC)
        self.pad = nn.ConstantPad1d((0, 0, 1, 0), 0)
        self.pool = nn.AvgPool2d(kernel_size=p_ks, stride=p_stride, padding=p_padding)

    def forward(self, input: torch.Tensor):
        conv = self.r_conv(input)
        cbam = self.cbam(conv)

        if self.pad_pooling:
            pad = self.pad(cbam)
            pool = self.pool(pad)
        else:
            pool = self.pool(pad)

        return pool


### Convblock with GLU or ReLU ###
class ACT(Enum):
    RELU = 0
    GLU = 1


class ConvBlock(nn.Module):
    def __init__(
        self,
        inC,
        outC,
        c_ks=(3, 3),
        c_stride=(1, 1),
        c_padding="same",
        p_ks=(2, 2),
        p_stride=(1, 1),
        p_pad=0,
        act=ACT.GLU,
        pad_pooling=False,
    ):
        super(ConvBlock, self).__init__()
        self.pad_pooling = pad_pooling

        self.conv = nn.Conv2d(
            in_channels=inC,
            out_channels=outC,
            kernel_size=c_ks,
            stride=c_stride,
            padding=c_padding,
        )
        self.pad = nn.ConstantPad2d((0, 1, 1, 0), 0)
        self.pool = nn.AvgPool2d(kernel_size=p_ks, stride=p_stride, padding=p_pad)
        self.act = self.act_func(act)
        self.bn = nn.BatchNorm2d(outC)

    def act_func(self, act=ACT.GLU):
        if act == ACT.GLU:
            return nn.GLU(dim=2)
        elif act == ACT.RELU:
            return nn.ReLU(inplace=True)
        else:
            return None

    def forward(self, input: torch.Tensor):
        conv = self.conv(input)

        if self.pad_pooling:
            pad = self.pad(conv)
            pool = self.pool(pad)
        else:
            pool = self.pool(conv)

        act = self.act(pool)
        bn = self.bn(act)
        return bn


"""
dcase 2021 - place 10 - With CBAM and residual conv. 
https://dcase.community/documents/challenge2021/technical_reports/DCASE2021_Park_101_t4.pdf
"""


class NeuralNetwork(nn.Module):
    def __init__(self, input_sample_rate: int, output_sample_rate: int = SAMPLE_RATE):
        super(NeuralNetwork, self).__init__()

        self.pre_process = PreProcess(input_sample_rate, output_sample_rate)

        # Batch Size, Kernels, Mel-bins, Frames
        # 8/32, 1, 128, 431/157
        self.conv1 = ConvBlock(
            inC=1,
            outC=16,
            c_ks=(3, 3),
            c_stride=(1, 1),
            c_padding="same",
            p_ks=(2, 2),
            p_stride=(1, 1),
            p_pad=0,
            act=ACT.GLU,
            pad_pooling=True,
        )

        # 8/32, 16, 64, 431/157
        self.conv2 = ConvBlock(
            inC=16,
            outC=32,
            c_ks=(3, 3),
            c_stride=(1, 1),
            c_padding="same",
            p_ks=(2, 2),
            p_stride=(1, 1),
            p_pad=0,
            act=ACT.GLU,
            pad_pooling=True,
        )

        # 8/32, 32, 32, 431/157
        self.r_conv_cbam3 = R_Conv_cbam_avg_pool(
            inC=32,
            outC=64,
            p_ks=(2, 1),
            p_stride=(2, 1),
            p_padding=0,
            pad_pooling=True,
        )

        # 8/32, 64, 16, 431/157
        self.r_conv_cbam4 = R_Conv_cbam_avg_pool(
            inC=64,
            outC=128,
            p_ks=(2, 1),
            p_stride=(2, 1),
            p_padding=0,
            pad_pooling=True,
        )

        # 8/32, 128, 8, 431/157
        self.r_conv_cbam5 = R_Conv_cbam_avg_pool(
            inC=128,
            outC=128,
            p_ks=(2, 1),
            p_stride=(2, 1),
            p_padding=0,
            pad_pooling=True,
        )

        # 8/32, 128, 4, 431/157
        self.r_conv_cbam6 = R_Conv_cbam_avg_pool(
            inC=128,
            outC=128,
            p_ks=(2, 1),
            p_stride=(2, 1),
            p_padding=0,
            pad_pooling=True,
        )

        # 8/32, 128, 2, 431/157
        self.r_conv_cbam7 = R_Conv_cbam_avg_pool(
            inC=128,
            outC=128,
            p_ks=(2, 1),
            p_stride=(2, 1),
            p_padding=0,
            pad_pooling=True,
        )

        # 8/32, 128, 1, 431/157
        self.gru = nn.GRU(
            input_size=128,
            hidden_size=128,
            num_layers=1,
            batch_first=True,
            bidirectional=True,
        )

        self.lin = nn.Linear(in_features=256, out_features=1)

        self.sigm = nn.Sigmoid()

    def forward(self, waveform: torch.Tensor):
        mel_spec = self.pre_process.forward(waveform)

        conv1 = self.conv1.forward(mel_spec)
        conv2 = self.conv2.forward(conv1)
        rconv3 = self.r_conv_cbam3.forward(conv2)
        rconv4 = self.r_conv_cbam4.forward(rconv3)
        rconv5 = self.r_conv_cbam5.forward(rconv4)
        rconv6 = self.r_conv_cbam6.forward(rconv5)
        rconv7 = self.r_conv_cbam7.forward(rconv6)

        last_sq = torch.squeeze(input=rconv7, dim=2)
        last_pe = torch.permute(input=last_sq, dims=(0, 2, 1))

        gru_out, h_n = self.gru(last_pe)
        lin = self.lin(gru_out)
        output = self.sigm(lin)

        return torch.permute(input=output, dims=(0, 2, 1))
