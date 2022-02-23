from turtle import forward
import torch
from torch import nn
from nnAudio.features.mel import MelSpectrogram
import julius
from enum import Enum

# User defined imports
from config import PARAMS_TO_MELSPEC, SAMPLE_RATE


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
        p_ks=(3, 1),
        p_stride=(1, 1),
        p_pad=(1, 0),
        act=ACT.GLU,
    ):
        super(ConvBlock, self).__init__()
        self.cnn_layer = nn.Sequential(
            nn.Conv2d(
                in_channels=inC,
                out_channels=outC,
                kernel_size=c_ks,
                stride=c_stride,
                padding=c_padding,
            ),
            nn.AvgPool2d(kernel_size=p_ks, stride=p_stride, padding=p_pad),
            self.act_func(act),
            nn.BatchNorm2d(outC),
        )

    def act_func(self, act=ACT.GLU):
        if act == ACT.GLU:
            return nn.GLU(dim=2)
        elif act == ACT.RELU:
            return nn.ReLU(inplace=True)
        else:
            return None

    def forward(self, input: torch.Tensor):
        return self.cnn_layer(input)


"""
dcase 2021 - place 10
https://dcase.community/documents/challenge2021/technical_reports/DCASE2021_Park_101_t4.pdf
"""


class NeuralNetwork(nn.Module):
    def __init__(self, input_sample_rate: int, output_sample_rate: int = SAMPLE_RATE):
        super(NeuralNetwork, self).__init__()

        self.pre_process = PreProcess(input_sample_rate, output_sample_rate)

        # 8, 1, 128, 431
        self.conv1 = ConvBlock(
            inC=1,
            outC=16,
            c_ks=(3, 3),
            c_stride=(1, 1),
            c_padding="same",
            p_ks=(3, 3),
            p_stride=(1, 1),
            p_pad=1,
            act=ACT.GLU,
        )

        # 8, 16, 64, 431
        self.conv2 = ConvBlock(
            inC=16,
            outC=32,
            c_ks=(3, 3),
            c_stride=(1, 1),
            c_padding="same",
            p_ks=(3, 1),
            p_stride=(1, 1),
            p_pad=(1, 0),
            act=ACT.GLU,
        )

        # 8, 32, 32, 431
        self.conv3 = ConvBlock(
            inC=32,
            outC=64,
            c_ks=(3, 3),
            c_stride=(1, 1),
            c_padding="same",
            p_ks=(3, 1),
            p_stride=(1, 1),
            p_pad=(1, 0),
            act=ACT.GLU,
        )

        # 8, 64, 16, 431
        self.conv4 = ConvBlock(
            inC=64,
            outC=128,
            c_ks=(3, 3),
            c_stride=(1, 1),
            c_padding="same",
            p_ks=(3, 1),
            p_stride=(1, 1),
            p_pad=(1, 0),
            act=ACT.GLU,
        )

        # 8, 128, 8, 431
        self.conv5 = ConvBlock(
            inC=128,
            outC=128,
            c_ks=(3, 3),
            c_stride=(1, 1),
            c_padding="same",
            p_ks=(3, 1),
            p_stride=(1, 1),
            p_pad=(1, 0),
            act=ACT.GLU,
        )

        # 8, 128, 4, 431
        self.conv6 = ConvBlock(
            inC=128,
            outC=128,
            c_ks=(3, 3),
            c_stride=(1, 1),
            c_padding="same",
            p_ks=(3, 1),
            p_stride=(1, 1),
            p_pad=(1, 0),
            act=ACT.GLU,
        )

        # 8, 128, 2, 431
        self.conv7 = ConvBlock(
            inC=128,
            outC=128,
            c_ks=(3, 3),
            c_stride=(1, 1),
            c_padding="same",
            p_ks=(3, 1),
            p_stride=(1, 1),
            p_pad=(1, 0),
            act=ACT.GLU,
        )

        # 8, 128, 1, 431
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
        conv3 = self.conv3.forward(conv2)
        conv4 = self.conv4.forward(conv3)
        conv5 = self.conv5.forward(conv4)
        conv6 = self.conv6.forward(conv5)
        conv7 = self.conv7.forward(conv6)

        last_sq = torch.squeeze(input=conv7, dim=2)
        last_pe = torch.permute(input=last_sq, dims=(0, 2, 1))

        gru_out, h_n = self.gru(last_pe)
        lin = self.lin(gru_out)
        output = self.sigm(lin)

        return torch.permute(input=output, dims=(0, 2, 1))