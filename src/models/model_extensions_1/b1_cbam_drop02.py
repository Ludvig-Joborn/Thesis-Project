import torch
from torch import nn
from nnAudio.features.mel import MelSpectrogram
import julius

# User defined imports
from config import PARAMS_TO_MELSPEC, SAMPLE_RATE

"""
CHANGES: 
Inspiration from: 
https://dcase.community/documents/challenge2021/technical_reports/DCASE2021_Kim_23_t4.pdf 

Note: Works best with 'batch_size' = 64 (have not tried larger)
* 'batch_size' tests: 8 -> 8:15; 32 -> 7:20; 64 -> 7:05;
"""


class ChannelAttention(nn.Module):
    def __init__(self, chin, reduction=16):
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)

        self.fc = nn.Sequential(
            nn.Conv2d(
                in_channels=chin,
                out_channels=chin // reduction,
                kernel_size=1,
                bias=False,
            ),
            nn.ReLU(),
            nn.Conv2d(
                in_channels=chin // reduction,
                out_channels=chin,
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
    def __init__(self, chin, reduction=16, c_ks=7):
        super(CBAM, self).__init__()

        self.CA = ChannelAttention(chin, reduction)
        self.SA = SpatialAttention(c_ks)

    def forward(self, x: torch.Tensor):
        residual = x
        out = x * self.CA(x)
        out = out * self.SA(out)
        out += residual
        return out


class NeuralNetwork(nn.Module):
    def __init__(self, input_sample_rate: int, output_sample_rate: int = SAMPLE_RATE):
        super(NeuralNetwork, self).__init__()

        # layer that downsamples the waveform to lower sample rate
        self.resampler = julius.resample.ResampleFrac(
            input_sample_rate, output_sample_rate
        )

        # layer that converts waveforms to log mel spectrograms
        self.spec_layer = MelSpectrogram(**PARAMS_TO_MELSPEC)

        # CNN layers
        self.stem_1_32 = self.stem_block(1, 32, 7)
        self.stem_32_64 = self.stem_block(32, 64, 7)
        self.cnn_64_128 = self.conv_block_x2(64, 128, 3)
        self.cnn_128_1 = self.conv_block_x2(128, 128, 3)
        self.cnn_128_2 = self.conv_block_x2(128, 128, 3)
        self.cnn_128_3 = self.conv_block_x2(128, 128, 3)
        self.cnn_128_4 = self.conv_block_x2(128, 128, 3)

        self.gru = nn.GRU(
            input_size=128,
            hidden_size=128,
            num_layers=1,
            batch_first=True,
            bidirectional=True,
        )

        self.lin = nn.Linear(in_features=256, out_features=1)
        self.sigm = nn.Sigmoid()

    def stem_block(self, chin, chout, kern_size):
        return nn.Sequential(
            nn.Conv2d(
                in_channels=chin,
                out_channels=chout,
                kernel_size=kern_size,
                stride=(1, 1),
                padding="same",
            ),
            nn.GLU(dim=2),
            nn.BatchNorm2d(chout),
            nn.Dropout(p=0.2),
            nn.ConstantPad2d((0, 1, 1, 0), 0),
            nn.AvgPool2d(kernel_size=(2, 2), stride=(2, 1)),
        )

    def conv_block_x2(self, chin, chout, kern_size):
        return nn.Sequential(
            nn.Conv2d(
                in_channels=chin,
                out_channels=chout,
                kernel_size=kern_size,
                stride=(1, 1),
                padding="same",
            ),
            nn.ReLU(),
            nn.BatchNorm2d(chout),
            nn.Dropout(p=0.2),
            nn.Conv2d(
                in_channels=chout,
                out_channels=chout,
                kernel_size=kern_size,
                stride=(1, 1),
                padding="same",
            ),
            nn.ReLU(),
            nn.BatchNorm2d(chout),
            nn.Dropout(p=0.2),
            CBAM(chout),
            nn.ConstantPad2d((0, 0, 1, 0), 0),
            nn.AvgPool2d(kernel_size=(2, 1), stride=(2, 1)),
        )

    def forward(self, waveform: torch.Tensor):
        device = "cuda" if torch.cuda.is_available() else "cpu"
        waveform_ds = self.resampler(waveform)
        mel_spec = self.spec_layer(waveform_ds)
        spec = torch.unsqueeze(input=mel_spec, dim=1)

        ############################
        # Stem blocks

        stem1 = self.stem_1_32(spec)
        stem2 = self.stem_32_64(stem1)

        ############################
        # CNN blocks (Residual?)

        conv1 = self.cnn_64_128(stem2)
        conv2 = self.cnn_128_1(conv1)
        conv3 = self.cnn_128_2(conv2)
        conv4 = self.cnn_128_3(conv3)

        ############################
        # Recurrent block

        rec_1 = torch.squeeze(input=conv4, dim=2)
        rec_2 = torch.permute(input=rec_1, dims=(0, 2, 1))

        gru_out, h_n = self.gru(rec_2)
        lin = self.lin(gru_out)
        output = self.sigm(lin)
        return torch.permute(input=output, dims=(0, 2, 1))
