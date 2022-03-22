import torch
from torch import nn

# User defined imports
from config import SAMPLE_RATE
from models.preprocess import PreProcess
from models.r_conv import R_Conv_cbam_avg_pool
from models.conv_block import ConvBlock
from models.model_utils import ACT, POOL


"""
dcase 2021 - place 10 - With CBAM and residual conv. 
https://dcase.community/documents/challenge2021/technical_reports/DCASE2021_Park_101_t4.pdf
"""


class NeuralNetwork(nn.Module):
    def __init__(self, input_sample_rates: set, output_sample_rate: int = SAMPLE_RATE):
        super(NeuralNetwork, self).__init__()

        self.pre_process = PreProcess(input_sample_rates, output_sample_rate)

        # Batch Size, Kernels, Mel-bins, Frames
        # 8/32, 1, 128, 431/157
        self.conv1 = ConvBlock(
            inC=1,
            outC=16,
            c_ks=(3, 3),
            c_stride=(1, 1),
            c_padding="same",
            pool=POOL.AVG,
            p_ks=(2, 2),
            p_stride=(1, 1),
            p_pad=0,
            act=ACT.GLU,
            pad_pooling=(0, 1, 1, 0),
        )

        # 8/32, 16, 64, 431/157
        self.conv2 = ConvBlock(
            inC=16,
            outC=32,
            c_ks=(3, 3),
            c_stride=(1, 1),
            c_padding="same",
            pool=POOL.AVG,
            p_ks=(2, 2),
            p_stride=(1, 1),
            p_pad=0,
            act=ACT.GLU,
            pad_pooling=(0, 1, 1, 0),
        )

        # 8/32, 32, 32, 431/157
        self.r_conv_cbam3 = R_Conv_cbam_avg_pool(
            inC=32,
            outC=64,
            c_ks=(3, 3),
            c_stride=(1, 1),
            c_padding="same",
            p_ks=(2, 1),
            p_stride=(2, 1),
            p_padding=0,
            pad_pooling=(0, 0, 1, 0),
            dropout=0,
            act=ACT.RELU,
            pool=POOL.AVG,
        )

        # 8/32, 64, 16, 431/157
        self.r_conv_cbam4 = R_Conv_cbam_avg_pool(
            inC=64,
            outC=128,
            c_ks=(3, 3),
            c_stride=(1, 1),
            c_padding="same",
            p_ks=(2, 1),
            p_stride=(2, 1),
            p_padding=0,
            pad_pooling=(0, 0, 1, 0),
            dropout=0,
            act=ACT.RELU,
            pool=POOL.AVG,
        )

        # 8/32, 128, 8, 431/157
        self.r_conv_cbam5 = R_Conv_cbam_avg_pool(
            inC=128,
            outC=128,
            c_ks=(3, 3),
            c_stride=(1, 1),
            c_padding="same",
            p_ks=(2, 1),
            p_stride=(2, 1),
            p_padding=0,
            pad_pooling=(0, 0, 1, 0),
            dropout=0,
            act=ACT.RELU,
            pool=POOL.AVG,
        )

        # 8/32, 128, 4, 431/157
        self.r_conv_cbam6 = R_Conv_cbam_avg_pool(
            inC=128,
            outC=128,
            c_ks=(3, 3),
            c_stride=(1, 1),
            c_padding="same",
            p_ks=(2, 1),
            p_stride=(2, 1),
            p_padding=0,
            pad_pooling=(0, 0, 1, 0),
            dropout=0,
            act=ACT.RELU,
            pool=POOL.AVG,
        )

        # 8/32, 128, 2, 431/157
        self.r_conv_cbam7 = R_Conv_cbam_avg_pool(
            inC=128,
            outC=128,
            c_ks=(3, 3),
            c_stride=(1, 1),
            c_padding="same",
            p_ks=(2, 1),
            p_stride=(2, 1),
            p_padding=0,
            pad_pooling=(0, 0, 1, 0),
            dropout=0,
            act=ACT.RELU,
            pool=POOL.AVG,
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

    def forward(self, waveform: torch.Tensor, sample_rate: torch.Tensor):
        mel_spec = self.pre_process(waveform, sample_rate)

        conv1 = self.conv1(mel_spec)
        conv2 = self.conv2(conv1)
        rconv3 = self.r_conv_cbam3(conv2)
        rconv4 = self.r_conv_cbam4(rconv3)
        rconv5 = self.r_conv_cbam5(rconv4)
        rconv6 = self.r_conv_cbam6(rconv5)
        rconv7 = self.r_conv_cbam7(rconv6)

        last_sq = torch.squeeze(input=rconv7, dim=2)
        last_pe = torch.permute(input=last_sq, dims=(0, 2, 1))

        gru_out, h_n = self.gru(last_pe)
        lin = self.lin(gru_out)
        output = self.sigm(lin)

        return torch.permute(input=output, dims=(0, 2, 1))
