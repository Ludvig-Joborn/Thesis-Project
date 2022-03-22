import torch
from torch import nn

# User defined imports
from config import SAMPLE_RATE
from models.preprocess import PreProcess
from models.cbam import CBAM


CNN_DROPOUT = 0.1

"""
CHANGES: 
Inspiration from: 
https://dcase.community/documents/challenge2021/technical_reports/DCASE2021_Kim_23_t4.pdf 

Note: Works best with 'batch_size' = 64 (have not tried larger)
* 'batch_size' tests: 8 -> 8:15; 32 -> 7:20; 64 -> 7:05;
"""


class NeuralNetwork(nn.Module):
    def __init__(self, input_sample_rates: set, output_sample_rate: int = SAMPLE_RATE):
        super(NeuralNetwork, self).__init__()

        self.pre_process = PreProcess(input_sample_rates, output_sample_rate)

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
            nn.Dropout(p=CNN_DROPOUT),
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
            nn.Dropout(p=CNN_DROPOUT),
            nn.Conv2d(
                in_channels=chout,
                out_channels=chout,
                kernel_size=kern_size,
                stride=(1, 1),
                padding="same",
            ),
            nn.ReLU(),
            nn.BatchNorm2d(chout),
            nn.Dropout(p=CNN_DROPOUT),
            CBAM(chout),
            nn.ConstantPad2d((0, 0, 1, 0), 0),
            nn.AvgPool2d(kernel_size=(2, 1), stride=(2, 1)),
        )

    def forward(self, waveform: torch.Tensor, sample_rate: torch.Tensor):
        mel_spec = self.pre_process(waveform, sample_rate)

        ############################
        # Stem blocks

        stem1 = self.stem_1_32(mel_spec)
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
