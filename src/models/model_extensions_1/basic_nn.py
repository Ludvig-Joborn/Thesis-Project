import torch
from torch import nn

# User defined imports
from config import SAMPLE_RATE
from models.preprocess import PreProcess


class NeuralNetwork(nn.Module):
    def __init__(self, input_sample_rates: set, output_sample_rate: int = SAMPLE_RATE):
        super(NeuralNetwork, self).__init__()

        self.pre_process = PreProcess(input_sample_rates, output_sample_rate)

        # CNN layer
        self.conv1 = nn.Conv2d(
            in_channels=1,
            out_channels=16,
            kernel_size=(3, 3),
            stride=(1, 1),
            padding=(1, 1),
        )
        self.pad1 = nn.ConstantPad2d((0, 1, 1, 0), 0)
        self.avg_pool1 = nn.AvgPool2d(kernel_size=(2, 2), stride=(1, 1))
        self.glu1 = nn.GLU(dim=2)
        self.bn1 = nn.BatchNorm2d(16)

        self.conv2 = nn.Conv2d(
            in_channels=16,
            out_channels=32,
            kernel_size=(3, 3),
            stride=(1, 1),
            padding=(1, 1),
        )
        self.pad2 = nn.ConstantPad2d((0, 1, 1, 0), 0)
        self.avg_pool2 = nn.AvgPool2d(kernel_size=(2, 2), stride=(2, 1))
        self.glu2 = nn.GLU(dim=2)
        self.bn2 = nn.BatchNorm2d(32)

        self.conv3 = nn.Conv2d(
            in_channels=32,
            out_channels=32,
            kernel_size=(5, 3),
            stride=(3, 1),
            padding=(2, 1),
        )
        self.pad3 = nn.ConstantPad2d((0, 0, 1, 0), 0)
        self.avg_pool3 = nn.AvgPool2d(kernel_size=(2, 1), stride=(2, 1))
        self.relu3 = nn.ReLU(inplace=True)
        self.bn3 = nn.BatchNorm2d(32)

        self.conv4 = nn.Conv2d(
            in_channels=32,
            out_channels=32,
            kernel_size=(5, 3),
            stride=(3, 1),
            padding=(2, 1),
        )
        self.pad4 = nn.ConstantPad2d((0, 0, 1, 0), 0)
        self.avg_pool4 = nn.AvgPool2d(kernel_size=(2, 1), stride=(2, 1))
        self.relu4 = nn.ReLU(inplace=True)
        self.bn4 = nn.BatchNorm2d(32)

        self.gru = nn.GRU(
            input_size=32,
            hidden_size=32,
            num_layers=1,
            batch_first=True,
            bidirectional=True,
        )

        self.lin = nn.Linear(in_features=64, out_features=1)

        self.sigm = nn.Sigmoid()

    def forward(self, waveform: torch.Tensor, sample_rate: torch.Tensor):
        mel_spec = self.pre_process(waveform, sample_rate)

        conv1 = self.conv1(mel_spec)
        glu1 = self.glu1(conv1)
        pad1 = self.pad1(glu1)
        avg_pool1 = self.avg_pool1(pad1)
        bn1 = self.bn1(avg_pool1)

        ############################

        conv2 = self.conv2(bn1)
        glu2 = self.glu2(conv2)
        pad2 = self.pad2(glu2)
        avg_pool2 = self.avg_pool2(pad2)
        bn2 = self.bn2(avg_pool2)

        ############################

        conv3 = self.conv3(bn2)
        relu3 = self.relu3(conv3)
        pad3 = self.pad3(relu3)
        avg_pool3 = self.avg_pool3(pad3)
        bn3 = self.bn3(avg_pool3)

        ############################

        conv4 = self.conv4(bn3)
        relu4 = self.relu4(conv4)
        pad4 = self.pad4(relu4)
        avg_pool4 = self.avg_pool4(pad4)
        bn4 = self.bn4(avg_pool4)

        ############################

        bn4_sq = torch.squeeze(input=bn4, dim=2)
        bn4_pe = torch.permute(input=bn4_sq, dims=(0, 2, 1))

        gru_out, h_n = self.gru(bn4_pe)
        lin = self.lin(gru_out)
        output = self.sigm(lin)

        return torch.permute(input=output, dims=(0, 2, 1))
