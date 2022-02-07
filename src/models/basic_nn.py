import torch
from torch import nn
from nnAudio.features.mel import MelSpectrogram
import julius

# User defined imports
from config import PARAMS_TO_MELSPEC, SAMPLE_RATE


class NeuralNetwork(nn.Module):
    def __init__(self, input_sample_rate: int, output_sample_rate: int = SAMPLE_RATE):
        super(NeuralNetwork, self).__init__()

        self.epsilon = 1e-10

        # layer that downsamples the waveform to lower sample rate
        self.resampler = julius.resample.ResampleFrac(
            input_sample_rate, output_sample_rate
        )

        # layer that converts waveforms to log mel spectrograms
        self.spec_layer = MelSpectrogram(**PARAMS_TO_MELSPEC)

        # Flattening layer
        self.flatten = nn.Flatten()

        # CNN layer
        """"
        self.cnn_layer1 = nn.Sequential(
            nn.Conv2d(
                in_channels=1,
                out_channels=16,
                kernel_size=(2, 2),
                stride=(1, 1),
                padding=1,
            ),
            nn.AvgPool2d(kernel_size=(3, 3), stride=(1, 1), padding=0),
            nn.GLU(dim=2),
            nn.BatchNorm2d(16),
        )
        """
        self.conv1 = nn.Conv2d(
            in_channels=1,
            out_channels=16,
            kernel_size=(3, 3),
            stride=(1, 1),
            padding=(1, 1),
        )
        self.avg_pool1 = nn.AvgPool2d(kernel_size=(3, 3), stride=(1, 1), padding=1)
        self.glu1 = nn.GLU(dim=2)
        self.bn1 = nn.BatchNorm2d(16)

        self.conv2 = nn.Conv2d(
            in_channels=16,
            out_channels=32,
            kernel_size=(3, 3),
            stride=(1, 1),
            padding=(1, 1),
        )
        self.avg_pool2 = nn.AvgPool2d(kernel_size=(3, 3), stride=(1, 1), padding=1)
        self.glu2 = nn.GLU(dim=2)
        self.bn2 = nn.BatchNorm2d(32)

        self.conv3 = nn.Conv2d(
            in_channels=32,
            out_channels=32,
            kernel_size=(5, 3),
            stride=(3, 1),
            padding=(2, 1),
        )
        self.avg_pool3 = nn.AvgPool2d(kernel_size=(3, 1), stride=(2, 1), padding=(1, 0))
        self.relu3 = nn.ReLU(inplace=True)
        self.bn3 = nn.BatchNorm2d(32)

        self.conv4 = nn.Conv2d(
            in_channels=32,
            out_channels=32,
            kernel_size=(5, 3),
            stride=(3, 1),
            padding=(2, 1),
        )
        self.avg_pool4 = nn.AvgPool2d(kernel_size=(3, 1), stride=(2, 1), padding=(1, 0))
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
        # self.lin = nn.Flatten()

        # TODO: try with and without dropout
        # TODO: work with output and see if we can learn

    def forward(self, waveform: torch.Tensor):
        device = "cuda" if torch.cuda.is_available() else "cpu"
        waveform_ds = self.resampler(waveform)
        mel_spec = self.spec_layer(waveform_ds)

        foo = torch.unsqueeze(input=mel_spec, dim=1)
        # print("shape foo:", foo.shape)

        conv1 = self.conv1(foo)
        # print("shape conv:", conv1.shape)

        glu1 = self.glu1(conv1)
        # print("shape glu:", glu1.shape)

        avg_pool1 = self.avg_pool1(glu1)
        # print("shape avg_pool:", avg_pool1.shape)

        bn1 = self.bn1(avg_pool1)
        # print("shape bn:", bn1.shape)

        ############################

        conv2 = self.conv2(bn1)
        # print("shape conv2:", conv2.shape)

        glu2 = self.glu2(conv2)
        # print("shape glu2:", glu2.shape)

        avg_pool2 = self.avg_pool2(glu2)
        # print("shape avg_pool2:", avg_pool2.shape)

        bn2 = self.bn2(avg_pool2)
        # print("shape bn2:", bn2.shape)

        ############################

        conv3 = self.conv3(bn2)
        # print("shape conv3:", conv3.shape)

        relu3 = self.relu3(conv3)
        # print("shape relu3:", relu3.shape)

        avg_pool3 = self.avg_pool3(relu3)
        # print("shape avg_pool3:", avg_pool3.shape)

        bn3 = self.bn3(avg_pool3)
        # print("shape bn3:", bn3.shape)

        ############################

        conv4 = self.conv4(bn3)
        # print("shape conv4:", conv4.shape)

        relu4 = self.relu4(conv4)
        # print("shape relu4:", relu4.shape)

        avg_pool4 = self.avg_pool4(relu4)
        # print("shape avg_pool4:", avg_pool4.shape)

        bn4 = self.bn4(avg_pool4)
        # print("shape bn4:", bn4.shape)

        ########

        bn4_sq = torch.squeeze(input=bn4, dim=2)
        bn4_pe = torch.permute(input=bn4_sq, dims=(0, 2, 1))

        gru_out, h_n = self.gru(bn4_pe)
        # print("shape gru:", gru_out.shape)

        lin = self.lin(gru_out)
        # print("shape lin:", lin.shape)

        return torch.permute(input=lin, dims=(0, 2, 1))
