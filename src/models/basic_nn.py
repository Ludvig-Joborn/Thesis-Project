import os
import torch
from torch import nn
<<<<<<< HEAD
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
=======
from nnAudio.features.mel import MelSpectrogram
import julius

# User defined imports
from config import PARAMS_TO_MELSPEC, SAMPLE_RATE
>>>>>>> d322581... feat: downsampling layer in basic nn

device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f'Using {device} device')

class NeuralNetwork(nn.Module):
    def __init__(self, input_sample_rate: int, output_sample_rate: int = SAMPLE_RATE):
        super(NeuralNetwork, self).__init__()

        # layer that downsamples the waveform to lower sample rate
        self.downsampler = julius.resample.ResampleFrac(
            input_sample_rate, output_sample_rate
        )

        # layer that converts waveforms to mel spectrograms
        self.spec_layer = MelSpectrogram(**PARAMS_TO_MELSPEC)

        self.flatten = nn.Flatten()
        self.linear_relu_stack = nn.Sequential(
            nn.Linear(128*313, 4096),
            nn.ReLU(),
            nn.Linear(4096, 512),
            nn.ReLU(),
            nn.Linear(512, 2),
        )

    def forward(self, waveform: torch.Tensor):
        waveform_ds = self.downsampler(waveform)
        mel_spec = self.spec_layer(waveform_ds)
        z = self.flatten(mel_spec)
        logits = self.linear_relu_stack(z)
        return logits
