import torch
from torch import nn
from nnAudio.features.mel import MelSpectrogram
import julius

# User defined imports
from config import PARAMS_TO_MELSPEC, SAMPLE_RATE


class NeuralNetwork(nn.Module):
    def __init__(self, input_sample_rate: int, output_sample_rate: int = SAMPLE_RATE):
        super(NeuralNetwork, self).__init__()

        # layer that downsamples the waveform to lower sample rate
        self.resampler = julius.resample.ResampleFrac(
            input_sample_rate, output_sample_rate
        )

        # layer that converts waveforms to mel spectrograms
        self.spec_layer = MelSpectrogram(**PARAMS_TO_MELSPEC)

        self.flatten = nn.Flatten()
        self.linear_relu_stack = nn.Sequential(
            nn.Linear(128 * 313, 4096),
            nn.ReLU(),
            nn.Linear(4096, 512),
            nn.ReLU(),
            nn.Linear(512, 2),
        )

    def forward(self, waveform: torch.Tensor):
        waveform_ds = self.resampler(waveform)
        mel_spec = self.spec_layer(waveform_ds)
        z = self.flatten(mel_spec)
        logits = self.linear_relu_stack(z)
        return logits
