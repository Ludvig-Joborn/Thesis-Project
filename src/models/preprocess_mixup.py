import torch
from torch import nn
from nnAudio.features.mel import MelSpectrogram
import julius

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

    def __call__(self, waveform, lam=1, perm_index=None):
        waveform_ds = self.resampler(waveform)
        mel_spec = self.spec_layer(waveform_ds)
        # Should we apply mixup?
        if lam != 1 and perm_index is not None:
            mel_spec = self.mixup_data(mel_spec, lam, perm_index)
        return torch.unsqueeze(input=mel_spec, dim=1)

    def mixup_data(self, mel_spec, lam, perm_index):
        # Create a mixed input
        mixed_melspec = lam * mel_spec + (1 - lam) * mel_spec[perm_index, :]
        return mixed_melspec
