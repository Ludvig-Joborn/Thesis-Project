import torch
from torch import nn
from nnAudio.features.mel import MelSpectrogram
import julius
from collections import OrderedDict

# User defined imports
from config import PARAMS_TO_MELSPEC, SAMPLE_RATE


class PreProcessMix(nn.Module):
    def __init__(self, input_sample_rates: set, output_sample_rate: int = SAMPLE_RATE):
        super(PreProcessMix, self).__init__()

        # Dict containing 'sample_rate: ResampleFrac' where each ResampleFrac
        # has its own initialization for a specific input sample_rate
        self.sr_resample_table = OrderedDict()
        for sr in input_sample_rates:
            self.sr_resample_table[sr] = julius.resample.ResampleFrac(
                sr, output_sample_rate
            ).to("cuda")

        # layer that converts waveforms to log mel spectrograms
        self.spec_layer = MelSpectrogram(**PARAMS_TO_MELSPEC)

    def __call__(
        self,
        waveform: torch.Tensor,
        sample_rate: torch.Tensor,
        lam: int = 1,
        perm_index: torch.Tensor = None,
    ):

        first_sample_rate = sample_rate[0].tolist()  # CPU operation

        # Check that all samples rates are equal
        if (sample_rate != sample_rate[0]).any():
            raise ValueError("Conflicting sample rates within the dataset")

        if not first_sample_rate in self.sr_resample_table.keys():
            raise ValueError(
                f"Uninitialized sample rate {first_sample_rate} found in dataset"
            )

        waveform_ds = self.sr_resample_table[first_sample_rate](waveform)

        mel_spec = self.spec_layer(waveform_ds)
        # Should we apply mixup?
        if lam != 1 and perm_index is not None:
            mel_spec = self.mixup_data(mel_spec, lam, perm_index)
        return torch.unsqueeze(input=mel_spec, dim=1)

    def mixup_data(
        self, mel_spec: torch.Tensor, lam: int, perm_index: torch.Tensor
    ) -> torch.Tensor:
        # Create a mixed input
        mixed_melspec = lam * mel_spec + (1 - lam) * mel_spec[perm_index, :]
        return mixed_melspec
