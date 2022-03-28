import torch
import pandas as pd
from torch.utils.data import Dataset

# Pytorch imports / definitions
import bisect
import warnings
from typing import (
    Iterable,
    Iterator,
    List,
    TypeVar,
)

T_co = TypeVar("T_co", covariant=True)
T = TypeVar("T")

UNTRACABLE_DATAFRAME_PIPES = [
    "batch",  # As it returns DataChunks
    "groupby",  # As it returns DataChunks
    "_dataframes_as_tuples",  # As it unpacks DF
    "trace_as_dataframe",  # As it used to mark DF for tracing
]

# NOTE: Pytorch source code
class IterableDataset(Dataset[T_co]):
    def __iter__(self) -> Iterator[T_co]:
        raise NotImplementedError

    def __add__(self, other: Dataset[T_co]):
        return ChainDataset([self, other])


# NOTE: Pytorch source code
class ChainDataset(IterableDataset):
    def __init__(self, datasets: Iterable[Dataset]) -> None:
        super(ChainDataset, self).__init__()
        self.datasets = datasets

    def __iter__(self):
        for d in self.datasets:
            assert isinstance(
                d, IterableDataset
            ), "ChainDataset only supports IterableDataset"
            for x in d:
                yield x

    def __len__(self):
        total = 0
        for d in self.datasets:
            assert isinstance(
                d, IterableDataset
            ), "ChainDataset only supports IterableDataset"
            total += len(d)
        return total


# NOTE: Pytorch source code (modified)
class ConcatDataset(Dataset[T_co]):
    r"""Dataset as a concatenation of multiple datasets.

    This class is useful to assemble different existing datasets.

    Args:
        datasets (sequence): List of datasets to be concatenated
    """
    datasets: List[Dataset[T_co]]
    cumulative_sizes: List[int]

    @staticmethod
    def cumsum(sequence):
        r, s = [], 0
        for e in sequence:
            l = len(e)
            r.append(l + s)
            s += l
        return r

    def __init__(self, datasets: Iterable[Dataset]) -> None:
        super(ConcatDataset, self).__init__()
        self.datasets = list(datasets)
        assert len(self.datasets) > 0, "datasets should not be an empty iterable"  # type: ignore[arg-type]
        self.sample_rate = datasets[0].get_sample_rate()
        for d in self.datasets:
            assert not isinstance(
                d, IterableDataset
            ), "ConcatDataset does not support IterableDataset"
            assert self.sample_rate == d.get_sample_rate()
        self.annotations = pd.concat(
            [d.get_annotations() for d in self.datasets], ignore_index=True
        )
        # Drop nan rows
        self.annotations = self.annotations.dropna()
        self.annotations = self.annotations.reset_index(drop=True)

        self.cumulative_sizes = self.cumsum(self.datasets)
        self.name = "_".join([str(d) for d in datasets])

    def __len__(self):
        return self.cumulative_sizes[-1]

    def __getitem__(self, idx):
        dataset_idx, sample_idx = self._get_indices(idx)
        return self.datasets[dataset_idx][sample_idx]

    def filename(self, idx: int) -> str:
        dataset_idx, sample_idx = self._get_indices(idx)
        return self.datasets[dataset_idx].filename(sample_idx)

    def get_annotations(self) -> pd.DataFrame:
        return self.annotations

    def get_sample_rate(self):
        return self.sample_rate

    @property
    def cummulative_sizes(self):
        warnings.warn(
            "cummulative_sizes attribute is renamed to " "cumulative_sizes",
            DeprecationWarning,
            stacklevel=2,
        )
        return self.cumulative_sizes

    def __str__(self) -> str:
        return self.name

    def _get_indices(self, idx):
        if idx < 0:
            if -idx > len(self):
                raise ValueError(
                    "absolute value of index should not exceed dataset length"
                )
            idx = len(self) + idx
        dataset_idx = bisect.bisect_right(self.cumulative_sizes, idx)
        if dataset_idx == 0:
            sample_idx = idx
        else:
            sample_idx = idx - self.cumulative_sizes[dataset_idx - 1]
        return dataset_idx, sample_idx
