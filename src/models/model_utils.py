import torch
from typing import Tuple

# User defined imports
from config import ACT_THRESHOLD


def nr_parameters(model) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def activation(output: torch.Tensor) -> torch.Tensor:
    """
    Activation function that sqaushes the output to a value of 0 or 1
    depending on the threshold.
    """
    return (output > ACT_THRESHOLD).float()


def update_acc(
    predictions: torch.Tensor, labels: torch.Tensor, total: int
) -> Tuple[torch.Tensor, int]:
    """
    Helps track the number of correct and total prediction used to calculate accuracy.
    """
    return (predictions.eq(labels).sum(), total + torch.numel(labels))
