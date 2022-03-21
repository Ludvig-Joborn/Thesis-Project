import torch
from torch import nn
from typing import Dict, Tuple
from pathlib import Path
from enum import Enum


def nr_parameters(model) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def activation(output: torch.Tensor, act_threshold: float) -> torch.Tensor:
    """
    Activation function that sqaushes the output to a value of 0 or 1
    depending on the threshold.
    """
    return (output > act_threshold).float()


def update_acc(
    predictions: torch.Tensor, labels: torch.Tensor, total: int
) -> Tuple[torch.Tensor, int]:
    """
    Helps track the number of correct and total prediction used to calculate accuracy.
    """
    return (predictions.eq(labels).sum(), total + torch.numel(labels))


def save_model(
    state: Dict,
    save_best: bool,
    checkpoint_path: Path,
    best_model_path: Path = None,
):
    """
    The state dictionary is saved to a checkpoint file and has the following format::

    ```
    state = {
        'epoch': epoch,
        'state_dict': model.state_dict(),
        'optimizer': optimizer.state_dict(),
        ...
    }
    ```

    If the model is the best so far, also save it to the best_model_path.
    """
    # Save a checkpoint state
    torch.save(state, checkpoint_path)

    # Save best model so far
    if save_best:
        torch.save(state, best_model_path)


def load_model(model_path: Path) -> Dict:
    """
    Loads the model state.
    """
    if model_path.exists():
        state = torch.load(model_path)
        return state
    else:
        exit(f"Failed to load model: {model_path}")


### Convblock with GLU or ReLU ###
class ACT(Enum):
    RELU = 0
    GLU = 1
    LS_RELU = 2
    SWISH = 3

    # With Trainable Parameters
    LS_RELU_TR = 4
    SWISH_TR = 5


class POOL(Enum):
    AVG = 0
    MAX = 1
    LP = 2


"""
LS ReLU Inpired by: 
https://www.mdpi.com/2076-3417/10/5/1897
"""


class LS_ReLU(nn.Module):
    def __init__(self, trainable=False):
        super(LS_ReLU, self).__init__()

        self.upper_threshold = 10
        self.alpha = 2.0

        if trainable:
            self.alpha = nn.Parameter(torch.tensor(1.0))
            self.alpha.requires_grad = True

    def get_alpha(self):
        return self.alpha

    def forward(self, x):
        return self._ls_relu(x)

    def _ls_relu(self, x):
        inds_x0 = x <= 0
        inds_x0_thr = torch.mul((self.upper_threshold >= x), (x > 0))
        inds_thr = x > self.upper_threshold

        x[inds_x0] = torch.div(x[inds_x0], torch.add(torch.abs(x[inds_x0]), 1))
        x[inds_x0_thr] = torch.max(x[inds_x0_thr])

        log_ = torch.log(torch.add(torch.mul(x[inds_thr], self.alpha), 1))
        x[inds_thr] = torch.add(log_, torch.abs(torch.sub(log_, self.upper_threshold)))

        return x

        if x <= 0:
            d = torch.add(torch.abs(x), 1)
            return torch.div(x, d)
        elif self.upper_threshold >= x > 0:
            return torch.max(x)
        elif x > self.upper_threshold:
            inside = torch.add(torch.mul(x, self.alpha), 1)
            l = torch.log(inside)
            a = torch.abs(torch.sub(l, self.upper_threshold))
            return torch.add(l, a)


"""
Swish inspired by:
https://openreview.net/pdf?id=SkBYYyZRZ
"""


class SWISH(nn.Module):
    def __init__(self, trainable=False):
        super(SWISH, self).__init__()

        self.beta = 1.0
        self.sigmoid = nn.Sigmoid()
        if trainable:
            self.beta = nn.Parameter(torch.tensor(1.0))
            self.beta.requires_grad = True

    def get_beta(self):
        return self.beta

    def forward(self, x):
        # return x * self.sigmoid(x * self.beta)
        return torch.mul(x, self.sigmoid(torch.mul(x, self.beta)))
