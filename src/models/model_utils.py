import torch
from typing import Dict, Tuple
from pathlib import Path
from logger import CustomLogger as Logger

# User defined imports
from config import ACT_THRESHOLD
from utils import get_datetime


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


def create_path(
    dir: Path, filename: str = get_datetime(), ending: str = ".pt", best: bool = False
) -> Path:
    # Create directory
    if not dir.exists():
        dir.mkdir(parents=True, exist_ok=True)

    b = "best_" if best else ""
    path = dir / f"{b}{filename}{ending}"
    return path


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
