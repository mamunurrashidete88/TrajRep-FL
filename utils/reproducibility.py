

from __future__ import annotations

import os
import platform
import random

import numpy as np
import torch


def set_all_seeds(seed: int):

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def get_device(prefer_gpu: bool = True) -> torch.device:
    if prefer_gpu and torch.cuda.is_available():
        return torch.device("cuda")
    if prefer_gpu and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def print_environment():
    print(f"Platform:    {platform.platform()}")
    print(f"Python:      {platform.python_version()}")
    print(f"PyTorch:     {torch.__version__}")
    print(f"CUDA avail:  {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"CUDA device: {torch.cuda.get_device_name(0)}")


# Alias expected by experiments/run_experiment.py
def seed_everything(seed: int):
    """Alias for set_all_seeds."""
    set_all_seeds(seed)
