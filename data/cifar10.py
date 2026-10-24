"""
data/cifar10.py
===============
CIFAR-10 dataset loader with Dirichlet non-IID partitioning (alpha=0.3).
Classes: 10.  Input: 32×32 RGB images.
"""
from data.loaders import load_cifar10
NUM_CLASSES = 10
IMG_CHANNELS = 3
IMG_SIZE = 32
__all__ = ["load_cifar10", "NUM_CLASSES"]
