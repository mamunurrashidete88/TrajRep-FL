"""
data/femnist.py
===============
FEMNIST dataset loader and client partitioner.

Wraps load_femnist() from data/loaders.py with a convenient class
interface and dataset-specific constants.

Natural non-IID partitioning: 500 writers, ~423 ± 187 samples each.
Classes: 62 (upper/lower-case letters + digits).
Input: 28×28 greyscale images.
"""
from data.loaders import (
    FEMNISTClientDataset,
    load_femnist,
)

NUM_CLASSES = 62
IMG_CHANNELS = 1
IMG_SIZE = 28

__all__ = ["FEMNISTClientDataset", "load_femnist", "NUM_CLASSES"]
