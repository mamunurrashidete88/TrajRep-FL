
from data.loaders import (
    ShakespeareClientDataset,
    load_shakespeare,
    SHAKESPEARE_VOCAB,
    VOCAB_SIZE,
    CHAR2IDX,
)
NUM_CLASSES = VOCAB_SIZE
SEQ_LEN = 80
__all__ = ["ShakespeareClientDataset", "load_shakespeare", "NUM_CLASSES", "VOCAB_SIZE"]
