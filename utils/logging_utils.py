"""
utils/logging_utils.py
======================
"""

from __future__ import annotations

import logging
import os
import sys
from datetime import datetime
from typing import Optional


def setup_logger(
    name: str = "trajrep_fl",
    level: int = logging.INFO,
    log_file: Optional[str] = None,
) -> logging.Logger:
    """Configure and return a named logger."""
    logger = logging.getLogger(name)
    logger.setLevel(level)

    if logger.handlers:
        return logger

    fmt = logging.Formatter(
        "%(asctime)s | %(name)s | %(levelname)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    ch = logging.StreamHandler(sys.stdout)
    ch.setFormatter(fmt)
    logger.addHandler(ch)

    if log_file:
        os.makedirs(os.path.dirname(log_file) or ".", exist_ok=True)
        fh = logging.FileHandler(log_file)
        fh.setFormatter(fmt)
        logger.addHandler(fh)

    return logger


def get_experiment_log_path(
    output_dir: str, defense: str, dataset: str, attack: str
) -> str:
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    return os.path.join(output_dir, "logs", f"{defense}_{dataset}_{attack}_{ts}.log")


# Alias expected by experiments/run_experiment.py
def setup_logging(log_path: str = None, name: str = "trajrep_fl"):
    """Wrapper alias for setup_logger."""
    import logging, os
    logger = logging.getLogger(name)
    if not logger.handlers:
        logger.setLevel(logging.INFO)
        fmt = logging.Formatter("%(asctime)s [%(levelname)s] %(message)s")
        sh = logging.StreamHandler()
        sh.setFormatter(fmt)
        logger.addHandler(sh)
        if log_path:
            os.makedirs(os.path.dirname(log_path), exist_ok=True)
            fh = logging.FileHandler(log_path)
            fh.setFormatter(fmt)
            logger.addHandler(fh)
    return logger
