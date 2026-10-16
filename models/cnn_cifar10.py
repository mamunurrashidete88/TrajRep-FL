"""
models/cnn_cifar10.py
=====================

"""

import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# Building block
# ---------------------------------------------------------------------------

class _VGGBlock(nn.Module):
    
    def __init__(self, in_ch: int, out_ch: int, num_convs: int = 2):
        super().__init__()
        layers: list = []
        for i in range(num_convs):
            ch_in  = in_ch if i == 0 else out_ch
            layers += [
                nn.Conv2d(ch_in, out_ch, kernel_size=3, padding=1, bias=False),
                nn.BatchNorm2d(out_ch),
                nn.ReLU(inplace=True),
            ]
        self.convs = nn.Sequential(*layers)

        # Residual shortcut
        if in_ch != out_ch:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_ch, out_ch, kernel_size=1, bias=False),
                nn.BatchNorm2d(out_ch),
            )
        else:
            self.shortcut = nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.convs(x) + self.shortcut(x)


# ---------------------------------------------------------------------------
# Main model
# ---------------------------------------------------------------------------

class CIFAR10_VGG(nn.Module):
    

    def __init__(self, num_classes: int = 10):
        super().__init__()

        self.features = nn.Sequential(
            # ── Block 1: 32×32 → 16×16 ─────────────────────────────────
            _VGGBlock(3,   64,  num_convs=2),
            nn.MaxPool2d(2, 2),
            nn.Dropout2d(p=0.10),

            # ── Block 2: 16×16 → 8×8 ───────────────────────────────────
            _VGGBlock(64,  128, num_convs=2),
            nn.MaxPool2d(2, 2),
            nn.Dropout2d(p=0.15),

            # ── Block 3: 8×8 → 4×4 ─────────────────────────────────────
            _VGGBlock(128, 256, num_convs=3),
            nn.MaxPool2d(2, 2),
            nn.Dropout2d(p=0.20),

            # ── Block 4: 4×4 → 2×2 ─────────────────────────────────────
            _VGGBlock(256, 512, num_convs=3),
            nn.MaxPool2d(2, 2),
            nn.Dropout2d(p=0.25),
        )

        # 512 channels × 2×2 spatial = 2048 features after flatten
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(512 * 2 * 2, 1024),
            nn.BatchNorm1d(1024),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.50),
            nn.Linear(1024, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.50),
            nn.Linear(512, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        return self.classifier(x)

    @property
    def param_count(self) -> int:
        return sum(p.numel() for p in self.parameters())


# ---------------------------------------------------------------------------
# Factory (keeps the same name as the original file)
# ---------------------------------------------------------------------------

def get_cifar10_model(num_classes: int = 10) -> CIFAR10_VGG:
    return CIFAR10_VGG(num_classes=num_classes)
