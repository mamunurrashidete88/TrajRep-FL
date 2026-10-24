

import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# Building block
# ---------------------------------------------------------------------------

class _ResBlock(nn.Module):


    def __init__(self, in_ch: int, out_ch: int, stride: int = 1):
        super().__init__()
        self.bn1   = nn.BatchNorm2d(in_ch)
        self.conv1 = nn.Conv2d(
            in_ch, out_ch, kernel_size=3,
            stride=stride, padding=1, bias=False
        )
        self.bn2   = nn.BatchNorm2d(out_ch)
        self.conv2 = nn.Conv2d(out_ch, out_ch, kernel_size=3, padding=1, bias=False)


        if stride != 1 or in_ch != out_ch:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_ch, out_ch, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_ch),
            )
        else:
            self.shortcut = nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = self.shortcut(x)
        out = self.conv1(F.relu(self.bn1(x)))
        out = self.conv2(F.relu(self.bn2(out)))
        return out + residual


# ---------------------------------------------------------------------------
# Main model
# ---------------------------------------------------------------------------

class FEMNIST_CNN(nn.Module):


    def __init__(self, num_classes: int = 62):
        super().__init__()

        # ── Stem: establish feature maps from grayscale input ─────────────
        self.stem = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
        )

        # ── Stage 1: 28×28 → 14×14 ────────────────────────────────────────
        self.stage1 = nn.Sequential(
            _ResBlock(32, 64),
            _ResBlock(64, 64),
            nn.MaxPool2d(2, 2),
            nn.Dropout2d(p=0.10),
        )

        # ── Stage 2: 14×14 → 7×7 ─────────────────────────────────────────
        self.stage2 = nn.Sequential(
            _ResBlock(64, 128),
            _ResBlock(128, 128),
            nn.MaxPool2d(2, 2),
            nn.Dropout2d(p=0.15),
        )

        # ── Stage 3: 7×7 → 3×3 ───────────────────────────────────────────
        self.stage3 = nn.Sequential(
            _ResBlock(128, 256),
            _ResBlock(256, 256),
            nn.MaxPool2d(2, 2),
            nn.Dropout2d(p=0.20),
        )

        # ── Global average pool: collapses spatial dims → (B, 256) ────────
        self.gap = nn.AdaptiveAvgPool2d(1)

        # ── Classifier head ───────────────────────────────────────────────
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(256, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.50),
            nn.Linear(512, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.30),
            nn.Linear(256, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.stem(x)
        x = self.stage1(x)
        x = self.stage2(x)
        x = self.stage3(x)
        x = self.gap(x)
        return self.classifier(x)

    @property
    def param_count(self) -> int:
        return sum(p.numel() for p in self.parameters())


# ---------------------------------------------------------------------------
# 
# ---------------------------------------------------------------------------

def get_femnist_model(num_classes: int = 62) -> FEMNIST_CNN:
    return FEMNIST_CNN(num_classes=num_classes)
