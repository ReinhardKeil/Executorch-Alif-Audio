"""Pinned Moonshine Micro WordCNN architecture.

Derived from moonshine-ai/moonshine commit
d8d8fa29d05ca67441a7719e70099f9f48917a93, micro/stt-training/stt_training/model.py.
"""

from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F


def _make_divisible(value: int, divisor: int = 8) -> int:
    result = max(divisor, (value + divisor // 2) // divisor * divisor)
    return result + divisor if result < 0.9 * value else result


class ConvBNAct(nn.Sequential):
    def __init__(self, in_channels, out_channels, kernel=3, stride=1, groups=1, act=True):
        padding = (kernel - 1) // 2
        layers = [
            nn.Conv2d(in_channels, out_channels, kernel, stride, padding, groups=groups, bias=False),
            nn.BatchNorm2d(out_channels),
        ]
        if act:
            layers.append(nn.ReLU6(inplace=True))
        super().__init__(*layers)


class InvertedResidual(nn.Module):
    def __init__(self, in_channels, out_channels, stride, expand_ratio):
        super().__init__()
        hidden = in_channels * expand_ratio
        self.use_residual = stride == 1 and in_channels == out_channels
        layers = []
        if expand_ratio != 1:
            layers.append(ConvBNAct(in_channels, hidden, kernel=1))
        layers += [
            ConvBNAct(hidden, hidden, kernel=3, stride=stride, groups=hidden),
            nn.Conv2d(hidden, out_channels, 1, bias=False),
            nn.BatchNorm2d(out_channels),
        ]
        self.block = nn.Sequential(*layers)

    def forward(self, value):
        output = self.block(value)
        return value + output if self.use_residual else output


class WordCNN(nn.Module):
    CONFIG = [
        (1, 16, 1, 1), (4, 24, 2, 2), (4, 32, 3, 2), (4, 64, 3, 2),
        (4, 96, 2, 1), (4, 160, 2, 2), (4, 240, 1, 1),
    ]

    def __init__(self, num_classes, width_mult=1.0, stem_stride=(2, 2), pad_to_odd=True):
        super().__init__()
        self.pad_to_odd = bool(pad_to_odd)
        stem_channels = _make_divisible(int(32 * width_mult))
        self.stem = ConvBNAct(1, stem_channels, stride=stem_stride)
        blocks = []
        in_channels = stem_channels
        for expand, out_channels, count, first_stride in self.CONFIG:
            out_channels = _make_divisible(int(out_channels * width_mult))
            for index in range(count):
                stride = first_stride if index == 0 else 1
                blocks.append(InvertedResidual(in_channels, out_channels, stride, expand))
                in_channels = out_channels
        self.blocks = nn.Sequential(*blocks)
        head_channels = _make_divisible(int(640 * width_mult))
        self.head_conv = ConvBNAct(in_channels, head_channels, kernel=1)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.classifier = nn.Sequential(nn.Dropout(0.2), nn.Linear(head_channels, num_classes))

    def forward(self, value):
        if self.pad_to_odd:
            value = F.pad(value, (0, 1, 0, 1))
        value = self.stem(value)
        value = self.blocks(value)
        value = self.head_conv(value)
        value = self.pool(value).flatten(1)
        return self.classifier(value)
