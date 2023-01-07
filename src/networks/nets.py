# Pure networks without additional functions or any python loops / ifs etc.
# Network has to exportable via ONNX

import torch
import torch.nn as nn
import torch.nn.functional as F


class SEBlock(nn.Module):
    """Squeeze and Excitation Block from https://github.com/moskomule/senet.pytorch/blob/master/senet/se_module.py"""
    def __init__(self, channel, reduction=16):
        super(SEBlock, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // reduction),
            nn.ReLU(inplace=True),
            nn.Linear(channel // reduction, channel),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y.expand_as(x)


class ResBlock(nn.Module):
    """ Residual Block with Skip Connection, just like ResNet. """
    def __init__(self, num_filters, kernel_size):
        super(ResBlock, self).__init__()
        self.layers = nn.Sequential(
            nn.Conv2d(num_filters, num_filters, kernel_size, padding=1),
            nn.BatchNorm2d(num_filters),
            nn.ReLU(inplace=True),
            nn.Conv2d(num_filters, num_filters, kernel_size, padding=1),
            nn.BatchNorm2d(num_filters),
        )

    def forward(self, x):
        out = self.layers(x)
        return F.relu(x + out)


class ResSEBlock(nn.Module):
    """ Residual Block with Skip Connection and Squeeze & Excitation. """
    def __init__(self, num_filters, kernel_size):
        super(ResBlock, self).__init__()
        self.layers = nn.Sequential(
            nn.Conv2d(num_filters, num_filters, kernel_size, padding=1),
            nn.BatchNorm2d(num_filters),
            nn.ReLU(inplace=True),
            nn.Conv2d(num_filters, num_filters, kernel_size, padding=1),
            nn.BatchNorm2d(num_filters),
        )
        self.se = SEBlock(num_filters)

    def forward(self, x):
        out = self.layers(x)
        out = self.se(out)
        return F.relu(x + out)


class ConvResNetwork(nn.Module):
    """ Policy network. Deep convolutional Residual Network """

    def __init__(self, num_filters, kernel_size, use_se, num_res_blocks, size, num_acts):
        super(ConvResNetwork, self).__init__()

        # Layers
        if use_se:
            self.body = nn.Sequential(
                nn.Conv2d(2, num_filters, kernel_size=kernel_size, padding=1),
                nn.BatchNorm2d(num_filters),
                nn.ReLU(inplace=True),
                *[ResSEBlock(num_filters, kernel_size) for _ in range(num_res_blocks)],
            )
        else:
            self.body = nn.Sequential(
                nn.Conv2d(2, num_filters, kernel_size=kernel_size, padding=1),
                nn.BatchNorm2d(num_filters),
                nn.ReLU(inplace=True),
                *[ResBlock(num_filters, kernel_size) for _ in range(num_res_blocks)],
            )

        # Heads
        self.policy = nn.Sequential(
            nn.Conv2d(num_filters, 16, 1),
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True),
            nn.Flatten(1, -1),
        )
        self.policy_head = nn.Linear(size*size*16, num_acts)

        self.value = nn.Sequential(
            nn.Conv2d(num_filters, 16, 1),
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True),
            nn.Flatten(1, -1),
            nn.Linear(size*size*16, 256),
            nn.ReLU(inplace=True),
        )
        self.value_head = nn.Sequential(
            nn.Linear(256, 1),
            nn.Tanh(),
        )

    def forward(self, x):
        x = self.body(x)

        logits = self.policy_head(self.policy(x))
        val = self.value_head(self.value(x)).reshape(-1)
        return logits, val
