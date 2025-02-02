import torch
from torch import nn

class InvertedResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride, expansion_factor):
        super().__init__()
        hidden_dim = in_channels * expansion_factor
        self.skip_connection = (stride == 1) and (in_channels == out_channels)

        self.expand = nn.Sequential(
            nn.Conv2d(in_channels, hidden_dim, kernel_size=1, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(hidden_dim),
            nn.ReLU6(inplace=True)
        )

        self.depthwise = nn.Sequential(
            nn.Conv2d(hidden_dim, hidden_dim, 3, stride, 1, groups=hidden_dim, bias=False),
            nn.BatchNorm2d(hidden_dim),
            nn.ReLU6(inplace=True)
        )

        self.out = nn.Sequential(
            nn.Conv2d(hidden_dim, out_channels, 1, 1, 0, bias=False),
            nn.BatchNorm2d(out_channels)
        )

    def forward(self, x):
        initial = x

        out = self.expand(x)
        out = self.depthwise(x)
        out = self.out(x)

        if self.skip_connection:
            out += initial
        
        return out

class SSDhead(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.ssdhead = nn.Sequential(
            nn.Conv2d(in_channels, 1280, 1, 1, bias=False),
            nn.BatchNorm2d(1280),
            nn.ReLU6(inplace=True),

            nn.Conv2d(1280, 512, 1, 1, bias=False, padding=0),
            nn.BatchNorm2d(512),
            nn.ReLU6(inplace=True),

            nn.Conv2d(512, 256, 3, 2, bias=False, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU6(inplace=True),

            nn.Conv2d(256, 256, 1, 1, bias=False, padding=0),
            nn.BatchNorm2d(256),
            nn.ReLU6(inplace=True),

            nn.Conv2d(256, out_channels, 3, 2, bias=False, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU6(inplace=True)
        )
    
    def forward(self, x):
        out = self.ssdhead(x)
        return(out)

class ClassificationBlock(nn.Module):
    def __init__(self, in_channels, num_classes):
        super().__init__()
        self.flow = nn.Sequential(
            nn.Linear(in_channels, 512, bias=False),
            nn.BatchNorm1d(512),
            nn.ReLU(inplace=True),

            nn.Linear(512, 256, bias=False),
            nn.BatchNorm1d(256),
            nn.ReLU(inplace=True),

            nn.Linear(256, 128, bias=False),
            nn.BatchNorm1d(128),
            nn.ReLU(inplace=True),

            nn.Linear(128, num_classes, bias=True)
        )

    def forward(self, x):
        out = self.flow(x)
        return out

class KeypointBlock(nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        self.flow = nn.Sequential(
            nn.Linear(in_channels, 512, bias=False),
            nn.BatchNorm1d(512),
            nn.ReLU(inplace=True),

            nn.Linear(512, 256, bias=False),
            nn.BatchNorm1d(256),
            nn.ReLU(inplace=True),

            nn.Linear(256, 128, bias=False),
            nn.BatchNorm1d(128),
            nn.ReLU(inplace=True),

            nn.Linear(128, 8, bias=False),
        )

    def forward(self, x):
        out = self.flow(x)
        return out