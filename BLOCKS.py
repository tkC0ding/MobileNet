import torch
from torch import nn
import math

class InvertedResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride, expansion_factor, input_tensor_size=0):
        super().__init__()
        hidden_dim = in_channels * expansion_factor
        self.skip_connection = (stride == 1) and (in_channels == out_channels)
        if (in_channels == out_channels):
            padding = math.ceil(((input_tensor_size*(stride - 1)) + (3 - stride)) / 2)
        else:
            padding = 1

        self.expand = nn.Sequential(
            nn.Conv2d(in_channels, hidden_dim, kernel_size=1, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(hidden_dim),
            nn.ReLU6(inplace=True)
        )

        self.depthwise = nn.Sequential(
            nn.Conv2d(hidden_dim, hidden_dim, 3, stride, padding, groups=hidden_dim, bias=False),
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
        out = self.depthwise(out)
        out = self.out(out)

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
            nn.Dropout2d(0.5),  # Dropout for conv layers

            nn.Conv2d(1280, 512, 1, 1, bias=False, padding=0),
            nn.BatchNorm2d(512),
            nn.ReLU6(inplace=True),
            nn.Dropout2d(0.5),  # Dropout for conv layers

            nn.Conv2d(512, 256, 3, 2, bias=False, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU6(inplace=True),
            nn.Dropout2d(0.5),  # Dropout for conv layers

            nn.Conv2d(256, 256, 1, 1, bias=False, padding=0),
            nn.BatchNorm2d(256),
            nn.ReLU6(inplace=True),
            nn.Dropout2d(0.5),  # Dropout for conv layers

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
            nn.Dropout(0.6), #Added Dropouts

            nn.Linear(512, 256, bias=False),
            nn.BatchNorm1d(256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.6), #Added Dropouts

            nn.Linear(256, 128, bias=False),
            nn.BatchNorm1d(128),
            nn.ReLU(inplace=True),
            nn.Dropout(0.6), #Added Dropouts

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
            nn.Dropout(0.6), #Added Dropouts

            nn.Linear(512, 256, bias=False),
            nn.BatchNorm1d(256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.6), #Added Dropouts

            nn.Linear(256, 128, bias=False),
            nn.BatchNorm1d(128),
            nn.ReLU(inplace=True),
            nn.Dropout(0.6), #Added Dropouts

            nn.Linear(128, 8, bias=False),
        )

    def forward(self, x):
        out = self.flow(x)
        return out

class Backbone(nn.Module):
    def __init__(self):
        super().__init__()
        self.initial_convolution = nn.Sequential(
            nn.Conv2d(3, 32, 3, 2, 1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU6(inplace=True)
        )

        self.BottleNeck1 = InvertedResidualBlock(32, 16, 1, 1)

        self.BottleNeck2 = InvertedResidualBlock(16, 24, 2, 6)
        self.BottleNeck3 = InvertedResidualBlock(24, 24, 2, 6, 56)

        self.BottleNeck4 = InvertedResidualBlock(24, 32, 2, 6)
        self.BottleNeck5 = InvertedResidualBlock(32, 32, 2, 6, 28)
        self.BottleNeck6 = InvertedResidualBlock(32, 32, 2, 6, 28)

        self.BottleNeck7 = InvertedResidualBlock(32, 64, 2, 6)
        self.BottleNeck8 = InvertedResidualBlock(64, 64, 2, 6, 14)
        self.BottleNeck9 = InvertedResidualBlock(64, 64, 2, 6, 14)
        self.BottleNeck10 = InvertedResidualBlock(64, 64, 2, 6, 14)

        self.BottleNeck11 = InvertedResidualBlock(64, 96, 1, 6)
        self.BottleNeck12 = InvertedResidualBlock(96, 96, 1, 6, 14)
        self.BottleNeck13 = InvertedResidualBlock(96, 96, 1, 6, 14)

        self.BottleNeck14 = InvertedResidualBlock(96, 160, 2, 6)
        self.BottleNeck15 = InvertedResidualBlock(160, 160, 2, 6, 7)
        self.BottleNeck16 = InvertedResidualBlock(160, 160, 2, 6, 7)

        self.BottleNeck17 = InvertedResidualBlock(160, 320, 1, 6, 7)

        self.last_flow = nn.Sequential(
            nn.Conv2d(320, 1280, 1, 1),
            nn.AvgPool2d(7),
            nn.BatchNorm2d(1280),
            nn.ReLU6(inplace=True),
            nn.Conv2d(1280, 1000, 1),
            nn.BatchNorm2d(1000),
            nn.ReLU6(inplace=True)
        )
    
    def forward(self, x):
        
        out = self.initial_convolution(x)
        
        out= self.BottleNeck1(out)
        out= self.BottleNeck2(out)
        out= self.BottleNeck3(out)
        out= self.BottleNeck4(out)
        out= self.BottleNeck5(out)
        out= self.BottleNeck6(out)
        out= self.BottleNeck7(out)
        out= self.BottleNeck8(out)
        out= self.BottleNeck9(out)
        out= self.BottleNeck10(out)
        out= self.BottleNeck11(out)
        out= self.BottleNeck12(out)
        out= self.BottleNeck13(out)
        out= self.BottleNeck14(out)
        out= self.BottleNeck15(out)
        out= self.BottleNeck16(out)
        out= self.BottleNeck17(out)

        out = self.last_flow(out)

        return out