import os
import torch
from torch import nn
from DATA_LOAD import data_loading
from torchvision.transforms import transforms, ToTensor, Normalize
from torch.utils.data import random_split, DataLoader
from BLOCKS import InvertedResidualBlock, SSDhead, ClassificationBlock, KeypointBlock, Backbone

transform = transforms.Compose(
    [transforms.ToTensor(),
    transforms.Normalize([0.5, 0.5, 0.5], [0.5,0.5,0.5])]
)

image_dir = "data"
annotations_dir = "annotations/data_1.xml"
batch_size = 5
epochs = 10
alpha = 0.5

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

dataset = data_loading(image_dir, annotations_dir,transforms=transform)
print(len(dataset))
train_size = int(0.7 * len(dataset))
test_size = int(0.1 * len(dataset))
val_size = int(0.2 * len(dataset))
print(train_size)
print(test_size)
print(val_size)