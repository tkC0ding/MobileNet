import torch
from torch import nn
from DATA_LOAD import data_loading
from torchvision.transforms import transforms, ToTensor, Normalize
from torch.utils.data import random_split, DataLoader
from BLOCKS import InvertedResidualBlock, SSDhead, ClassificationBlock, KeypointBlock

image_dir = "data"
annotations_dir = "annotations/data1_data2_annotations.xml"
batch_size = 2
epochs = 10

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

dataset = data_loading(image_dir, annotations_dir,ToTensor())

train_size = int(0.7 * len(dataset)) + 2
test_size = int(0.1 * len(dataset))
val_size = int(0.2 * len(dataset))

train_data, valid_data, test_data = random_split(dataset, [train_size, val_size, test_size])

train_loader = DataLoader(train_data, batch_size, shuffle = True)
val_loader = DataLoader(valid_data, batch_size, shuffle = False)
test_loader = DataLoader(test_data,  batch_size, shuffle = False)

class MobileNetSSDv2(nn.Module):
    def __init__(self):
        super().__init__()

        self.initial_convolution = nn.Sequential(
            nn.Conv2d(3, 32, 3, 2, 1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU6(inplace=True)
        )

        self.BottleNeck1 = InvertedResidualBlock(32, 16, 1, 1)
        self.BottleNeck2 = InvertedResidualBlock(16, 24, 2, 6)
        self.BottleNeck3 = InvertedResidualBlock(24, 24, 1, 6)
        self.BottleNeck4 = InvertedResidualBlock(24, 32, 2, 6)
        self.BottleNeck5 = InvertedResidualBlock(32, 32, 1, 6)
        self.BottleNeck6 = InvertedResidualBlock(32, 64, 2, 6)
        self.BottleNeck7 = InvertedResidualBlock(64, 64, 1, 6)
        self.BottleNeck8 = InvertedResidualBlock(64, 64, 1, 6)
        self.BottleNeck9 = InvertedResidualBlock(64, 96, 2, 6)
        self.BottleNeck10 = InvertedResidualBlock(96, 96, 1, 6)
        self.BottleNeck11 = InvertedResidualBlock(96, 96, 1, 6)
        self.BottleNeck12 = InvertedResidualBlock(96, 160, 2, 6)
        self.BottleNeck13 = InvertedResidualBlock(160, 160, 1, 6)
        self.BottleNeck14 = InvertedResidualBlock(160, 160, 1, 6)
        self.BottleNeck15 = InvertedResidualBlock(160, 320, 1, 6)

        self.ssdhead = SSDhead(320, 128)

        self.flatten = nn.Flatten()

        self.classification = ClassificationBlock(768, 1)

        self.keypoints = KeypointBlock(768)

    def forward(self, x):
        out = self.initial_convolution(x)
        

        out = self.BottleNeck1(out)
        out = self.BottleNeck2(out)
        out = self.BottleNeck3(out)
        out = self.BottleNeck4(out)
        out = self.BottleNeck5(out)
        out = self.BottleNeck6(out)
        out = self.BottleNeck7(out)
        out = self.BottleNeck8(out)
        out = self.BottleNeck9(out)
        out = self.BottleNeck10(out)
        out = self.BottleNeck11(out)
        out = self.BottleNeck12(out)
        out = self.BottleNeck13(out)
        out = self.BottleNeck14(out)
        out = self.BottleNeck15(out)

        out = self.ssdhead(out)

        pick_off = self.flatten(out)

        classification_out = self.classification(pick_off)
        keypoint_out = self.keypoints(pick_off)

        return (classification_out, keypoint_out)

model = MobileNetSSDv2().to(device)

mse_loss = nn.MSELoss()
cross_entropy_loss = nn.BCEWithLogitsLoss()

optimizer = torch.optim.Adam(model.parameters(), 0.001)
for epoch in range(epochs):
    for img, label, keypoints in train_loader:
        img, label, keypoints = img.to(device), label.to(device), keypoints.to(device)

        label_pred, keypoints_pred = model(img)
        label_pred = label_pred.flatten()

        classification_loss = cross_entropy_loss(label_pred, label)

        if(not torch.all(keypoints == 0).item()):
            keypoint_loss = mse_loss(keypoints_pred, keypoints)
            total_loss = classification_loss + keypoint_loss
        else:
            total_loss = classification_loss
        

        optimizer.zero_grad()
        total_loss.backward()
        optimizer.step()
    print(epoch)