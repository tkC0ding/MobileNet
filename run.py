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

os.mkdir('Checkpoints')

image_dir = "data"
annotations_dir = "annotations/modified_annotations.xml"
batch_size = 5
epochs = 10
alpha = 0.5

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

dataset = data_loading(image_dir, annotations_dir,transforms=transform)
train_size = int(0.7 * len(dataset))
test_size = int(0.1 * len(dataset))
val_size = int(0.2 * len(dataset))

train_data, valid_data, test_data = random_split(dataset, [train_size, val_size, test_size])

train_loader = DataLoader(train_data, batch_size, shuffle = True)
val_loader = DataLoader(valid_data, batch_size, shuffle = False)
test_loader = DataLoader(test_data,  batch_size, shuffle = False)

class MobileNetSSDv2(nn.Module):
    def __init__(self):
        super().__init__()

        self.backbone = Backbone()
        self.ssdhead = SSDhead(1000, 128)
        self.flatten = nn.Flatten()
        self.classification = ClassificationBlock(128, 1)
        self.keypoints = KeypointBlock(128)

    def forward(self, x):

        out = self.backbone(x)
        out = self.ssdhead(out)
        pick_off = self.flatten(out)
        classification_out   = self.classification(pick_off)
        keypoint_out = self.keypoints(pick_off)

        return (classification_out, keypoint_out)

model = MobileNetSSDv2().to(device)

mse_loss = nn.MSELoss()
cross_entropy_loss = nn.BCEWithLogitsLoss()

optimizer = torch.optim.Adam(model.parameters(), lr=0.0005, weight_decay=5e-4)  # weight_decay adds L2 regularization
for epoch in range(epochs):
    loss_sum = 0
    n = len(train_loader) * batch_size
    model.train()
    for img, label, keypoints in train_loader:
        img, label, keypoints = img.to(device), label.to(device), keypoints.to(device)

        label_pred, keypoints_pred = model(img)
        label_pred = label_pred.flatten()

        classification_loss = cross_entropy_loss(label_pred, label)

        if(not torch.all(keypoints == 0).item()):
            keypoint_loss = mse_loss(keypoints_pred, keypoints)
            total_loss = (alpha*classification_loss) + ((1 - alpha)*keypoint_loss)
        else:
            total_loss = classification_loss
        
        loss_sum += total_loss.item()

        optimizer.zero_grad()
        total_loss.backward()
        optimizer.step()


    avg_loss = loss_sum/n

    print(f"Epoch : {epoch}\tavg_loss : {avg_loss}\t")

    model.eval()
    val_loss_sum = 0
    with torch.no_grad():
        for img, label, keypoints in val_loader:
            img, label, keypoints = img.to(device), label.to(device), keypoints.to(device)

            label_pred, keypoints_pred = model(img)
            label_pred = label_pred.flatten()

            val_classification_loss = cross_entropy_loss(label_pred, label)

            if(not torch.all(keypoints == 0).item()):
                keypoint_loss = mse_loss(keypoints_pred, keypoints)
                total_loss = val_classification_loss + keypoint_loss
            else:
                total_loss = val_classification_loss
            
            val_loss_sum += total_loss

        avg_val_loss = val_loss_sum/n
        print(f"Validation Loss : {avg_val_loss}\t")
    os.mkdir(f"Checkpoints/model_{epoch}")
    torch.save(model, f'Checkpoints/model_{epoch}/entire_model_{epoch}.pth')
    torch.save(model.state_dict(), f'Checkpoints/model_{epoch}/model_weights_{epoch}.pth')