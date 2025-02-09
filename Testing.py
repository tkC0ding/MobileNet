from torch import nn
from BLOCKS import InvertedResidualBlock, SSDhead, ClassificationBlock, KeypointBlock, Backbone
from torch.utils.data import random_split, DataLoader
from DATA_LOAD import data_loading
from torchvision.transforms import transforms, ToTensor, Normalize
import torch
import cv2

batch_size = 10

transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
])

model_checkpoint_path = 'Checkpoints/model_9/model_weights_9.pth'
image_dir = "data"
annotations_dir = "annotations/data1_data2_annotations.xml"

dataset = data_loading(image_dir, annotations_dir,transforms=transform)

train_size = int(0.7 * len(dataset)) + 2
test_size = int(0.1 * len(dataset))
val_size = int(0.2 * len(dataset))

train_data, valid_data, test_data = random_split(dataset, [train_size, val_size, test_size])

test_loader = DataLoader(test_data,  batch_size, shuffle = False)

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
        classification_out = self.classification(pick_off)
        keypoint_out = self.keypoints(pick_off)

        return (classification_out, keypoint_out)

model = MobileNetSSDv2()
model_weights = torch.load(model_checkpoint_path, weights_only=True)
model.load_state_dict(model_weights)

mse_loss = nn.MSELoss()
cross_entropy_loss = nn.BCEWithLogitsLoss()

with torch.no_grad():
    model.eval()
    test_loss = 0
    n = len(test_loader)
    for img, label, keypoints in test_loader:
        label_pred, keypoints_pred = model(img)
        label_pred = label_pred.flatten()

        val_classification_loss = cross_entropy_loss(label_pred, label)

        if(not torch.all(keypoints == 0).item()):
            keypoint_loss = mse_loss(keypoints_pred, keypoints)
            total_loss = val_classification_loss + keypoint_loss
        else:
            total_loss = val_classification_loss
        
        test_loss += total_loss

    avg_val_loss = test_loss/n
    print(f"Average testing loss : {avg_val_loss}")