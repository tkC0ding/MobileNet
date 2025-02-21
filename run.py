import os
import torch
from torch import nn
import numpy as np
from DATA_LOAD import data_loading
from torchvision.transforms import transforms
from torch.utils.data import random_split, DataLoader
from BLOCKS import InvertedResidualBlock, SSDhead, ClassificationBlock, KeypointBlock, Backbone

# Image transformations
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
])

# Dataset parameters
image_dir = "data"
annotations_dir = "annotations/modified_annotations.xml"
batch_size = 5
epochs = 10
alpha = 0.5

# Device configuration
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load dataset
dataset = data_loading(image_dir, annotations_dir, transforms=transform)

# Random split with shuffling and fixed seed for reproducibility
train_size = int(0.7 * len(dataset))
test_size = int(0.1 * len(dataset))
val_size = len(dataset) - train_size - test_size

# Shuffle dataset before splitting
generator = torch.Generator().manual_seed(42)
train_data, valid_data, test_data = random_split(dataset, [train_size, val_size, test_size], generator=generator)

# Print class distribution in each split
print("\nClass distribution in splits:")
def count_labels(dataloader):
    gate_count = 0
    background_count = 0
    for _, label, _ in dataloader:
        gate_count += torch.sum(label[:, 0]).item()
        background_count += torch.sum(label[:, 1]).item()
    return gate_count, background_count

train_loader = DataLoader(train_data, batch_size, shuffle=True)
val_loader = DataLoader(valid_data, batch_size, shuffle=False)
test_loader = DataLoader(test_data, batch_size, shuffle=False)

train_gates, train_background = count_labels(train_loader)
val_gates, val_background = count_labels(val_loader)
test_gates, test_background = count_labels(test_loader)

print(f"Train - Gates: {train_gates}, Backgrounds: {train_background}")
print(f"Validation - Gates: {val_gates}, Backgrounds: {val_background}")
print(f"Test - Gates: {test_gates}, Backgrounds: {test_background}")

class MobileNetSSDv2(nn.Module):
    def __init__(self):
        super().__init__()
        self.backbone = Backbone()
        self.ssdhead = SSDhead(1000, 128)
        self.flatten = nn.Flatten()
        self.classification = ClassificationBlock(128, 2)
        self.keypoints = KeypointBlock(128)

    def forward(self, x):
        out = self.backbone(x)
        out = self.ssdhead(out)
        pick_off = self.flatten(out)
        classification_out = self.classification(pick_off)
        keypoint_out = self.keypoints(pick_off)
        return (classification_out, keypoint_out)

# Initialize model
model = MobileNetSSDv2().to(device)

# Create checkpoints directory
os.makedirs('Checkpoints', exist_ok=True)

# Loss functions and optimizer
mse_loss = nn.MSELoss()
cross_entropy_loss = nn.BCEWithLogitsLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

print("\nStarting training...")
for epoch in range(epochs):
    model.train()
    loss_sum = 0
    num_batches = 0

    for img, label, keypoints in train_loader:
        img, label, keypoints = img.to(device), label.to(device), keypoints.to(device)
        optimizer.zero_grad()
        label_pred, keypoints_pred = model(img)

        classification_loss = cross_entropy_loss(label_pred, label)
        if not torch.all(keypoints == 0).item():
            keypoint_loss = mse_loss(keypoints_pred, keypoints)
            total_loss = (alpha * classification_loss) + ((1 - alpha) * keypoint_loss)
        else:
            total_loss = classification_loss

        loss_sum += total_loss.item()
        num_batches += 1

        total_loss.backward()
        optimizer.step()

    avg_loss = loss_sum / num_batches

    # Validation phase
    model.eval()
    val_loss_sum = 0
    num_val_batches = 0

    with torch.no_grad():
        for img, label, keypoints in val_loader:
            img, label, keypoints = img.to(device), label.to(device), keypoints.to(device)
            label_pred, keypoints_pred = model(img)

            val_classification_loss = cross_entropy_loss(label_pred, label)
            if not torch.all(keypoints == 0).item():
                val_keypoint_loss = mse_loss(keypoints_pred, keypoints)
                total_loss = (alpha * val_classification_loss) + ((1 - alpha) * val_keypoint_loss)
            else:
                total_loss = val_classification_loss

            val_loss_sum += total_loss.item()
            num_val_batches += 1

    avg_val_loss = val_loss_sum / num_val_batches

    # Save checkpoints
    checkpoint_dir = f"Checkpoints/model_{epoch}"
    os.makedirs(checkpoint_dir, exist_ok=True)
    torch.save(model, f'{checkpoint_dir}/entire_model_{epoch}.pth')
    torch.save(model.state_dict(), f'{checkpoint_dir}/model_weights_{epoch}.pth')

    print(f"Epoch {epoch + 1}/{epochs} - Avg Loss: {avg_loss:.4f}, Val Loss: {avg_val_loss:.4f}")

print("\nTraining completed!")
