import torch
from torch import nn
from torchvision import transforms
import cv2
from BLOCKS import InvertedResidualBlock, SSDhead, ClassificationBlock, KeypointBlock, Backbone

# Class labels
classes = ['Gate', 'Background']

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

# Load model weights
PATH = "Checkpoints/model_9/model_weights_9.pth"
model = MobileNetSSDv2()
model.load_state_dict(torch.load(PATH, map_location=torch.device('cpu')))
model.eval()
from PIL import Image
# Image preprocessing function
def filters(img):
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    b, g, r = cv2.split(img_rgb)
    
    b = cv2.normalize(b, None, 0, 255, cv2.NORM_MINMAX)
    g = cv2.normalize(g, None, 0, 255, cv2.NORM_MINMAX)
    r = cv2.normalize(r, None, 0, 255, cv2.NORM_MINMAX)

    img_corrected = cv2.merge((b, g, r))
    
    lab = cv2.cvtColor(img_corrected, cv2.COLOR_RGB2Lab)
    l_channel, a_channel, b_channel = cv2.split(lab)

    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(10, 50))
    cl = clahe.apply(l_channel)

    lab_corrected = cv2.merge((cl, a_channel, b_channel))
    img_enhanced = cv2.cvtColor(lab_corrected, cv2.COLOR_Lab2BGR)

    return img_enhanced

# Define image transformation pipeline
transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
])
img_path = "data/frame_00295.png"
img = cv2.imread(img_path)
img = filters(img)
input_tensor = transform(img).unsqueeze(0)

with torch.no_grad():
    label, keypoints = model(input_tensor)
    label_prob = torch.softmax(label, dim=1)
    print("Label probabilities:", label_prob)
    print("Keypoints:", keypoints.flatten().tolist())
