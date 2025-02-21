import torch
from torch import nn
from torchvision import transforms
import cv2
from PIL import Image
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
        return classification_out, keypoint_out

# Load model weights
PATH = "Checkpoints/model_9/model_weights_9.pth"
model = MobileNetSSDv2().to(torch.device('cuda' if torch.cuda.is_available() else 'cpu'))
model.load_state_dict(torch.load(PATH, map_location=torch.device('cpu')))
model.eval()

# Image preprocessing function (matching DATA_LOAD.py)
def filters(img):
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    return img_rgb

# Define image transformation pipeline (matching DATA_LOAD.py)
transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
])

# Video processing function
def process_video(video_path, output_path='video/output_with_detections.mp4'):
    cap = cv2.VideoCapture(video_path)
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    if not cap.isOpened():
        print(f"Error: Cannot open video {video_path}")
        return

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, 30.0, (w, h))

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        enhanced_frame = filters(frame)
        input_tensor = transform(enhanced_frame).unsqueeze(0).to(next(model.parameters()).device)

        with torch.no_grad():
            label, keypoints = model(input_tensor)
            label_prob = torch.softmax(label, dim=1)
            label_idx = torch.argmax(label_prob, dim=1).item()
            confidence = label_prob[0, label_idx].item()
            label_name = classes[label_idx]

            print(f"Classification probabilities: {label_prob}, Label: {label_name}, Confidence: {confidence}")
            print("Keypoints:", keypoints.flatten().tolist())

            if label_name == "Gate":
                k = keypoints.flatten().tolist()
                for i in range(0, len(k), 2):
                    x, y = int(k[i] * w), int(k[i + 1] * h)
                    cv2.circle(frame, (x, y), 2, (255, 255, 255), -1)
                cv2.putText(frame, label_name, (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            else:
                cv2.putText(frame, label_name, (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

        out.write(frame)
        cv2.imshow('Object Detection', frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    out.release()
    cv2.destroyAllWindows()

# Run video processing
video_path = 'video/output_video.mp4'
process_video(video_path)
