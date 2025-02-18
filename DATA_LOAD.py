from PIL import Image
from torch.utils.data import Dataset
from XML_PARSER import xml_parse
import torch
import numpy as np
import cv2

class data_loading(Dataset):
    def __init__(self, image_dir, annotations_dir, transforms=None):
        self.image_dir = image_dir
        self.annotations_dir = annotations_dir
        self.transforms = transforms
        self.annotations = self.load_annotations()

    
    def load_annotations(self):
        annotations = xml_parse(self.annotations_dir,target_points=8)
        return annotations
    
    def __len__(self):
        return len(self.annotations)
    
    def label_to_int(self, label):
        l_n = {"No-Gate" : 0, "Gate" : 1}
        return l_n[label]
    
    def filters(self, img):
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
    
    def __getitem__(self, idx):
        annotation = self.annotations[idx]
        filename = annotation['filename']

        keypoints = annotation['keypoints']
        label = annotation['label']
        img = Image.open(f"{self.image_dir}/{filename}")
        img = img.resize((224, 224))
        img = np.array(img)
        img = self.filters(img)


        label = torch.tensor(self.label_to_int(label), dtype=torch.float32)
        keypoints = torch.tensor(keypoints, dtype=torch.float32)

        if self.transforms != None:
            img = self.transforms(img)
        
        return(img, label, keypoints)