from PIL import Image
from torch.utils.data import Dataset
from XML_PARSER import xml_parse
import torch

class data_loading(Dataset):
    def __init__(self, image_dir, annotations_dir, transforms=None):
        self.image_dir = image_dir
        self.annotations_dir = annotations_dir
        self.transforms = transforms
        self.annotations = self.load_annotations()

    
    def load_annotations(self):
        annotations = xml_parse(self.annotations_dir)
        return annotations
    
    def __len__(self):
        return len(self.annotations)
    
    def label_to_int(self, label):
        l_n = {"No-Gate" : 0, "Gate" : 1}
        return l_n[label]
    
    def __getitem__(self, idx):
        annotation = self.annotations[idx]
        filename = annotation['filename']

        keypoints = annotation['keypoints']
        label = annotation['label']
        img = Image.open(f"{self.image_dir}/{filename}")

        label = torch.tensor(self.label_to_int(label), dtype=torch.float32)
        keypoints = torch.tensor(keypoints, dtype=torch.float32)

        if self.transforms != None:
            img = self.transforms(img)
        
        return(img, label, keypoints)