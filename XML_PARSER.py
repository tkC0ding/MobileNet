import xml.etree.ElementTree as ET
import numpy as np

def xml_parse(xml_file):
    tree = ET.parse(xml_file)
    root = tree.getroot()

    annotations = []
    for img in root.findall("image"):
        filename = img.get("name")
        point = img.find("points")
        if point != None:
            kp = np.array([i.split(',') for i in point.get("points").split(';')]).flatten().astype(np.float16).astype(np.int16).tolist()
            l = "Gate"
        else:
            kp = [0, 0, 0, 0, 0, 0, 0, 0]
            l = "No-Gate"
        
        annotations.append({
            "filename" : filename,
            "label" : l,
            "keypoints" : kp
        })
    
    return annotations