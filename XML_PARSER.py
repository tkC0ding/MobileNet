import xml.etree.ElementTree as ET
import numpy as np

def debug_point_conversion(points_str):
    points_array = np.array([i.split(',') for i in points_str.split(';')]).flatten().astype(np.float32)
    normalized = points_array.copy()
    normalized[::2] /= 640  # x coordinates
    normalized[1::2] /= 480  # y coordinates
    return normalized

def xml_parse(xml_file, target_points=8):
    print("\nStarting XML parsing...")
    num_images = 0
    num_gates = 0
    num_backgrounds = 0
    
    tree = ET.parse(xml_file)
    root = tree.getroot()
    annotations = []
    
    for img in root.findall("image"):
        num_images += 1
        filename = img.get("name")
        point = img.find("points")
        
        if point is not None:
            points_str = point.get("points")
            if points_str and len(points_str.split(';')) == target_points // 2:
                kp = debug_point_conversion(points_str)
                kp = kp.tolist()
                l = "Gate"
                num_gates += 1
            else:
                kp = [0, 0, 0, 0, 0, 0, 0, 0]
                l = "Background"
                num_backgrounds += 1
        else:
            kp = [0, 0, 0, 0, 0, 0, 0, 0]
            l = "Background"
            num_backgrounds += 1
            
        annotations.append({
            "filename": filename,
            "label": l,
            "keypoints": kp
        })
    
    return annotations