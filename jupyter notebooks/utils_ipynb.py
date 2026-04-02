import matplotlib.pyplot as plt
from PIL import Image
import torch
import torchvision
import numpy as np
import os
from pathlib import Path

def showImage(path: str | Path, image_type: str = None):
    """
    Takes a path and shows the image presented in that path using matplotlib
    
    :param path: Path to the Image
    :type path: str | Path
    :param image_type: The type of image being shown (Optional)
    :type image_type: str
    """

    if not os.path.exists(path):
        print("[ERROR]: Path does not exists")
        return 
    
    if not isinstance(path, Path):
        path = Path(path)
    
    img = Image.open(path)
    img = np.asarray(img)
    
    plt.figure(figsize=(6, 4))
    plt.imshow(img)
    if image_type:
        plt.title(f"Name: {path.stem} | Type: {image_type} | Shape: {img.shape[0]} x {img.shape[1]} x {img.shape[2]}")
    else:
        plt.title(f"Name: {path.stem} | Shape: {img.shape[0]} x {img.shape[1]} x {img.shape[2]}")
    plt.axis('off')
    plt.show()
    

def denormalize(tensor: torch.Tensor):
    
    mean = torch.tensor([0.485, 0.456, 0.406]).view(1,3,1,1).to(tensor.device)
    std  = torch.tensor([0.229, 0.224, 0.225]).view(1,3,1,1).to(tensor.device)
    
    return tensor * std + mean

def load_rgb_pil(img):
    """
    Ensures a PIL image is RGB (drops alpha channel if present).
    """
    if img.mode != "RGB":
        img = img.convert("RGB")
    return img


def img_path_to_pil(path: str | Path):
    """
    Takes a path and return the image loaded in a PIL object.

    :param path: Path to the Image
    :type path: str | Path
    """
    
    if not os.path.exists(path):
        print("[ERROR]: Path does not exists")
        return 
    
    if not isinstance(path, Path):
        path = Path(path)
        
    img = Image.open(path)
    
    img = load_rgb_pil(img)
    
    return img