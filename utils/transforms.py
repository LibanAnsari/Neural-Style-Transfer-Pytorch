from torchvision import transforms

def get_transforms(IMG_SIZE: int):
    
    content_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean = [0.485, 0.456, 0.406],
                            std = [0.229, 0.224, 0.225]),
    ])

    style_transform = transforms.Compose([
        transforms.Resize(size=IMG_SIZE), # Resize only the style image
        transforms.ToTensor(),
        transforms.Normalize(mean = [0.485, 0.456, 0.406],
                            std = [0.229, 0.224, 0.225]),
    ])
    
    return content_transform, style_transform
