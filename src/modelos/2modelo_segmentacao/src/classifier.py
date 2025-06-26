import torch
import torch.nn as nn
import json
import cv2
import numpy as np
from pathlib import Path
from PIL import Image
from torchvision import transforms


def load_class_map():
    # Load class_to_idx.json to determine classes
    root = Path(__file__).resolve().parents[2]
    mapping_path = root / 'modeloB' / 'models' / 'class_to_idx.json'
    with open(mapping_path, 'r') as f:
        class_to_idx = json.load(f)
    idx_to_class = {v: k for k, v in class_to_idx.items()}
    return class_to_idx, idx_to_class


class CrackClassifierCNN(nn.Module):    
    def __init__(self, num_classes):
        super(CrackClassifierCNN, self).__init__()
        # Two conv layers as used in modeloB
        self.conv = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, kernel_size=3),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )
        # After two conv+pool, input 128x128 -> conv->126x126 -> pool->63x63 -> conv->61x61 -> pool->30x30
        self.fc = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(0.5),
            nn.Linear(64 * 30 * 30, 64),
            nn.ReLU(),
            nn.Linear(64, num_classes)
        )

    def forward(self, x):
        x = self.conv(x)
        x = self.fc(x)
        return x


def load_classifier(device=None):
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Load class mappings
    class_to_idx, idx_to_class = load_class_map()
    num_classes = len(class_to_idx)

    # Initialize model
    model = CrackClassifierCNN(num_classes).to(device)

    # Load weights
    root = Path(__file__).resolve().parents[2]
    weights_path = root / 'modeloB' / 'models' / 'cnn_model.pt'
    state_dict = torch.load(weights_path, map_location=device)
    model.load_state_dict(state_dict)
    model.eval()

    return model, idx_to_class, device

# Preprocessing transform for each crop
cnn_transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Grayscale(num_output_channels=1),
    transforms.Resize((128, 128)),
    transforms.ToTensor(),
    transforms.Normalize([0.5], [0.5])
])


def classify_crop(crop_image, model, idx_to_class, device):
    # Convert numpy array to PIL.Image if needed
    if not isinstance(crop_image, Image.Image):
        if crop_image.ndim == 3:
            crop_pil = Image.fromarray(cv2.cvtColor(crop_image, cv2.COLOR_BGR2RGB))
        else:
            crop_pil = Image.fromarray(crop_image)
    else:
        crop_pil = crop_image

    # Apply transforms
    input_tensor = cnn_transform(np.array(crop_pil)).unsqueeze(0).to(device)

    # Predict
    with torch.no_grad():
        outputs = model(input_tensor)
        _, pred = torch.max(outputs, 1)
    class_name = idx_to_class[pred.item()]
    return class_name
