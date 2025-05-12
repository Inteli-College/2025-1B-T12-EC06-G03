import os
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from sklearn.metrics import classification_report
from tqdm import tqdm
from collections import Counter
import json
from torchvision.datasets import ImageFolder

# CONFIGURAÇÕES
IMG_SIZE = 128
BATCH_SIZE = 16
EPOCHS = 10
PATIENCE = 3
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# TRANSFORMAÇÃO DE IMAGEM
transform = transforms.Compose([
    transforms.Grayscale(num_output_channels=1),
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize([0.5], [0.5])
])

# DATASETS
train_dataset = datasets.ImageFolder('../data/train', transform=transform)
val_dataset = datasets.ImageFolder('../data/val', transform=transform)
test_dataset = datasets.ImageFolder('../data/test', transform=transform)

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE)
test_loader = DataLoader(test_dataset, batch_size=1)

