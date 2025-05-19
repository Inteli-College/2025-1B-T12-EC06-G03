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

# MODELO CNN
class CNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(32, 64, kernel_size=3), nn.ReLU(), nn.MaxPool2d(2)
        )
        self.fc = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(0.5),
            nn.Linear(64 * 30 * 30, 64),
            nn.ReLU(),
            nn.Linear(64, 2)
        )

    def forward(self, x):
        return self.fc(self.conv(x))

# TREINAMENTO
model = CNN().to(DEVICE)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

best_val_loss = float('inf')
patience_counter = 0

for epoch in range(EPOCHS):
    print(f"Epoch {epoch+1}/{EPOCHS}")
    model.train()
    running_loss = 0
    for images, labels in tqdm(train_loader, desc="Training"):
        images, labels = images.to(DEVICE), labels.to(DEVICE)
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()

    # VALIDAÇÃO
    model.eval()
    val_loss = 0
    correct, total = 0, 0
    with torch.no_grad():
        for images, labels in val_loader:
            images, labels = images.to(DEVICE), labels.to(DEVICE)
            outputs = model(images)
            loss = criterion(outputs, labels)
            val_loss += loss.item()
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    val_loss /= len(val_loader)
    val_acc = correct / total
    print(f"Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.4f}")

    if val_loss < best_val_loss:
        best_val_loss = val_loss
        torch.save(model.state_dict(), '../models/cnn_model.pt')
        # Salvar o mapeamento das classes para uso posterior
        with open('../models/class_to_idx.json', 'w') as f:
            json.dump(train_dataset.class_to_idx, f)
        patience_counter = 0
    else:
        patience_counter += 1
        if patience_counter >= PATIENCE:
            print("Early stopping.")
            break

# TESTE
print("\n🔍 Avaliação no conjunto de teste:")
model.load_state_dict(torch.load('../models/cnn_model.pt'))
model.eval()

y_true, y_pred = [], []
file_names = []

# Caminho para pegar os nomes reais das imagens
test_root = test_dataset.root
class_to_idx = test_dataset.class_to_idx
idx_to_class = {v: k for k, v in class_to_idx.items()}

with torch.no_grad():
    for i, (images, labels) in enumerate(test_loader):
        images = images.to(DEVICE)
        outputs = model(images)
        pred = torch.argmax(outputs, 1).cpu().item()
        true = labels.item()
        y_pred.append(pred)
        y_true.append(true)

        # Recuperar caminho real da imagem
        path, _ = test_dataset.samples[i]
        filename = os.path.basename(path)
        file_names.append(filename)

        pred_label = idx_to_class[pred]
        true_label = idx_to_class[true]

        print(f"{filename} → Predito: {pred_label} | Real: {true_label}")

# Relatório geral
from sklearn.metrics import classification_report
from collections import Counter

print("\n📊 Relatório geral:")
print(classification_report(y_true, y_pred, target_names=["thermal", "retraction"]))
print("\n📈 Contagem de predições feitas pelo modelo:", Counter(y_pred))




