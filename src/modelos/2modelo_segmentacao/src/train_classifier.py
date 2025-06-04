import os
import torch
from torch import nn, optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from classifier import CrackClassifierCNN


# ------------------------------
# Hyperparameters
# ------------------------------
BATCH_SIZE = 16
EPOCHS = 25
LR = 0.001
IMG_SIZE = 128


# ------------------------------
# Data Loaders
# ------------------------------
transform = transforms.Compose([
    transforms.Grayscale(num_output_channels=1),
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize([0.5], [0.5])
])

train_dataset = datasets.ImageFolder('data/crops', transform=transform)
train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)

# Class names
class_names = train_dataset.classes
print(f"Classes: {class_names}")


# ------------------------------
# Model, Loss, Optimizer
# ------------------------------
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

model = CrackClassifierCNN().to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=LR)


# ------------------------------
# Training Loop
# ------------------------------
for epoch in range(EPOCHS):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0

    for images, labels in train_loader:
        images, labels = images.to(device), labels.to(device)

        # Forward
        outputs = model(images)
        loss = criterion(outputs, labels)

        # Backward
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # Stats
        running_loss += loss.item()
        _, preds = torch.max(outputs, 1)
        correct += (preds == labels).sum().item()
        total += labels.size(0)

    acc = 100 * correct / total
    print(f'Epoch [{epoch + 1}/{EPOCHS}] | Loss: {running_loss/len(train_loader):.4f} | Accuracy: {acc:.2f}%')

# Save model
os.makedirs('models', exist_ok=True)
torch.save(model.state_dict(), 'models/crack_classifier.pth')
print("Model saved to models/crack_classifier.pth")