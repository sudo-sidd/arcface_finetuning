import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import transforms, datasets
from models.iresnet import iresnet100
from torch.optim import Adam

# Check if CUDA is available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load ArcFace model architecture
model = iresnet100(pretrained=False).to(device)  # Don't load default weights

# Load the pre-trained weights
model_path = "models/glink360k_cosface_r100_fp16_0.1.pth"
state_dict = torch.load(model_path, map_location=device)
model.load_state_dict(state_dict, strict=False)  # Load weights (strict=False allows missing layers)

# Modify the classification layer
num_classes = len(os.listdir("datasets/train"))  # Number of unique people in your dataset
model.fc = nn.Linear(model.fc.in_features, num_classes).to(device)  # Replace last layer

# Loss function & optimizer
criterion = nn.CrossEntropyLoss()
optimizer = Adam(model.parameters(), lr=0.0001)


# Data transformation and loading
transform = transforms.Compose([
    transforms.Resize((112, 112)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(10),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
])

train_dataset = datasets.ImageFolder(root="datasets/train", transform=transform)
train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)

val_dataset = datasets.ImageFolder(root="datasets/val", transform=transform)
val_loader = DataLoader(val_dataset, batch_size=16, shuffle=False)


def validate_model(model, val_loader):
    model.eval()
    correct, total = 0, 0
    with torch.no_grad():
        for images, labels in val_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs, 1)
            correct += (predicted == labels).sum().item()
            total += labels.size(0)
    return correct / total


def train_model(model, train_loader, val_loader, epochs=10):
    best_acc = 0.0
    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        correct, total = 0, 0

        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            _, predicted = torch.max(outputs, 1)
            correct += (predicted == labels).sum().item()
            total += labels.size(0)

        train_acc = correct / total
        val_acc = validate_model(model, val_loader)

        print(f"Epoch {epoch+1}: Loss: {running_loss:.4f}, Train Acc: {train_acc:.4f}, Val Acc: {val_acc:.4f}")

        # Save the best model
        if val_acc > best_acc:
            torch.save(model.state_dict(), "best_arcface.pth")
            best_acc = val_acc

    print("Training complete. Best model saved.")

# Start training
train_model(model, train_loader, val_loader, epochs=10)
