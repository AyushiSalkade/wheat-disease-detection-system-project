
#  # train.py


# import warnings
# warnings.filterwarnings("ignore")

# import os
# import torch
# import torch.nn as nn
# from torch.utils.data import DataLoader
# from torchvision import datasets, transforms, models

# # ---------------------------
# # Dataset folder
# dataset_dir = "dataset"
# train_dir = os.path.join(dataset_dir, "train")
# val_dir = os.path.join(dataset_dir, "val")

# print("Train folder contents:", os.listdir(train_dir))
# print("Val folder contents:", os.listdir(val_dir))

# # Transformations with normalization (for ResNet18)
# transform = transforms.Compose([
#     transforms.Resize((224, 224)),
#     transforms.ToTensor(),
#     transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])
# ])

# # Load datasets
# train_dataset = datasets.ImageFolder(root=train_dir, transform=transform)
# val_dataset = datasets.ImageFolder(root=val_dir, transform=transform)

# train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True, num_workers=0)
# val_loader = DataLoader(val_dataset, batch_size=16, shuffle=False, num_workers=0)

# print("Number of train samples:", len(train_dataset))
# print("Number of val samples:", len(val_dataset))
# print("Class to index mapping:", train_dataset.class_to_idx)

# # ---------------------------
# # Model
# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# model = models.resnet18(pretrained=True)
# num_ftrs = model.fc.in_features
# model.fc = nn.Linear(num_ftrs, len(train_dataset.classes))
# model = model.to(device)

# # Loss and optimizer
# criterion = nn.CrossEntropyLoss()
# optimizer = torch.optim.Adam(model.parameters(), lr=0.0001)

# # ---------------------------
# # Training function
# def train_model():
#     num_epochs = 3  # quick test
#     best_acc = 0.0

#     for epoch in range(num_epochs):
#         # ---- Train ----
#         model.train()
#         running_loss = 0.0
#         running_corrects = 0
#         for inputs, labels in train_loader:
#             inputs, labels = inputs.to(device), labels.to(device)
#             optimizer.zero_grad()
#             outputs = model(inputs)
#             _, preds = torch.max(outputs, 1)
#             loss = criterion(outputs, labels)
#             loss.backward()
#             optimizer.step()

#             running_loss += loss.item() * inputs.size(0)
#             running_corrects += torch.sum(preds == labels.data)

#         epoch_loss = running_loss / len(train_dataset)
#         epoch_acc = running_corrects.double() / len(train_dataset)

#         # ---- Validation ----
#         model.eval()
#         val_loss = 0.0
#         val_corrects = 0
#         with torch.no_grad():
#             for inputs, labels in val_loader:
#                 inputs, labels = inputs.to(device), labels.to(device)
#                 outputs = model(inputs)
#                 _, preds = torch.max(outputs, 1)
#                 loss = criterion(outputs, labels)

#                 val_loss += loss.item() * inputs.size(0)
#                 val_corrects += torch.sum(preds == labels.data)

#         val_epoch_loss = val_loss / len(val_dataset)
#         val_epoch_acc = val_corrects.double() / len(val_dataset)

#         print(f"Epoch [{epoch+1}/{num_epochs}]")
#         print(f"Train Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}")
#         print(f"Val Loss: {val_epoch_loss:.4f} Acc: {val_epoch_acc:.4f}\n")

#         # Save best model
#         if val_epoch_acc > best_acc:
#             best_acc = val_epoch_acc
#             torch.save(model.state_dict(), "best_model.pth")

#     print(f"Training complete. Best Validation Accuracy: {best_acc:.4f}")

# # ---------------------------
# if __name__ == "__main__":
#     train_model()


import warnings
warnings.filterwarnings("ignore")

import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms, models

# ---------------------------
# Dataset folder
dataset_dir = "dataset"
train_dir = os.path.join(dataset_dir, "train")
val_dir = os.path.join(dataset_dir, "val")

print("Train folder contents:", os.listdir(train_dir))
print("Val folder contents:", os.listdir(val_dir))

# Transformations with normalization
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])
])

# Load datasets
train_dataset = datasets.ImageFolder(root=train_dir, transform=transform)
val_dataset = datasets.ImageFolder(root=val_dir, transform=transform)

train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True, num_workers=0)
val_loader = DataLoader(val_dataset, batch_size=16, shuffle=False, num_workers=0)

print("Number of train samples:", len(train_dataset))
print("Number of val samples:", len(val_dataset))
print("Class to index mapping:", train_dataset.class_to_idx)

# ---------------------------
# Model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = models.resnet18(pretrained=False)  # fully offline
num_ftrs = model.fc.in_features
model.fc = nn.Linear(num_ftrs, len(train_dataset.classes))
model = model.to(device)

# Loss and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.0001)

# ---------------------------
# Training function
def train_model():
    num_epochs = 15  # final training
    best_acc = 0.0

    for epoch in range(num_epochs):
        # ---- Train ----
        model.train()
        running_loss = 0.0
        running_corrects = 0
        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * inputs.size(0)
            running_corrects += torch.sum(preds == labels.data)

        epoch_loss = running_loss / len(train_dataset)
        epoch_acc = running_corrects.double() / len(train_dataset)

        # ---- Validation ----
        model.eval()
        val_loss = 0.0
        val_corrects = 0
        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                _, preds = torch.max(outputs, 1)
                loss = criterion(outputs, labels)

                val_loss += loss.item() * inputs.size(0)
                val_corrects += torch.sum(preds == labels.data)

        val_epoch_loss = val_loss / len(val_dataset)
        val_epoch_acc = val_corrects.double() / len(val_dataset)

        print(f"Epoch [{epoch+1}/{num_epochs}]")
        print(f"Train Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}")
        print(f"Val Loss: {val_epoch_loss:.4f} Acc: {val_epoch_acc:.4f}\n")

        # Save best model
        if val_epoch_acc > best_acc:
            best_acc = val_epoch_acc
            torch.save(model.state_dict(), "best_model.pth")

    print(f"Training complete. Best Validation Accuracy: {best_acc:.4f}")

# ---------------------------
if __name__ == "__main__":
    train_model()