
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import models, datasets, transforms
from torch.utils.data import DataLoader
from training_metrics import compute_metrics, plot_curves, evaluate_model

# -------- Dataset Paths --------
train_dir = "dataset/train"
val_dir   = "dataset/val"

# -------- Transforms --------
transform = transforms.Compose([
    transforms.Resize((224,224)),
    transforms.ToTensor(),
])

# -------- Dataset & Loader --------
train_dataset = datasets.ImageFolder(train_dir, transform=transform)
val_dataset   = datasets.ImageFolder(val_dir, transform=transform)
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
val_loader   = DataLoader(val_dataset, batch_size=32, shuffle=False)

# -------- Classes --------
class_names = train_dataset.classes
print("Classes:", class_names)

# -------- Model --------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = models.resnet18(pretrained=False)
model.fc = nn.Linear(model.fc.in_features, len(class_names))
model.to(device)

# -------- Loss & Optimizer --------
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# -------- Training Loop --------
num_epochs = 5  # set according to your training
train_metrics_list = []
val_metrics_list = []

for epoch in range(num_epochs):
    model.train()
    running_loss = 0.0
    for images, labels in train_loader:
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()

    # Compute metrics after epoch
    train_metrics = compute_metrics(model, train_loader, criterion, device)
    val_metrics   = compute_metrics(model, val_loader, criterion, device)
    train_metrics_list.append(train_metrics)
    val_metrics_list.append(val_metrics)

    print(f"Epoch {epoch+1}/{num_epochs} done. Train Loss: {train_metrics[0]:.4f}, Val Loss: {val_metrics[0]:.4f}")

# -------- Save model --------
torch.save(model.state_dict(), "best_model.pth")

# -------- Plot Curves --------
plot_curves(train_metrics_list, val_metrics_list)

# -------- Evaluate on Validation --------
_, _, _, _, _, y_true_val, y_pred_val = val_metrics_list[-1]
evaluate_model(y_true_val, y_pred_val, class_names)