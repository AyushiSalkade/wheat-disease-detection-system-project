import torch
import matplotlib.pyplot as plt
from sklearn.metrics import precision_recall_fscore_support, classification_report, confusion_matrix
import seaborn as sns
import warnings

warnings.filterwarnings("ignore")

# ---------------- Metrics Computation ----------------
def compute_metrics(model, loader, criterion, device):
    model.eval()
    running_loss = 0.0
    correct, total = 0, 0
    y_true_all, y_pred_all = [], []

    with torch.no_grad():
        for images, labels in loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)
            running_loss += loss.item()
            _, preds = torch.max(outputs, 1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)
            y_true_all.extend(labels.cpu().numpy())
            y_pred_all.extend(preds.cpu().numpy())

    loss_avg = running_loss / len(loader)
    acc = 100 * correct / total
    prec, rec, f1, _ = precision_recall_fscore_support(y_true_all, y_pred_all, average='macro', zero_division=0)
    return loss_avg, acc, prec, rec, f1, y_true_all, y_pred_all

# ---------------- Plot Curves ----------------
def plot_curves(train_metrics_list, val_metrics_list):
    epochs = list(range(1, len(train_metrics_list)+1))

    train_losses = [x[0] for x in train_metrics_list]
    train_accs = [x[1] for x in train_metrics_list]
    train_precisions = [x[2] for x in train_metrics_list]
    train_recalls = [x[3] for x in train_metrics_list]
    train_f1s = [x[4] for x in train_metrics_list]

    val_losses = [x[0] for x in val_metrics_list]
    val_accs = [x[1] for x in val_metrics_list]
    val_precisions = [x[2] for x in val_metrics_list]
    val_recalls = [x[3] for x in val_metrics_list]
    val_f1s = [x[4] for x in val_metrics_list]

    # Loss
    plt.figure()
    plt.plot(epochs, train_losses, label="Train Loss", marker='o', color='blue')
    plt.plot(epochs, val_losses, label="Val Loss", marker='x', color='red')
    plt.xlabel("Epochs"); plt.ylabel("Loss"); plt.title("Loss Curve")
    plt.legend(); plt.savefig("loss_curve.png", dpi=300)

    # Accuracy
    plt.figure()
    plt.plot(epochs, train_accs, label="Train Acc", marker='o', color='blue')
    plt.plot(epochs, val_accs, label="Val Acc", marker='x', color='red')
    plt.xlabel("Epochs"); plt.ylabel("Accuracy (%)"); plt.title("Accuracy Curve")
    plt.legend(); plt.savefig("accuracy_curve.png", dpi=300)

    # Precision
    plt.figure()
    plt.plot(epochs, train_precisions, label="Train Precision", marker='o', color='green')
    plt.plot(epochs, val_precisions, label="Val Precision", marker='x', color='lime')
    plt.xlabel("Epochs"); plt.ylabel("Precision"); plt.title("Precision Curve")
    plt.legend(); plt.savefig("precision_curve.png", dpi=300)

    # Recall
    plt.figure()
    plt.plot(epochs, train_recalls, label="Train Recall", marker='o', color='orange')
    plt.plot(epochs, val_recalls, label="Val Recall", marker='x', color='red')
    plt.xlabel("Epochs"); plt.ylabel("Recall"); plt.title("Recall Curve")
    plt.legend(); plt.savefig("recall_curve.png", dpi=300)

    # F1-score
    plt.figure()
    plt.plot(epochs, train_f1s, label="Train F1-score", marker='o', color='purple')
    plt.plot(epochs, val_f1s, label="Val F1-score", marker='x', color='magenta')
    plt.xlabel("Epochs"); plt.ylabel("F1-score"); plt.title("F1-score Curve")
    plt.legend(); plt.savefig("f1_curve.png", dpi=300)

# ---------------- Evaluation ----------------
def evaluate_model(y_true, y_pred, class_names, save_path="conf_matrix.png"):
    print("\n📊 Classification Report:\n")
    print(classification_report(y_true, y_pred, target_names=class_names))
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(8,6))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=class_names, yticklabels=class_names)
    plt.xlabel("Predicted"); plt.ylabel("True"); plt.title("Confusion Matrix")
    plt.savefig(save_path, dpi=300)