# File: src/train_classifier.py

import os
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader, WeightedRandomSampler
from tqdm import tqdm
from collections import Counter

from utils import DEVICE, LABEL_NAMES

# ── 1. Hyperparameters & Paths ────────────────────────────────────────────────
DATA_DIR = "data"  # contains train/ and val/
NUM_CLASSES = len(LABEL_NAMES)  # 4
BATCH_SIZE = 8
NUM_EPOCHS = 30          # extended training
LEARNING_RATE = 1e-4
MODEL_SAVE_PATH = "animal_explicit_classifier.pth"


class FocalLoss(nn.Module):
    """
    Focal Loss to handle class imbalance.
    """
    def __init__(self, gamma: float = 2.0, weight=None):
        super().__init__()
        self.gamma = gamma
        self.weight = weight

    def forward(self, inputs, targets):
        ce = nn.functional.cross_entropy(inputs, targets, weight=self.weight, reduction='none')
        pt = torch.exp(-ce)
        loss = ((1 - pt) ** self.gamma * ce).mean()
        return loss


def main():
    # ── 2. Data Transforms ───────────────────────────────────────────────────────
    train_transform = transforms.Compose([
        transforms.Resize(256),
        transforms.RandomResizedCrop(224, scale=(0.7, 1.0)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomApply([transforms.ColorJitter(0.3, 0.3, 0.3)], p=0.5),
        transforms.RandomGrayscale(p=0.1),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])
    val_transform = transforms.Compose([
        transforms.Resize(224),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])

    train_dataset = datasets.ImageFolder(os.path.join(DATA_DIR, "train"), transform=train_transform)
    val_dataset = datasets.ImageFolder(os.path.join(DATA_DIR, "val"), transform=val_transform)

    # ── 3. Weighted Sampling ──────────────────────────────────────────────────────
    counts = Counter([label for _, label in train_dataset.samples])
    print("Class counts:", counts)
    weights = [1.0 / counts[label] for _, label in train_dataset.samples]
    sampler = WeightedRandomSampler(weights, num_samples=len(weights), replacement=True)

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, sampler=sampler, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)

    # ── 4. Model Setup ────────────────────────────────────────────────────────────
    # Use EfficientNet-B3 backbone
    backbone = models.efficientnet_b3(weights=models.EfficientNet_B3_Weights.DEFAULT)
    in_feats = backbone.classifier[1].in_features
    backbone.classifier[1] = nn.Linear(in_feats, NUM_CLASSES)
    model = backbone.to(DEVICE)

    # Prepare loss with class weights
    class_counts = torch.tensor([counts[i] for i in range(NUM_CLASSES)], dtype=torch.float32)
    class_weights = (class_counts.sum() / class_counts).to(DEVICE)
    criterion = FocalLoss(gamma=2.0, weight=class_weights)

    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=NUM_EPOCHS)

    best_acc = 0.0

    # ── 5. Training Loop ─────────────────────────────────────────────────────────
    for epoch in range(1, NUM_EPOCHS + 1):
        model.train()
        train_loss, train_correct = 0.0, 0
        for X, y in tqdm(train_loader, desc=f"Epoch {epoch}/{NUM_EPOCHS} [Train]"):
            X, y = X.to(DEVICE), y.to(DEVICE)
            optimizer.zero_grad()
            logits = model(X)
            loss = criterion(logits, y)
            loss.backward()
            optimizer.step()

            train_loss += loss.item() * X.size(0)
            preds = logits.argmax(dim=1)
            train_correct += (preds == y).sum().item()

        train_loss /= len(train_dataset)
        train_acc = train_correct / len(train_dataset)
        print(f"Train  → Loss: {train_loss:.4f}, Acc: {train_acc:.4f}")

        # ── Validation ──────────────────────────────────────────────────────────
        model.eval()
        val_loss, val_correct = 0.0, 0
        with torch.no_grad():
            for X, y in tqdm(val_loader, desc=f"Epoch {epoch}/{NUM_EPOCHS} [Val]"):
                X, y = X.to(DEVICE), y.to(DEVICE)
                logits = model(X)
                loss = criterion(logits, y)

                val_loss += loss.item() * X.size(0)
                preds = logits.argmax(dim=1)
                val_correct += (preds == y).sum().item()

        val_loss /= len(val_dataset)
        val_acc = val_correct / len(val_dataset)
        print(f"Val    → Loss: {val_loss:.4f}, Acc: {val_acc:.4f}\n")

        scheduler.step()

        # ── Checkpoint ────────────────────────────────────────────────────
        if val_acc > best_acc:
            best_acc = val_acc
            torch.save(model.state_dict(), MODEL_SAVE_PATH)
            print(f"  >> Best model saved (acc={best_acc:.4f})\n")

    print(f"Training complete. Best val acc: {best_acc:.4f}")

    # ── 6. Confusion Matrix & Report ───────────────────────────────────────────
    try:
        from sklearn.metrics import confusion_matrix, classification_report
        model.load_state_dict(torch.load(MODEL_SAVE_PATH, map_location=DEVICE))
        model.eval()
        all_preds, all_labels = [], []
        with torch.no_grad():
            for X, y in val_loader:
                X, y = X.to(DEVICE), y.to(DEVICE)
                logits = model(X)
                preds = logits.argmax(dim=1)
                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(y.cpu().numpy())

        cm = confusion_matrix(all_labels, all_preds)
        print("Confusion Matrix:\n", cm)
        print(classification_report(all_labels, all_preds, target_names=LABEL_NAMES, digits=4))
    except ImportError:
        print("scikit-learn not available; skipping report.")


if __name__ == "__main__":
    main()
