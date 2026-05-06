import os
import csv
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader
from PIL import Image
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ================================================================
# GLOBAL CONFIGURATION
# ================================================================
# Using pathlib for robust cross-platform path management
SCRIPT_DIR = Path(__file__).resolve().parent
DATASET_ROOT = SCRIPT_DIR / "data" / "rim_dataset"

TRAIN_DIR = DATASET_ROOT / "train"
VAL_DIR   = DATASET_ROOT / "val"
TEST_DIR  = DATASET_ROOT / "test"

# Output directory for models and logs
OUTPUT_DIR = SCRIPT_DIR / "outputs" / "cnn_rim_model"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Hyperparameters
BATCH_SIZE = 32
NUM_EPOCHS = 20
LEARNING_RATE = 1e-3

# Hardware Acceleration Setup
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"🚀 Training initiated on: {DEVICE}")

# ================================================================
# 1) DATA AUGMENTATION & VALIDATION STRATEGY
# ================================================================
# Industrial-grade augmentation to improve model generalization
AUGMENTATION_PIPELINE = transforms.Compose()

def generate_augmented_val_set(source_train_dir, target_val_dir):
    """
    Creates a validation set by applying augmentation to training samples.
    This ensures the model is validated against varied industrial conditions.
    """
    if not target_val_dir.exists():
        target_val_dir.mkdir(parents=True)

    for class_folder in source_train_dir.iterdir():
        if not class_folder.is_dir(): continue

        val_class_path = target_val_dir / class_folder.name
        val_class_path.mkdir(exist_ok=True)

        for img_path in class_folder.iterdir():
            if img_path.suffix.lower() not in ['.jpg', '.jpeg', '.png']: continue

            try:
                with Image.open(img_path) as img:
                    rgb_img = img.convert("RGB")
                    augmented_img = AUGMENTATION_PIPELINE(rgb_img)
                    
                    save_name = f"{img_path.stem}_aug{img_path.suffix}"
                    augmented_img.save(val_class_path / save_name)
            except Exception as e:
                print(f"⚠️ Error processing {img_path.name}: {e}")

    print("✅ Augmented validation set generated successfully.")

# Check for validation set existence
if not VAL_DIR.exists() or not any(VAL_DIR.iterdir()):
    print("No validation data found -> Running auto-augmentation...")
    generate_augmented_val_set(TRAIN_DIR, VAL_DIR)
else:
    print("Found existing validation directory. Skipping generation.")

# ================================================================
# 2) TRANSFORMS & DATA LOADING
# ================================================================
# Standard ImageNet normalization for transfer learning
NORM_MEAN = [0.485, 0.456, 0.406]
NORM_STD  = [0.229, 0.224, 0.225]

TRAIN_TRANSFORMS = transforms.Compose()

INFERENCE_TRANSFORMS = transforms.Compose()

# Loading Datasets
train_data = datasets.ImageFolder(str(TRAIN_DIR), transform=TRAIN_TRANSFORMS)
val_data   = datasets.ImageFolder(str(VAL_DIR),   transform=INFERENCE_TRANSFORMS)
test_data  = datasets.ImageFolder(str(TEST_DIR),  transform=INFERENCE_TRANSFORMS)

# DataLoaders for efficient batching
train_loader = DataLoader(train_data, batch_size=BATCH_SIZE, shuffle=True)
val_loader   = DataLoader(val_data,   batch_size=BATCH_SIZE, shuffle=False)
test_loader  = DataLoader(test_data,  batch_size=BATCH_SIZE, shuffle=False)

NUM_CLASSES = len(train_data.classes)
print(f"Dataset Stats: {len(train_data)} Train | {len(val_data)} Val | {len(test_data)} Test")

# ================================================================
# 3) MODEL ARCHITECTURE (TRANSFER LEARNING)
# ================================================================
# Using ResNet18 as a robust backbone for industrial real-time vision
model = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)

# Fine-tuning the final fully connected layer for our specific categories
model.fc = nn.Linear(model.fc.in_features, NUM_CLASSES)
model = model.to(DEVICE)

# Loss & Optimizer selection
CRITERION = nn.CrossEntropyLoss()
OPTIMIZER = optim.Adam(model.parameters(), lr=LEARNING_RATE)

# ================================================================
# 4) TRAINING & EVALUATION LOGIC
# ================================================================
def run_epoch(model, loader, criterion, optimizer=None, is_training=False):
    if is_training:
        model.train()
    else:
        model.eval()

    running_loss, correct, total = 0.0, 0, 0
    
    # Context manager to handle gradients
    context = torch.enable_grad() if is_training else torch.no_grad()
    
    with context:
        for images, labels in loader:
            images, labels = images.to(DEVICE), labels.to(DEVICE)

            if is_training:
                optimizer.zero_grad()
                outputs = model(images)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()
            else:
                outputs = model(images)
                loss = criterion(outputs, labels)

            running_loss += loss.item() * images.size(0)
            _, preds = torch.max(outputs, 1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)

    return running_loss / total, correct / total

# ================================================================
# 5) FULL TRAINING PIPELINE
# ================================================================
best_val_accuracy = 0.0
BEST_MODEL_PATH = OUTPUT_DIR / "best_rim_model.pth"
METRICS_CSV = OUTPUT_DIR / "training_metrics.csv"

with open(METRICS_CSV, mode="w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(["epoch", "train_loss", "val_loss", "train_acc", "val_acc"])

    for epoch in range(NUM_EPOCHS):
        train_loss, train_acc = run_epoch(model, train_loader, CRITERION, OPTIMIZER, is_training=True)
        val_loss, val_acc     = run_epoch(model, val_loader, CRITERION)

        writer.writerow([epoch + 1, train_loss, val_loss, train_acc, val_acc])

        print(f"Epoch "
              f"Loss (T/V): {train_loss:.4f}/{val_loss:.4f} | "
              f"Acc (T/V): {train_acc*100:.2f}%/{val_acc*100:.2f}%")

        # Checkpointing the best performing model
        if val_acc > best_val_accuracy:
            best_val_accuracy = val_acc
            torch.save(model.state_dict(), BEST_MODEL_PATH)
            print(f"📍 Milestone: New best model saved (Acc: {val_acc*100:.2f}%)")

# ================================================================
# 6) FINAL TEST BENCH
# ================================================================
print("\n🏁 Final testing phase initiated...")
model.load_state_dict(torch.load(BEST_MODEL_PATH, map_location=DEVICE))
final_loss, final_acc = run_epoch(model, test_loader, CRITERION)

print(f"🎯 PRODUCTION READY METRICS -> Loss: {final_loss:.4f} | Accuracy: {final_acc*100:.2f}%")
