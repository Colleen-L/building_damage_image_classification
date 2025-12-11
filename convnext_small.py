import os, random, warnings
from glob import glob
from pathlib import Path

import pandas as pd
import numpy as np
from PIL import Image

import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
import torchvision
import torchvision.transforms as T
from torchvision.models import convnext_small, ConvNeXt_Small_Weights
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report

# configs
TRAIN_DIR = "./train"
TEST_DIR = "./test"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

BATCH = 32
NUM_CLASSES = 4
EPOCHS = 15

# helper function
def parse_train_files(train_dir):
    files = glob(os.path.join(train_dir, "*.png"))
    rows=[]
    label_map = {"HvD":0,"MiD":1,"MoD":2,"UD":3}
    for file in files:
        name = os.path.basename(file)
        id_part, label_part = name.split("_")
        label = label_map[label_part.replace(".png","")]
        rows.append((file,label))
    return pd.DataFrame(rows, columns = ["path", "label"])

# datasets
class DamageDataset(Dataset):
    def __init__(self, samples, transform=None):
        self.samples = samples
        self.transform = transform

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path, label = self.samples[idx]
        img = Image.open(path).convert("RGB")
        if self.transform:
            img = self.transform(img)
        return img, label

class TestDataset(Dataset):
    def __init__(self, test_dir, transform=None):
        self.files = sorted(os.listdir(test_dir), key=lambda x: int(x.split(".")[0]))
        self.dir = test_dir
        self.transform = transform

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        path = os.path.join(self.dir, self.files[idx])
        img = Image.open(path).convert("RGB")
        if self.transform:
            img = self.transform(img)
        return img, self.files[idx]

# data preparation

df = parse_train_files(TRAIN_DIR)
train_df, val_df = train_test_split(df, test_size=0.2, stratify=df.label, random_state=42)

# transforms
train_transform = T.Compose([
    T.RandomResizedCrop(224, scale=(0.8, 1.0)),
    T.RandomHorizontalFlip(),
    T.RandomRotation(15),
    T.RandomApply([T.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.2, hue=0.1)], p=0.7),
    T.ToTensor(),
    T.Normalize(mean=[.485, .456, .406], std=[.229, .224, .225]),
    T.RandomErasing(p=0.4)
])

val_transform = T.Compose([
    T.Resize((224, 224)),
    T.ToTensor(),
    T.Normalize(mean=[.485, .456, .406], std=[.229, .224, .225])
])

train_samples = train_df.values.tolist()
val_samples = val_df.values.tolist()

train_ds = DamageDataset(train_samples, transform=train_transform)
val_ds = DamageDataset(val_samples, transform=val_transform)

train_loader = DataLoader(train_ds, batch_size=BATCH, shuffle=True, num_workers=4)
val_loader = DataLoader(val_ds, batch_size=BATCH, shuffle=False, num_workers=4)

# Model: ConvNeXt Small
weights = ConvNeXt_Small_Weights.DEFAULT
model = convnext_small(weights=weights)

# replace classifier
in_features = model.classifier[2].in_features
model.classifier[2] = nn.Linear(in_features, NUM_CLASSES)

model = model.to(DEVICE)

# unfreeze all layers
for p in model.parameters():
    p.requires_grad = True

# optimizer
opt = torch.optim.Adam([
    {'params': model.features.parameters(), 'lr': 1e-5},  # backbone
    {'params': model.classifier.parameters(), 'lr': 1e-4} # head
])

# loss function & scheduler
lossfn = nn.CrossEntropyLoss()
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=EPOCHS)

# training
best_acc = 0

for epoch in range(EPOCHS):
    model.train()
    for xb, yb in train_loader:
        xb, yb = xb.to(DEVICE), yb.to(DEVICE)
        preds = model(xb)
        loss = lossfn(preds, yb)
        opt.zero_grad()
        loss.backward()
        opt.step()
    
    scheduler.step()
    
    # validation
    model.eval()
    correct, total = 0, 0
    with torch.no_grad():
        for xb, yb in val_loader:
            xb, yb = xb.to(DEVICE), yb.to(DEVICE)
            out = model(xb).argmax(dim=1)
            correct += (out == yb).sum().item()
            total += yb.size(0)
    val_acc = correct / total
    print(f"Epoch {epoch+1}/{EPOCHS} - val acc: {val_acc:.4f}")

# validation
all_preds = []
all_labels = []

model.eval()
with torch.no_grad():
    for xb, yb in val_loader:
        xb = xb.to(DEVICE)
        out = model(xb)
        preds = out.argmax(dim=1).cpu().numpy()
        all_preds.extend(preds)
        all_labels.extend(yb.numpy())

cm = confusion_matrix(all_labels, all_preds)
print("\nCONFUSION MATRIX")
print(cm)

target_names = ["HvD", "MiD", "MoD", "UD"]
print("\nCLASSIFICATION REPORT")
print(classification_report(all_labels, all_preds, target_names=target_names))

# predict test set
test_ds = TestDataset(TEST_DIR, transform=val_transform)
test_loader = DataLoader(test_ds, batch_size=BATCH, shuffle=False)

rows = []
model.eval()
with torch.no_grad():
    for xb, fnames in test_loader:
        xb = xb.to(DEVICE)
        preds = model(xb).argmax(dim=1).cpu().numpy()
        rows.extend(zip(fnames, preds))

pd.DataFrame(rows, columns=["ID", "Label"]).to_csv("convnext_small.csv", index=False)
print("Wrote convnext_small.csv")