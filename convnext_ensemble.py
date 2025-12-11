import os
from glob import glob
import pandas as pd
import numpy as np
from PIL import Image

import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as T
from torchvision.models import (
    convnext_tiny, convnext_small,
    ConvNeXt_Tiny_Weights, ConvNeXt_Small_Weights
)
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report
import torch.nn.functional as F

# configs
TRAIN_DIR = "./train"
TEST_DIR = "./test"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
BATCH = 32
NUM_CLASSES = 4
EPOCHS = 15

# helper functions
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

df = parse_train_files(TRAIN_DIR)
train_df, val_df = train_test_split(df, test_size=0.2, stratify=df.label, random_state=42)

train_transform = T.Compose([
    T.RandomResizedCrop(224, scale=(0.8,1.0)),
    T.RandomHorizontalFlip(),
    T.RandomRotation(15),
    T.RandomApply([T.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.2, hue=0.1)], p=0.7),
    T.ToTensor(),
    T.Normalize(mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225]),
    T.RandomErasing(p=0.4)
])

val_transform = T.Compose([
    T.Resize((224,224)),
    T.ToTensor(),
    T.Normalize(mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225])
])

train_samples = train_df.values.tolist()
val_samples = val_df.values.tolist()

train_ds = DamageDataset(train_samples, transform=train_transform)
val_ds = DamageDataset(val_samples, transform=val_transform)
train_loader = DataLoader(train_ds, batch_size=BATCH, shuffle=True, num_workers=4)
val_loader = DataLoader(val_ds, batch_size=BATCH, shuffle=False, num_workers=4)

# training
def train_model(model, train_loader, val_loader, lr_backbone, lr_head, epochs, save_name):
    model = model.to(DEVICE)
    for p in model.parameters(): p.requires_grad = True

    opt = torch.optim.Adam([
        {'params': model.features.parameters(), 'lr': lr_backbone},
        {'params': model.classifier.parameters(), 'lr': lr_head}
    ])
    lossfn = nn.CrossEntropyLoss()
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)

    best_acc = 0
    for epoch in range(epochs):
        model.train()
        for xb, yb in train_loader:
            xb, yb = xb.to(DEVICE), yb.to(DEVICE)
            preds = model(xb)
            loss = lossfn(preds, yb)
            opt.zero_grad()
            loss.backward()
            opt.step()
        scheduler.step()

        # Validation
        model.eval()
        correct, total = 0,0
        with torch.no_grad():
            for xb, yb in val_loader:
                xb, yb = xb.to(DEVICE), yb.to(DEVICE)
                out = model(xb).argmax(dim=1)
                correct += (out == yb).sum().item()
                total += yb.size(0)
        val_acc = correct / total
        print(f"Epoch {epoch+1}/{epochs} - val acc: {val_acc:.4f}")

        if val_acc > best_acc:
            best_acc = val_acc
            torch.save(model.state_dict(), save_name)
            print(f"Saved best model: {save_name}")
    return model

# train convnext tiny
tiny_model = convnext_tiny(weights=ConvNeXt_Tiny_Weights.DEFAULT)
tiny_model.classifier[2] = nn.Linear(tiny_model.classifier[2].in_features, NUM_CLASSES)
tiny_model = train_model(tiny_model, train_loader, val_loader, lr_backbone=1e-5, lr_head=1e-4, epochs=EPOCHS, save_name="best_tiny_model.pth")

# train convnext small
small_model = convnext_small(weights=ConvNeXt_Small_Weights.DEFAULT)
small_model.classifier[2] = nn.Linear(small_model.classifier[2].in_features, NUM_CLASSES)
small_model = train_model(small_model, train_loader, val_loader, lr_backbone=1e-5, lr_head=1e-4, epochs=EPOCHS, save_name="best_small_model.pth")

# load best weight
tiny_model.load_state_dict(torch.load("best_tiny_model.pth"))
small_model.load_state_dict(torch.load("best_small_model.pth"))
tiny_model.eval().to(DEVICE)
small_model.eval().to(DEVICE)

# validate ensemble
all_preds, all_labels = [], []

with torch.no_grad():
    for xb, yb in val_loader:
        xb = xb.to(DEVICE)
        probs_tiny = F.softmax(tiny_model(xb), dim=1)
        probs_small = F.softmax(small_model(xb), dim=1)
        probs_ensemble = (probs_tiny + probs_small)/2.0
        preds = probs_ensemble.argmax(dim=1).cpu().numpy()
        all_preds.extend(preds)
        all_labels.extend(yb.numpy())

cm = confusion_matrix(all_labels, all_preds)
print("\nENSEMBLED CONFUSION MATRIX")
print(cm)

target_names = ["HvD", "MiD", "MoD", "UD"]
print("\nENSEMBLED CLASSIFICATION REPORT")
print(classification_report(all_labels, all_preds, target_names=target_names))

# test ensemble
test_ds = TestDataset(TEST_DIR, transform=val_transform)
test_loader = DataLoader(test_ds, batch_size=BATCH, shuffle=False)

rows = []
with torch.no_grad():
    for xb, fnames in test_loader:
        xb = xb.to(DEVICE)
        probs_tiny = F.softmax(tiny_model(xb), dim=1)
        probs_small = F.softmax(small_model(xb), dim=1)
        probs_ensemble = (probs_tiny + probs_small)/2.0
        preds = probs_ensemble.argmax(dim=1).cpu().numpy()
        rows.extend(zip(fnames, preds))

pd.DataFrame(rows, columns=["ID","Label"]).to_csv("convnext_ensemble.csv", index=False)
print("Wrote convnext_ensemble.csv")
