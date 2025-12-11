import os, random
import pandas as pd
import numpy as np
# Pillow = PIL = Python library for image processing
from PIL import Image
from glob import glob
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
import torchvision
import torchvision.transforms as T
from torchvision.models import resnet101, ResNet101_Weights

# configs
TRAIN_DIR = "./train"
TEST_DIR ="./test"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

BATCH = 32
NUM_CLASSES = 4
EPOCHS = 15

# HELPER FUNCTIONS

# parses labels from train filenames into ID and label
def parse_train_files(train_dir):
    files = glob(os.path.join(train_dir, "*.png"))
    rows=[]
    # HvD — Heavy Damage
    # MiD — Minor Damage
    # MoD — Moderate Damage
    # UD — Undamaged
    label_map = {"HvD":0,"MiD":1,"MoD":2,"UD":3}
    for file in files:
        name = os.path.basename(file)
        id_part, label_part = name.split("_")
        label = label_map[label_part.replace(".png","")]
        rows.append((file,label))
    return pd.DataFrame(rows, columns = ["path", "label"])

# Custom Dataset
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

# Custom Test Dataset
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

# parse train files
df = parse_train_files(TRAIN_DIR)

# stratified split
from sklearn.model_selection import train_test_split
train_df, val_df = train_test_split(df, test_size=0.2, stratify=df.label, random_state=42)

# training transformation with stronger augmentations
train_transform = T.Compose([
    T.RandomResizedCrop(224, scale=(0.8, 1.0)),
    T.RandomHorizontalFlip(),
    T.RandomRotation(15),
    T.RandomApply([T.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.2, hue=0.1)], p=0.7),
    T.ToTensor(),
    T.Normalize(mean=[.485, .456, .406], std=[.229, .224, .225]),
    T.RandomErasing(p=0.4)
])

# validation transformation
val_transform = T.Compose([
    T.Resize((224, 224)),
    T.ToTensor(),
    T.Normalize(mean=[.485, .456, .406], std=[.229, .224, .225])
])

# convert df to list of samples --> [(path, label), ...]
train_samples = train_df.values.tolist()
val_samples = val_df.values.tolist()

# create datasets
train_ds = DamageDataset(train_samples, transform=train_transform)
val_ds = DamageDataset(val_samples, transform=val_transform)

# create dataloaders
train_loader = DataLoader(train_ds, batch_size=BATCH, shuffle=True, num_workers=4)
val_loader = DataLoader(val_ds, batch_size=BATCH, shuffle=False, num_workers=4)

# create model
model = resnet101(weights=ResNet101_Weights.DEFAULT)

# replace final classification layer
model.fc = nn.Linear(model.fc.in_features, NUM_CLASSES)

# move model to GPU
model = model.to(DEVICE)

# unfreeze all layers
for p in model.parameters():
    p.requires_grad = True

# diff learning rates for backbone vs head
opt = torch.optim.Adam([
    {'params': model.layer1.parameters(), 'lr': 1e-5},
    {'params': model.layer2.parameters(), 'lr': 1e-5},
    {'params': model.layer3.parameters(), 'lr': 1e-5},
    {'params': model.layer4.parameters(), 'lr': 1e-5},
    {'params': model.fc.parameters(), 'lr': 1e-4}
])

# loss function
lossfn = nn.CrossEntropyLoss()

# learning rate scheduler
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=EPOCHS)

# track best validation accuracy
best_acc = 0

# training loop
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
    print(f"epoch {epoch} val acc {val_acc:.4f}")

    # save best model
    if val_acc > best_acc:
        # ---- EVALUATE ON VALIDATION SET WITH METRICS ----
        from sklearn.metrics import confusion_matrix, classification_report

        model.eval()
        all_preds = []
        all_labels = []

        with torch.no_grad():
            for xb, yb in val_loader:
                xb = xb.to(DEVICE)
                out = model(xb)
                preds = out.argmax(dim=1).cpu().numpy()
                all_preds.extend(preds)
                all_labels.extend(yb.numpy())

        # confusion matrix
        cm = confusion_matrix(all_labels, all_preds)
        print("\nCONFUSION MATRIX")
        print(cm)

        # classification report
        target_names = ["HvD", "MiD", "MoD", "UD"]
        print("\nCLASSIFICATION REPORT")
        print(classification_report(all_labels, all_preds, target_names=target_names))
 

# predicting test set
test_ds = TestDataset(TEST_DIR, transform=val_transform)
test_loader = DataLoader(test_ds, batch_size=BATCH, shuffle=False)

rows = []
model.eval()
with torch.no_grad():
    for xb, fnames in test_loader:
        xb = xb.to(DEVICE)
        preds = model(xb).argmax(dim=1).cpu().numpy()
        rows.extend(zip(fnames, preds))

# save submission
pd.DataFrame(rows, columns=["ID", "Label"]).to_csv("resnet101.csv", index=False)
print("Wrote resnet101.csv")
