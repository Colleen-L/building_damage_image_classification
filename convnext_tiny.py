import os, random
import pandas as pd
import numpy as np
from PIL import Image
from glob import glob
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
import torchvision.transforms as T
from sklearn.metrics import confusion_matrix, classification_report
import timm

# Configs
TRAIN_DIR = "./train"
TEST_DIR  = "./test"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
BATCH = 16
NUM_CLASSES = 4
EPOCHS_STAGE1 = 8
EPOCHS_STAGE2 = 7
BEST_MODEL_PATH = "best_convnext.pth"
SUBMISSION_FILE = "convnext_tiny.csv"
torch.manual_seed(42)
np.random.seed(42)
random.seed(42)

# Helper Functions
def parse_train_files(train_dir):
    files = glob(os.path.join(train_dir, "*.png"))
    rows=[]
    label_map = {"HvD":0,"MiD":1,"MoD":2,"UD":3}
    for file in files:
        name = os.path.basename(file)
        id_part, label_part = name.split("_")
        label = label_map[label_part.replace(".png","")]
        rows.append((file,label))
    return pd.DataFrame(rows, columns=["path", "label"])
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
        self.transform = transform
        self.dir = test_dir
    def __len__(self):
        return len(self.files)
    def __getitem__(self, idx):
        path = os.path.join(self.dir, self.files[idx])
        img = Image.open(path).convert("RGB")
        if self.transform:
            img = self.transform(img)
        return img, self.files[idx]
    
# data set and data loaders
df = parse_train_files(TRAIN_DIR)
from sklearn.model_selection import train_test_split
train_df, val_df = train_test_split(
    df, test_size=0.2, stratify=df.label, random_state=42
)
train_transform = T.Compose([
    T.RandomResizedCrop(224, scale=(0.75, 1.0)),
    T.RandomHorizontalFlip(),
    T.RandomApply([
        T.ColorJitter(brightness=0.25, contrast=0.25, saturation=0.2, hue=0.1)
    ], p=0.6),
    T.RandomPerspective(distortion_scale=0.25, p=0.4),
    T.ToTensor(),
    T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    T.RandomErasing(p=0.25)
])
val_transform = T.Compose([
    T.Resize(256),
    T.CenterCrop(224),
    T.ToTensor(),
    T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])
train_samples = train_df.values.tolist()
val_samples   = val_df.values.tolist()
class_counts = train_df.label.value_counts().sort_index().values
class_weights = 1.0 / class_counts
sample_weights = [class_weights[label] for _, label in train_samples]
sampler = WeightedRandomSampler(sample_weights, len(sample_weights), replacement=True)
train_ds = DamageDataset(train_samples, transform=train_transform)
val_ds   = DamageDataset(val_samples, transform=val_transform)
train_loader = DataLoader(train_ds, batch_size=BATCH, sampler=sampler,
                          num_workers=4, pin_memory=True)
val_loader   = DataLoader(val_ds, batch_size=BATCH, shuffle=False,
                          num_workers=4, pin_memory=True)

# Model: ConvNeXt-Tiny ImageNet pretrained
model = timm.create_model(
    "convnext_tiny",
    pretrained=True,
    num_classes=NUM_CLASSES
).to(DEVICE)
# loss and optimization
weights = torch.tensor(class_weights / class_weights.sum(), dtype=torch.float32).to(DEVICE)
lossfn = nn.CrossEntropyLoss(weight=weights)
from torch.amp import autocast, GradScaler
scaler = GradScaler()

# freeze backbone
for name, p in model.named_parameters():
    if "head" not in name and "classifier" not in name:
        p.requires_grad = False
opt = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=1e-4)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=EPOCHS_STAGE1)
best_acc = 0
print("\n=== STAGE 1 TRAINING (Frozen Backbone) ===\n")
for epoch in range(EPOCHS_STAGE1):
    model.train()
    for xb, yb in train_loader:
        xb, yb = xb.to(DEVICE), yb.to(DEVICE)
        with autocast(device_type="cuda"):
            preds = model(xb)
            loss = lossfn(preds, yb)
        opt.zero_grad()
        scaler.scale(loss).backward()
        scaler.step(opt)
        scaler.update()
    scheduler.step()
    
    # validation
    model.eval()
    correct, total = 0, 0
    all_preds, all_targets = [], []
    with torch.no_grad():
        for xb, yb in val_loader:
            xb, yb = xb.to(DEVICE), yb.to(DEVICE)
            pred = model(xb).argmax(1)
            correct += (pred == yb).sum().item()
            total += yb.size(0)
            all_preds.extend(pred.cpu().numpy())
            all_targets.extend(yb.cpu().numpy())
    val_acc = correct / total
    print(f"[Stage1] Epoch {epoch+1}/{EPOCHS_STAGE1} | val acc = {val_acc:.4f}")
    if val_acc > best_acc:
        best_acc = val_acc
        torch.save(model.state_dict(), BEST_MODEL_PATH)
# unfreeze last blocks
print("\n=== STAGE 2 TRAINING (Unfreeze Last Blocks) ===\n")
for name, p in model.named_parameters():
    if "stages.2" in name or "stages.3" in name or "head" in name:
        p.requires_grad = True
    else:
        p.requires_grad = False
opt = torch.optim.AdamW(
    [p for p in model.parameters() if p.requires_grad],
    lr=1e-4, weight_decay=1e-4
)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=EPOCHS_STAGE2)
for epoch in range(EPOCHS_STAGE2):
    model.train()
    for xb, yb in train_loader:
        xb, yb = xb.to(DEVICE), yb.to(DEVICE)
        with autocast(device_type="cuda"):
            preds = model(xb)
            loss = lossfn(preds, yb)
        opt.zero_grad()
        scaler.scale(loss).backward()
        scaler.step(opt)
        scaler.update()
    scheduler.step()
    
    # validation
    model.eval()
    correct, total = 0, 0
    all_preds, all_targets = [], []
    with torch.no_grad():
        for xb, yb in val_loader:
            xb, yb = xb.to(DEVICE), yb.to(DEVICE)
            pred = model(xb).argmax(1)
            correct += (pred == yb).sum().item()
            total += yb.size(0)
            all_preds.extend(pred.cpu().numpy())
            all_targets.extend(yb.cpu().numpy())
    val_acc = correct / total
    print(f"[Stage2] Epoch {epoch+1}/{EPOCHS_STAGE2} | val acc = {val_acc:.4f}")
    if val_acc > best_acc:
        best_acc = val_acc
        torch.save(model.state_dict(), BEST_MODEL_PATH)
# final metrics
model.load_state_dict(torch.load(BEST_MODEL_PATH))
model.eval()
all_preds, all_targets = [], []
with torch.no_grad():
    for xb, yb in val_loader:
        xb, yb = xb.to(DEVICE), yb.to(DEVICE)
        pred = model(xb).argmax(1)
        all_preds.extend(pred.cpu().numpy())
        all_targets.extend(yb.cpu().numpy())
print("\n=== FINAL CONFUSION MATRIX ===")
print(confusion_matrix(all_targets, all_preds))
print("\n=== CLASSIFICATION REPORT ===")
print(classification_report(all_targets, all_preds, digits=4))
# test time prediction (light tta) 
test_ds = TestDataset(TEST_DIR, transform=val_transform)
test_loader = DataLoader(test_ds, batch_size=BATCH, shuffle=False)
rows=[]
with torch.no_grad():
    for xb, fnames in test_loader:
        xb = xb.to(DEVICE)
        out1 = model(xb)
        out2 = model(torch.flip(xb, dims=[3]))
        preds = ((out1 + out2) / 2).argmax(1).cpu().numpy()
        rows.extend(zip(fnames, preds))
pd.DataFrame(rows, columns=["ID","Label"]).to_csv(SUBMISSION_FILE, index=False)
print(f"\nWrote {SUBMISSION_FILE}")

