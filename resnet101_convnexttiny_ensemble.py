import os
from glob import glob
import pandas as pd
from PIL import Image
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as T
import torchvision.models as models
from torchvision.models import convnext_tiny, ConvNeXt_Tiny_Weights
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report
import torch.nn.functional as F

# Configs
TRAIN_DIR = "train"
TEST_DIR  = "test"
BATCH = 32
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
NUM_CLASSES = 4
EPOCHS = 6

# parses labels from train filenames into ID and label
def file_label_map(train_dir):
    files = glob(os.path.join(train_dir, "*.png"))
    rows=[]
    label_map = {"HvD":0,"MiD":1,"MoD":2,"UD":3}
    for f in files:
        name = os.path.basename(f)
        id_part, label_part = name.split("_")
        label = label_map[label_part.replace(".png","")]
        rows.append((name,label))
    return pd.DataFrame(rows, columns=["file","label"])

df = file_label_map(TRAIN_DIR)
train_df, val_df = train_test_split(df, test_size=0.2, stratify=df.label, random_state=42)

class ImgDataset(Dataset):
    def __init__(self, df, root, transforms):
        self.df = df.reset_index(drop=True)
        self.root = root
        self.t = transforms
    def __len__(self): return len(self.df)
    def __getitem__(self, i):
        fn = self.df.loc[i,'file']
        label = int(self.df.loc[i,'label'])
        img = Image.open(os.path.join(self.root,fn)).convert("RGB")
        return self.t(img), label

tr_transform = T.Compose([T.Resize((224,224)), T.RandomHorizontalFlip(), T.ToTensor(), T.Normalize([.485,.456,.406],[.229,.224,.225])])
val_transform= T.Compose([T.Resize((224,224)), T.ToTensor(), T.Normalize([.485,.456,.406],[.229,.224,.225])])

train_ds = ImgDataset(train_df, TRAIN_DIR, tr_transform)
val_ds   = ImgDataset(val_df, TRAIN_DIR, val_transform)
train_loader = DataLoader(train_ds, batch_size=BATCH, shuffle=True, num_workers=4)
val_loader   = DataLoader(val_ds, batch_size=BATCH, shuffle=False, num_workers=4)

# MODELS
# ResNet101
resnet_model = models.resnet101(weights=models.ResNet101_Weights.DEFAULT)
for p in resnet_model.parameters(): p.requires_grad = False
resnet_model.fc = nn.Linear(resnet_model.fc.in_features, NUM_CLASSES)
resnet_model = resnet_model.to(DEVICE)
resnet_opt = torch.optim.Adam(resnet_model.fc.parameters(), lr=1e-3)
lossfn = nn.CrossEntropyLoss()

# ConvNeXt-Tiny
conv_model = convnext_tiny(weights=ConvNeXt_Tiny_Weights.DEFAULT)
for p in conv_model.parameters(): p.requires_grad = False
conv_model.classifier[2] = nn.Linear(conv_model.classifier[2].in_features, NUM_CLASSES)
conv_model = conv_model.to(DEVICE)
conv_opt = torch.optim.Adam(conv_model.classifier.parameters(), lr=1e-3)

# training
def train_model(model, opt, loader, epochs):
    model.train()
    for epoch in range(epochs):
        for xb, yb in loader:
            xb, yb = xb.to(DEVICE), yb.to(DEVICE)
            preds = model(xb)
            loss = lossfn(preds, yb)
            opt.zero_grad()
            loss.backward()
            opt.step()
        print(f"Epoch {epoch+1}/{epochs} done.")
    return model

# train ResNet101 and ConvNeXt-Tiny
resnet_model = train_model(resnet_model, resnet_opt, train_loader, EPOCHS)
conv_model   = train_model(conv_model, conv_opt, train_loader, EPOCHS)

# validation of ensemble
all_preds = []
all_labels = []

resnet_model.eval()
conv_model.eval()

with torch.no_grad():
    for xb, yb in val_loader:
        xb = xb.to(DEVICE)
        yb = yb.numpy()
        all_labels.extend(yb)
        
        probs_res = F.softmax(resnet_model(xb), dim=1)
        probs_conv = F.softmax(conv_model(xb), dim=1)
        probs_ensemble = (probs_res + probs_conv)/2.0
        preds = probs_ensemble.argmax(dim=1).cpu().numpy()
        all_preds.extend(preds)

cm = confusion_matrix(all_labels, all_preds)
print("\nENSEMBLED CONFUSION MATRIX")
print(cm)

target_names = ["HvD","MiD","MoD","UD"]
print("\nENSEMBLED CLASSIFICATION REPORT")
print(classification_report(all_labels, all_preds, target_names=target_names))

# testing ensemble
test_files = sorted(os.listdir(TEST_DIR), key=lambda x: int(x.split(".")[0]))
rows=[]
with torch.no_grad():
    for fn in test_files:
        img = Image.open(os.path.join(TEST_DIR,fn)).convert("RGB")
        x = val_transform(img).unsqueeze(0).to(DEVICE)
        probs_res = F.softmax(resnet_model(x), dim=1)
        probs_conv = F.softmax(conv_model(x), dim=1)
        probs_ensemble = (probs_res + probs_conv)/2.0
        p = probs_ensemble.argmax(dim=1).item()
        rows.append((fn, int(p)))

pd.DataFrame(rows, columns=["ID","Label"]).to_csv("resnet101_convnext_tiny_ensemble.csv", index=False)
print("Wrote resnet101_convnext_tiny_ensemble.csv")