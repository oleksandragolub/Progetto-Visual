import os
import re
import torch
import torch.nn as nn
import numpy as np

from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader

from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    classification_report, confusion_matrix
)

# =========================
# PATHS
# =========================
MODEL_PATH = r"F:\psd\file visual\task1\Nuova cartella\Nuova cartella\pesi\best_model_for_task1-db2-resnet-with-seed.pth"
TEST_PATH  = r"F:\psd\file visual\visual_exam_dataset\visual_dataset\test"  

#MODEL_PATH = r"F:\psd\file visual\best_model_task2_denoiseTestSet.pth"
#TEST_PATH  = r"F:\psd\file visual\visual_exam_dataset\visual_dataset\test_degradato"

DEVICE = torch.device("cpu")

# =========================
# TRANSFORMS (match training)
# =========================
test_tf = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225]),
])

# =========================
# DATASET
# =========================
test_ds = datasets.ImageFolder(TEST_PATH, transform=test_tf)
test_loader = DataLoader(test_ds, batch_size=64, shuffle=False)
class_names = test_ds.classes
num_classes = len(class_names)

print(f"Loaded test set: {len(test_ds)} images | {num_classes} classes")
print("Example classes:", class_names[:5])

# =========================
# HELPERS
# =========================
def extract_state_dict(ckpt):
    """Supporta file salvati come state_dict puro o dict con chiavi varie."""
    if isinstance(ckpt, dict):
        for k in ["state_dict", "model_state_dict", "model", "net"]:
            if k in ckpt and isinstance(ckpt[k], dict):
                return ckpt[k]
    return ckpt  # già state_dict

class SEBlock(nn.Module):
    def __init__(self, channels: int, reduction: int = 16):
        super().__init__()
        hidden = max(1, channels // reduction)

        # pool -> (B,C,1,1) e poi conv 1x1
        self.pool = nn.AdaptiveAvgPool2d(1)

        # IMPORTANT: conv 1x1 così i pesi hanno shape [out,in,1,1]
        # e le chiavi diventano se.fc.0.weight / se.fc.2.weight come nel state_dict
        self.fc = nn.Sequential(
            nn.Conv2d(channels, hidden, kernel_size=1, bias=True),  # fc.0
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden, channels, kernel_size=1, bias=True),  # fc.2
            nn.Sigmoid()
        )

    def forward(self, x):
        s = self.pool(x)      # (B,C,1,1)
        s = self.fc(s)        # (B,C,1,1)
        return x * s


class SEBasicBlock(nn.Module):
    expansion = 1
    def __init__(self, inplanes, planes, stride=1, downsample=None, reduction=16):
        super().__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1   = nn.BatchNorm2d(planes)
        self.relu  = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2   = nn.BatchNorm2d(planes)

        self.downsample = downsample
        self.stride = stride

        # nome "se" deve matchare layerX.Y.se.fc...
        self.se = SEBlock(planes, reduction=reduction)

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        out = self.se(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)
        return out

class SEResNet(nn.Module):
    def __init__(self, layers=(2,2,2,2), num_classes=100, reduction=16):
        super().__init__()
        self.inplanes = 64
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1   = nn.BatchNorm2d(64)
        self.relu  = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        self.layer1 = self._make_layer(64,  layers[0], stride=1, reduction=reduction)
        self.layer2 = self._make_layer(128, layers[1], stride=2, reduction=reduction)
        self.layer3 = self._make_layer(256, layers[2], stride=2, reduction=reduction)
        self.layer4 = self._make_layer(512, layers[3], stride=2, reduction=reduction)

        self.avgpool = nn.AdaptiveAvgPool2d((1,1))
        self.fc = nn.Linear(512, num_classes)

    def _make_layer(self, planes, blocks, stride, reduction):
        downsample = None
        if stride != 1 or self.inplanes != planes:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes),
            )

        layers = []
        layers.append(SEBasicBlock(self.inplanes, planes, stride=stride, downsample=downsample, reduction=reduction))
        self.inplanes = planes
        for _ in range(1, blocks):
            layers.append(SEBasicBlock(self.inplanes, planes, stride=1, downsample=None, reduction=reduction))
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x); x = self.bn1(x); x = self.relu(x); x = self.maxpool(x)
        x = self.layer1(x); x = self.layer2(x); x = self.layer3(x); x = self.layer4(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        return x

def build_model_from_filename(model_path: str, num_classes: int):
    name = os.path.basename(model_path).lower()

    if "efficientnet" in name:
        m = models.efficientnet_b0(weights=None)
        in_f = m.classifier[1].in_features
        m.classifier[1] = nn.Linear(in_f, num_classes)
        return m

    if "task2" in name or "denoise" in name:
        m = models.resnet18(weights=None)
        m.fc = nn.Linear(m.fc.in_features, num_classes)
        return m

    if "se-resnet" in name or "sp-se-resnet" in name or "dp-se-resnet" in name:
        return SEResNet(layers=(2,2,2,2), num_classes=num_classes, reduction=16)

    if "resnet" in name:
        m = models.resnet18(weights=None)
        m.fc = nn.Linear(m.fc.in_features, num_classes)
        return m

    raise ValueError(f"Tipo modello non riconosciuto dal nome file: {name}")

# =========================
# LOAD MODEL + WEIGHTS
# =========================
model = build_model_from_filename(MODEL_PATH, num_classes).to(DEVICE)
ckpt = torch.load(MODEL_PATH, map_location=DEVICE)
state = extract_state_dict(ckpt)

model.load_state_dict(state, strict=True)  # strict=True così segnala mismatch reali
model.eval()

# =========================
# EVALUATION
# =========================
all_preds = []
all_targets = []

with torch.no_grad():
    for inputs, labels in test_loader:
        inputs = inputs.to(DEVICE)
        labels = labels.to(DEVICE)
        outputs = model(inputs)
        preds = outputs.argmax(dim=1)

        all_preds.append(preds.cpu().numpy())
        all_targets.append(labels.cpu().numpy())

all_preds = np.concatenate(all_preds)
all_targets = np.concatenate(all_targets)

acc = accuracy_score(all_targets, all_preds)
macro_p = precision_score(all_targets, all_preds, average="macro", zero_division=0)
macro_r = recall_score(all_targets, all_preds, average="macro", zero_division=0)
macro_f1 = f1_score(all_targets, all_preds, average="macro", zero_division=0)
weighted_f1 = f1_score(all_targets, all_preds, average="weighted", zero_division=0)

print("\n====================")
print("GLOBAL METRICS (TEST)")
print("====================")
print(f"Accuracy:        {acc:.4f} ({acc*100:.2f}%)")
print(f"Macro Precision: {macro_p:.4f}")
print(f"Macro Recall:    {macro_r:.4f}")
print(f"Macro F1:        {macro_f1:.4f}")
print(f"Weighted F1:     {weighted_f1:.4f}")

print("\n=======================")
print("CLASSIFICATION REPORT")
print("=======================\n")
print(classification_report(all_targets, all_preds, target_names=class_names, zero_division=0))
