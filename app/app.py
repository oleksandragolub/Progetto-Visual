import os
import numpy as np
import streamlit as st
from PIL import Image
import pandas as pd

# --- Task1/2 (PyTorch) ---
import torch
import torch.nn as nn
from torchvision import models, transforms

# --- Task3 (sklearn joblib) ---
import joblib
from skimage.feature import hog, local_binary_pattern

# =========================
# CONFIG PATHS (METTI I TUOI)
# =========================
SPORTS_CSV = r"F:\psd\file visual\task1\Nuova cartella\Nuova cartella\app\sports_labels.csv"

TASK1_PTH = r"F:\psd\file visual\task1\Nuova cartella\Nuova cartella\app\best_model_for_task1-db2-resnet-with-seed.pth"
TASK2_PTH = r"F:\psd\file visual\task1\Nuova cartella\Nuova cartella\app\best_model_task2_denoiseTestSet.pth"
TASK3_JOBLIB = r"F:\psd\file visual\task1\Nuova cartella\Nuova cartella\app\best_model_task3_SPHOG_1x1_2x2_SPLBP_1x1_2x2__PCA480_C8_lbfgs.joblib"

DEVICE = torch.device("cpu") 

@st.cache_resource
def load_class_names_from_csv(csv_path: str):
    df = pd.read_csv(csv_path)
    class_names = sorted(df["labels"].unique().tolist())
    return class_names

# =========================
# COMMON
# =========================
MEAN = [0.485, 0.456, 0.406]
STD  = [0.229, 0.224, 0.225]

tf_cnn = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(MEAN, STD),
])

# =========================
# TASK1/2 MODEL BUILDERS
# =========================
class SEBlock(nn.Module):
    def __init__(self, channels: int, reduction: int = 16):
        super().__init__()
        hidden = max(1, channels // reduction)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Conv2d(channels, hidden, kernel_size=1, bias=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden, channels, kernel_size=1, bias=True),
            nn.Sigmoid()
        )
    def forward(self, x):
        s = self.fc(self.pool(x))
        return x * s

class SEBasicBlock(nn.Module):
    expansion = 1
    def __init__(self, inplanes, planes, stride=1, downsample=None, reduction=16):
        super().__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, 3, stride=stride, padding=1, bias=False)
        self.bn1   = nn.BatchNorm2d(planes)
        self.relu  = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(planes, planes, 3, stride=1, padding=1, bias=False)
        self.bn2   = nn.BatchNorm2d(planes)
        self.se = SEBlock(planes, reduction=reduction)
        self.downsample = downsample

    def forward(self, x):
        identity = x
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out = self.se(out)
        if self.downsample is not None:
            identity = self.downsample(x)
        out = self.relu(out + identity)
        return out

class SEResNet(nn.Module):
    def __init__(self, layers=(2,2,2,2), num_classes=100, reduction=16):
        super().__init__()
        self.inplanes = 64
        self.conv1 = nn.Conv2d(3, 64, 7, stride=2, padding=3, bias=False)
        self.bn1   = nn.BatchNorm2d(64)
        self.relu  = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(3, stride=2, padding=1)

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
                nn.Conv2d(self.inplanes, planes, 1, stride=stride, bias=False),
                nn.BatchNorm2d(planes),
            )
        layers = [SEBasicBlock(self.inplanes, planes, stride=stride, downsample=downsample, reduction=reduction)]
        self.inplanes = planes
        for _ in range(1, blocks):
            layers.append(SEBasicBlock(self.inplanes, planes, stride=1, downsample=None, reduction=reduction))
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.maxpool(self.relu(self.bn1(self.conv1(x))))
        x = self.layer4(self.layer3(self.layer2(self.layer1(x))))
        x = torch.flatten(self.avgpool(x), 1)
        return self.fc(x)

def extract_state_dict(ckpt):
    if isinstance(ckpt, dict):
        for k in ["state_dict", "model_state_dict", "model", "net"]:
            if k in ckpt and isinstance(ckpt[k], dict):
                return ckpt[k]
    return ckpt

def build_cnn_from_filename(path: str, num_classes: int):
    name = os.path.basename(path).lower()
    if "efficientnet" in name:
        m = models.efficientnet_b0(weights=None)
        in_f = m.classifier[1].in_features
        m.classifier[1] = nn.Linear(in_f, num_classes)
        return m
    if "se-resnet" in name or "sp-se-resnet" in name or "dp-se-resnet" in name:
        return SEResNet(num_classes=num_classes, reduction=16)
    # default
    m = models.resnet18(weights=None)
    m.fc = nn.Linear(m.fc.in_features, num_classes)
    return m

@st.cache_resource
def load_cnn_model(path: str, num_classes: int = 100):
    model = build_cnn_from_filename(path, num_classes).to(DEVICE)
    ckpt = torch.load(path, map_location=DEVICE)
    state = extract_state_dict(ckpt)
    model.load_state_dict(state, strict=True)
    model.eval()
    return model

# =========================
# TASK3 FEATURE (da joblib bundle)
# =========================
def split_2x2(gray: np.ndarray):
    H, W = gray.shape
    h2, w2 = H // 2, W // 2
    return [gray[0:h2,0:w2], gray[0:h2,w2:W], gray[h2:H,0:w2], gray[h2:H,w2:W]]

def hog_feat(gray: np.ndarray, hog_conf: dict):
    return hog(gray,
               orientations=hog_conf["orient"],
               pixels_per_cell=hog_conf["ppc"],
               cells_per_block=hog_conf["cpb"],
               block_norm="L2-Hys",
               transform_sqrt=True,
               feature_vector=True).astype(np.float32)

def extract_hog_pyramid(gray: np.ndarray, hog_conf: dict):
    feats = [hog_feat(gray, hog_conf)]
    for p in split_2x2(gray):
        feats.append(hog_feat(p, hog_conf))
    return np.concatenate(feats).astype(np.float32)

def lbp_hist(gray: np.ndarray, P: int, R: int):
    lbp = local_binary_pattern(gray, P=P, R=R, method="uniform")
    n_bins = P + 2
    hist, _ = np.histogram(lbp.ravel(), bins=np.arange(0, n_bins + 1), range=(0, n_bins))
    hist = hist.astype(np.float32)
    return hist / (hist.sum() + 1e-6)

def extract_lbp_pyramid(gray: np.ndarray, P: int, R: int):
    feats = [lbp_hist(gray, P, R)]
    for p in split_2x2(gray):
        feats.append(lbp_hist(p, P, R))
    return np.concatenate(feats).astype(np.float32)

@st.cache_resource
def load_task3_bundle(joblib_path: str):
    bundle = joblib.load(joblib_path)
    return bundle

def predict_task3(img_pil: Image.Image, bundle: dict):
    pipeline = bundle["pipeline"]
    class_names = bundle["class_names"]
    hog_conf = bundle["hog_conf"]
    lbp_conf = bundle["lbp_conf"]
    img_size = bundle["img_size"]

    img = img_pil.convert("RGB").resize((img_size, img_size))
    gray = np.array(img.convert("L"))

    hog_pyr = extract_hog_pyramid(gray, hog_conf)
    lbp_pyr = extract_lbp_pyramid(gray, lbp_conf["P"], lbp_conf["R"])
    feat = np.concatenate([hog_pyr, lbp_pyr]).astype(np.float32).reshape(1, -1)

    # proba
    if hasattr(pipeline, "predict_proba"):
        proba = pipeline.predict_proba(feat)[0]
        top_idx = np.argsort(proba)[::-1][:5]
        return class_names[int(np.argmax(proba))], [(class_names[i], float(proba[i])) for i in top_idx]
    else:
        pred = int(pipeline.predict(feat)[0])
        return class_names[pred], [(class_names[pred], 1.0)]

# =========================
# PREDICT CNN
# =========================
@torch.no_grad()
def predict_cnn(img_pil: Image.Image, model: nn.Module):
    x = tf_cnn(img_pil.convert("RGB")).unsqueeze(0).to(DEVICE)
    logits = model(x)
    probs = torch.softmax(logits, dim=1).cpu().numpy()[0]
    top_idx = probs.argsort()[::-1][:5]
    return int(np.argmax(probs)), [(int(i), float(probs[i])) for i in top_idx]

# =========================
# STREAMLIT UI
# =========================
st.set_page_config(page_title="Visual Exam Classifier", layout="centered")
st.title("Visual Exam Classifier (Task1/Task2/Task3)")

task = st.selectbox("Seleziona task", ["Task1 (CNN .pth)", "Task2 (CNN .pth)", "Task3 (HOG/LBP .joblib)"])
uploaded = st.file_uploader("Carica un'immagine", type=["jpg", "jpeg", "png", "bmp", "webp"])

if uploaded is not None:
    img = Image.open(uploaded)
    st.image(img, caption="Input", use_container_width=True)

    if task.startswith("Task3"):
        bundle = load_task3_bundle(TASK3_JOBLIB)
        pred_name, top5 = predict_task3(img, bundle)
        st.subheader(f"Predizione: {pred_name}")
        st.write("Top-5:")
        for name, p in top5:
            st.write(f"- {name}: {p:.4f}")

    else:
        path = TASK1_PTH if task.startswith("Task1") else TASK2_PTH
        model = load_cnn_model(path, num_classes=100)

        class_names = load_class_names_from_csv(SPORTS_CSV)

        st.info("Per Task1/2 assicurati che l'ordine classi (class_names) sia lo stesso del dataset ImageFolder usato in training.")

        pred_idx, top5 = predict_cnn(img, model)

        # indice -> nome
        pred_name = class_names[pred_idx]
        st.subheader(f"Predizione: {pred_name} (index={pred_idx})")

        st.write("Top-5 (classe, prob):")
        for idx, p in top5:
            name = class_names[idx]
            st.write(f"- {name}: {p:.4f}  (idx={idx})")

st.caption("Tip: avvia con `streamlit run app.py`")