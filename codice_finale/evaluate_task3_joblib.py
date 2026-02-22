# =========================
# TASK 3 - EVAL da .JOBLIB
# SP-HOG (1x1+2x2) + SP-LBP (1x1+2x2) + PCA + LogisticRegression
# =========================

import os
import numpy as np
import joblib
from PIL import Image

from skimage.feature import hog, local_binary_pattern
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    classification_report, confusion_matrix
)

# =========================
# PATHS
# =========================
JOBLIB_PATH = r"F:\psd\file visual\task3\Nuova cartella\best_model_task3_SPHOG_1x1_2x2_SPLBP_1x1_2x2__PCA480_C8_lbfgs.joblib"
TEST_PATH   = r"F:\psd\file visual\visual_exam_dataset\visual_dataset\test_degradato"  

# =========================
# FEATURE EXTRACTION 
# =========================
def split_2x2(gray: np.ndarray):
    H, W = gray.shape
    h2, w2 = H // 2, W // 2
    return [
        gray[0:h2,   0:w2],   # TL
        gray[0:h2,   w2:W],   # TR
        gray[h2:H,   0:w2],   # BL
        gray[h2:H,   w2:W],   # BR
    ]

def hog_feat(gray: np.ndarray, hog_conf: dict) -> np.ndarray:
    return hog(
        gray,
        orientations=hog_conf["orient"],
        pixels_per_cell=hog_conf["ppc"],
        cells_per_block=hog_conf["cpb"],
        block_norm="L2-Hys",
        transform_sqrt=True,
        feature_vector=True
    ).astype(np.float32)

def extract_hog_pyramid(gray: np.ndarray, hog_conf: dict) -> np.ndarray:
    feats = [hog_feat(gray, hog_conf)]
    for p in split_2x2(gray):
        feats.append(hog_feat(p, hog_conf))
    return np.concatenate(feats, axis=0).astype(np.float32)

def lbp_hist(gray: np.ndarray, P: int, R: int) -> np.ndarray:
    lbp = local_binary_pattern(gray, P=P, R=R, method="uniform")
    n_bins = P + 2
    hist, _ = np.histogram(lbp.ravel(), bins=np.arange(0, n_bins + 1), range=(0, n_bins))
    hist = hist.astype(np.float32)
    hist /= (hist.sum() + 1e-6)
    return hist

def extract_lbp_pyramid(gray: np.ndarray, P: int, R: int) -> np.ndarray:
    feats = [lbp_hist(gray, P=P, R=R)]
    for p in split_2x2(gray):
        feats.append(lbp_hist(p, P=P, R=R))
    return np.concatenate(feats, axis=0).astype(np.float32)

def extract_feature(img_path: str, img_size: int, hog_conf: dict, lbp_conf: dict) -> np.ndarray:
    img = Image.open(img_path).convert("RGB").resize((img_size, img_size))
    gray = np.array(img.convert("L"))  # uint8

    hog_pyr = extract_hog_pyramid(gray, hog_conf)
    lbp_pyr = extract_lbp_pyramid(gray, P=lbp_conf["P"], R=lbp_conf["R"])

    return np.concatenate([hog_pyr, lbp_pyr], axis=0).astype(np.float32)

# =========================
# LOAD JOBLIB
# =========================
bundle = joblib.load(JOBLIB_PATH)
pipeline   = bundle["pipeline"]
class_names = bundle["class_names"]
hog_conf   = bundle["hog_conf"]
lbp_conf   = bundle["lbp_conf"]
img_size   = bundle["img_size"]
tag        = bundle.get("tag", "task3_model")

print("Loaded:", JOBLIB_PATH)
print("Tag:", tag)
print("Classes:", len(class_names))
print("img_size:", img_size)
print("hog_conf:", hog_conf)
print("lbp_conf:", lbp_conf)

# =========================
# BUILD X,y from folder (ImageFolder-like)
# =========================
def collect_samples(root_dir: str, class_names: list):
    samples = []
    for cls_idx, cls in enumerate(class_names):
        cls_dir = os.path.join(root_dir, cls)
        if not os.path.isdir(cls_dir):
            continue
        for fn in os.listdir(cls_dir):
            if fn.lower().endswith((".jpg", ".jpeg", ".png", ".bmp", ".webp")):
                samples.append((os.path.join(cls_dir, fn), cls_idx))
    return samples

samples = collect_samples(TEST_PATH, class_names)
print(f"Loaded test samples: {len(samples)}")

X = []
y = []
for path, label in samples:
    X.append(extract_feature(path, img_size, hog_conf, lbp_conf))
    y.append(label)

X = np.array(X, dtype=np.float32)
y = np.array(y, dtype=np.int64)

print("X shape:", X.shape)

# =========================
# PREDICT + METRICS
# =========================
y_pred = pipeline.predict(X)

acc = accuracy_score(y, y_pred)
macro_p = precision_score(y, y_pred, average="macro", zero_division=0)
macro_r = recall_score(y, y_pred, average="macro", zero_division=0)
macro_f1 = f1_score(y, y_pred, average="macro", zero_division=0)

print("\n====================")
print("GLOBAL METRICS (TEST)")
print("====================")
print(f"Accuracy:        {acc:.4f} ({acc*100:.2f}%)")
print(f"Macro Precision: {macro_p:.4f}")
print(f"Macro Recall:    {macro_r:.4f}")
print(f"Macro F1:        {macro_f1:.4f}")

print("\n=======================")
print("CLASSIFICATION REPORT")
print("=======================\n")
print(classification_report(y, y_pred, target_names=class_names, zero_division=0))