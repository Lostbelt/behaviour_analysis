# Behavior Pipeline UI

A GUI application for **YOLO Pose** inference on batches of videos, with **annotated video rendering**, **behavior metrics**, group **statistics**, and a simple **classifier**.  
Runs data-parallel across videos, preserves video aspect ratio, and avoids Qt geometry glitches.

<div align="center">
  <img src="figures/main_window.png" width="600"/>
</div>

---

## 🚀 Features

- Add **groups** (e.g., WT vs KO) by selecting folders with videos (`.mp4/.avi/.mov`).
- Run **single-frame YOLO Pose** on each video (left-most subject chosen).
- Write **annotated .mp4** (bbox + keypoints) and show them in an embedded player.
- Derive per-video metrics (freezing, research, rearing, nose distance, transitions).
- **Stats** tab: box/strip plots + **Mann–Whitney U** with **FDR-BH** correction.
- **Classifier** tab: RandomForest (+ optional **SHAP** feature importance).
- Robustness: **CUDA→CPU** fallback, keypoint **confidence filter**, geometry safeguards.

---

## 📦 Installation

Create and activate an environment:
```bash
conda create -n behav python=3.10 -y
conda activate behav
pip install -U pip wheel setuptools
```

Install PyTorch (pick one that matches your setup):
```bash
# CUDA example
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
# CPU-only
# pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
```

Install required libraries:
```bash
pip install PySide6 ultralytics opencv-python numpy pandas tqdm
pip install scikit-learn matplotlib seaborn scipy statsmodels
# optional for SHAP plots
pip install shap
```

> **Windows tip:** Install **PySide6 via pip** (not conda) to avoid Qt DLL conflicts.

---

## ▶️ Getting Started

Run the app:
```bash
python behav_ano_fixed_v2.py
```

Typical workflow:
1. **Pick model** (`.pt`, Ultralytics YOLO **keypoints** model) and select device (`cuda`/`cpu`).
2. **Add groups** → choose folders with videos.
3. Click **Run full pipeline** (parallel across videos).
4. Open **Videos** tab to play **annotated** outputs.
5. See **Table** for per-video metrics; use menu **Export table_data.csv**.
6. In **Stats**, choose two groups → **Draw plots** → `combined_stats.png` saved.
7. In **Classifier & SHAP**, select two groups → **Train RF + SHAP**.

---

## 📂 Outputs

```
runs/pose_app/<timestamp>/
├─ annot_<video>.mp4
├─ predictions_<group>.json
├─ table_data.csv
├─ combined_stats.png
└─ shap_summary.png   # if SHAP installed
```

---

## 🔧 Notes

- If you see a `torchvision::nms`/CUDA error, the app automatically retries on **CPU**.
- Increase **conf** if keypoints are noisy; decrease if detections are missing.
- Workers can be estimated via **Auto workers by VRAM** in the UI.
