# Behavior Pipeline UI

A GUI application for **YOLO Pose** inference on batches of videos, with **annotated video rendering**, **behavior metrics**, group **statistics**, and a simple **classifier**.  
Runs data-parallel across videos, preserves video aspect ratio, and avoids Qt geometry glitches.

<div align="center">
  <img src="figures/main_window.png" width="600"/>
</div>

---

## 🚀 Features

- Add **groups** (e.g., WT vs KO) by selecting folders with videos (`.mp4/.avi/.mov`).
- Run **single-frame YOLO Pose** on each video (left-most subject chosen). Detection itself always uses `conf=0.10`; the GUI “conf” controls only the post-processing/metric stage.
- Write **annotated .mp4** (bbox + keypoints) and show them in an embedded player.
- Automatic per-video JSON cache (`runs/.../cache/*.json`) so rerunning the same video is instant; use the **Load JSON** button on the Table tab to pull cached predictions back in.
- Derive per-video metrics (freezing, research, rearing, nose distance, transitions). The Table tab now exposes UI controls for behaviour thresholds (rear ratio, research displacement, trim duration) plus a **Rebuild table** button to recalc instantly with new settings.
- **Classifier & SHAP** tab: RandomForest with CV-based hyperparameter tuning, stricter OOB bootstrap (AUC confidence intervals + hit stats), SHAP beeswarm, and feature-importance table.
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
python behav_ano.py
```

Typical workflow:
1. **Pick model** (`.pt`, Ultralytics YOLO **keypoints** model) and select device (`cuda`/`cpu`).
2. **Add groups** → choose folders with videos.
3. Click **Run full pipeline** (parallel across videos).
4. Open **Videos** tab to play **annotated** outputs.
5. See **Table** for per-video metrics. Adjust **conf**, **rear ratio threshold**, **research displacement threshold**, and **trim duration** as needed, then click **Rebuild table** (no inference rerun needed). Use **Load JSON** to import cached predictions and **Export table** to save the sheet.
6. In **Classifier & SHAP**, select two groups → **Train RF + SHAP**. You’ll get bootstrap stats, per-feature SHAP means, and a beeswarm figure.

---

## 📂 Outputs

```
runs/pose_app/<timestamp>/
├─ annot_<video>.mp4
├─ predictions_all.json          # merged predictions for the run
├─ cache/
│  └─ <video>_<hash>.json        # per-video intermediate cache
├─ table_data.xlsx
├─ combined_stats.png
└─ shap_summary_<pair>.png       # if SHAP installed / classifier run
```

---

## 🔧 Notes

- If you see a `torchvision::nms`/CUDA error, the app automatically retries on **CPU**.
- Increase **conf** if keypoints are noisy; decrease if detections are missing. Detection stays at `conf=0.10`, so this only affects post-processing.
- Behaviour thresholds (rear ratio, displacement, trim duration) live on the Table tab—change them and click **Rebuild table** to refresh metrics.
- Workers can be estimated via **Auto workers by VRAM** in the UI.
