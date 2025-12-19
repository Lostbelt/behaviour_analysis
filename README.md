# Behavior Pipeline UI

A GUI application for **YOLO Pose** inference and behavior analysis.

<div align="center">
  <img src="figures/main_window.png" width="600"/>
</div>



## Installation

Create an environment:
```bash
conda create -n lab_env Python=3.10
conda activate lab_env

git clone https://github.com/Lostbelt/behaviour_analysis.git
cd behaviour_analysis
pip install -r requirements.txt
# for gpu inference needs cuda PyTorch (choose the wheel appropriate for your system/driver)
# CUDA example:
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu126
```

---

> **Windows tip:** Install **PySide6 via pip** (not conda) to avoid Qt DLL conflicts.

---

## Getting Started

Run the app:
```bash
python behav_ano.py
```

Typical workflow:
1. **Pick model** (`.pt`, Ultralytics YOLO **keypoints** model) and select device (`cuda`/`cpu`).
2. **Add groups** → choose folders with videos.
3. Open **Videos** tab to play **annotated** outputs.
4. See **Table** for per-video metrics. Adjust **conf**, **rear ratio threshold**, **research displacement threshold**, and **trim duration** as needed, then click **Rebuild table** (no inference rerun needed). Use **Load JSON** to import cached predictions and **Export table** to save the sheet.
5. In **Classifier & SHAP**, select two groups → **Train RF + SHAP**.

---

You can download model weights and video examples on google drive [link](https://drive.google.com/drive/folders/19Ow9olyP1Yj2Pnr0URGYc5XFKusuZTtM?usp=sharing).

---
