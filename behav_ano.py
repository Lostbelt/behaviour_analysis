import os
import sys
import glob
import json
import time
from dataclasses import dataclass, field
from typing import Dict, List, Tuple, Optional
from threading import Thread, Lock
from queue import Queue

import numpy as np
import pandas as pd

from PySide6.QtCore import (Qt, QTimer)
from PySide6.QtGui import (QImage, QPixmap, QTextCursor)
from PySide6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, QLabel, QPushButton,
    QFileDialog, QDockWidget, QFormLayout, QDoubleSpinBox, QSpinBox, QTabWidget, QListWidget,
    QListWidgetItem, QLineEdit, QMessageBox, QTextEdit, QComboBox, QProgressBar, QHeaderView,
    QGroupBox, QTableWidget, QTableWidgetItem, QAbstractScrollArea,
    QSizePolicy, QAbstractItemView
)
import torch

from ultralytics import YOLO

import matplotlib.pyplot as plt

from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
from sklearn.ensemble import RandomForestClassifier

import shap


# ======== Helpers & constants ========
VIDEO_EXTS = {".mp4", ".mov", ".avi", ".MP4", ".MOV", ".AVI"}


def list_videos_in_folder(folder: str) -> List[str]:
    files = []
    for ext in VIDEO_EXTS:
        files.extend(glob.glob(os.path.join(folder, f"**/*{ext}"), recursive=True))
    # Deduplicate keeping stable order
    seen = set()
    out = []
    for p in sorted(files):
        if p not in seen:
            seen.add(p)
            out.append(p)
    return out


def safe_ultra_predict(model, frame, conf, device, on_cuda_error_fallback_cpu=True, log_cb=None):
    """
    Run single-frame inference with graceful CUDA->CPU fallback when torchvision::nms is missing.
    """
    try:
        return model(frame, verbose=False, conf=conf)
    except Exception as e:
        msg = str(e)
        if ("torchvision::nms" in msg or "Couldn't" in msg) and on_cuda_error_fallback_cpu:
            if log_cb:
                log_cb("CUDA NMS error detected; falling back to CPU and retrying...")
            try:
                model.to('cpu')
                return model(frame, verbose=False, conf=conf)
            except Exception as e2:
                if log_cb:
                    log_cb(f"CPU retry failed: {e2}")
                raise
        raise


def render_overlay(video_path: str, predictions: List[dict], out_path: str, fps: float, frame_size: Tuple[int,int]):
    """
    Draw bbox + keypoints on frames and write annotated .mp4
    predictions: list per-frame dict or [].
    """
    import cv2
    w, h = frame_size
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    writer = cv2.VideoWriter(out_path, fourcc, fps if fps > 0 else 25.0, (w, h))
    cap = cv2.VideoCapture(video_path)
    idx = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        det = predictions[idx] if idx < len(predictions) else []
        if isinstance(det, dict):
            # bbox
            bbox = det.get("bbox")
            if bbox:
                x1, y1, x2, y2 = map(int, bbox)
                cv2.rectangle(frame, (x1,y1), (x2,y2), (0,255,0), 2)
            # keypoints (list of [x,y,conf])
            kpts = det.get("keypoints")
            if isinstance(kpts, list):
                for kp in kpts:
                    if len(kp) >= 2:
                        x, y = int(kp[0]), int(kp[1])
                        cv2.circle(frame, (x,y), 3, (0,0,255), -1)
        writer.write(frame)
        idx += 1
    cap.release()
    writer.release()


def extract_primary_detection(result, conf: float):
    """Return the left-most detection dict or [] without relying on try/except."""
    if result is None:
        return []

    keypoints = getattr(result, "keypoints", None)
    boxes = getattr(result, "boxes", None)
    if keypoints is None or getattr(keypoints, "conf", None) is None:
        return []

    data = getattr(keypoints, "data", None)
    if data is None:
        return []

    kp_tensor = data.to('cpu') if hasattr(data, 'to') else data
    kp_np = kp_tensor.numpy() if hasattr(kp_tensor, 'numpy') else np.asarray(kp_tensor)
    if kp_np.ndim != 3 or kp_np.shape[0] == 0:
        return []

    mean_x = kp_np[:, :, 0].mean(axis=1)
    kp_idx = int(np.argmin(mean_x))
    kp_arr = kp_np[kp_idx]
    if kp_arr.ndim != 2 or kp_arr.shape[0] == 0:
        return []

    mask = kp_arr[:, 2] >= float(conf)
    filtered = kp_arr[mask]
    keypoints_out = filtered if filtered.size else kp_arr

    bbox = None
    if boxes is not None and getattr(boxes, 'xyxy', None) is not None:
        bbox_tensor = boxes.xyxy.to('cpu') if hasattr(boxes.xyxy, 'to') else boxes.xyxy
        bbox_np = bbox_tensor.numpy() if hasattr(bbox_tensor, 'numpy') else np.asarray(bbox_tensor)
        if bbox_np.ndim == 2 and bbox_np.shape[0] > kp_idx:
            bbox = bbox_np[kp_idx].tolist()

    return {
        "keypoints": keypoints_out.tolist(),
        "bbox": bbox,
    }


# ======= Your action/metric functions (unchanged logic) ========

def calibrate_bbox_size(
    predictions,
    max_frames: int = 20,
    min_aspect_ratio: float = 0.8,
    max_aspect_ratio: float = 1.2,
):
    diagonals = []
    count_frames = len(predictions)
    for i in range(count_frames):
        if len(diagonals) == max_frames:
            break
        frame_data = predictions[i]
        if not frame_data:
            continue
        bbox = frame_data.get("bbox") if isinstance(frame_data, dict) else None
        if bbox:
            x1, y1, x2, y2 = bbox
            w = x2 - x1
            h = y2 - y1
            if w > 0 and h > 0:
                ar = w / h
                if min_aspect_ratio <= ar <= max_aspect_ratio:
                    diag = float(np.sqrt(w**2 + h**2))
                    diagonals.append(diag)
    if not diagonals:
        return 'error'
    return float(np.mean(diagonals))


def raw_data_to_actions(
    predictions,
    frame_dims,
    mean_dist_body=None,
    conf_thr=0.5,
    frame_step=3,
    fps=30,
):
    frame_width, frame_height = frame_dims
    if mean_dist_body is None:
        mean_dist_body = calibrate_bbox_size(predictions, max_frames=50)
        if mean_dist_body == 'error' or mean_dist_body == 0:
            mean_dist_body = max(frame_height * 0.2, 1.0)

    nose_distances = []
    prev_nose = None
    prev_keypoints = np.zeros((17, 3))
    actions = []
    metrics_list = []
    buffer = []
    freeze_thresh = int(2 * fps)

    for idx, frame_data in enumerate(predictions):
        if frame_data and isinstance(frame_data, dict):
            kp = np.array(frame_data.get("keypoints", []))
            # kp is (K,3) after extraction; filter by confidence here too for safety
            if kp.ndim == 2 and kp.shape[1] >= 3:
                vis = kp[:, 2] >= conf_thr
                if kp.shape[0] > 2 and vis[2]:
                    nose_xy = kp[2, :2]
                    d = float(np.linalg.norm(nose_xy - prev_nose)) if prev_nose is not None else 0.0
                    prev_nose = nose_xy
                else:
                    d = 0.0
                    prev_nose = None
            else:
                d = 0.0
                prev_nose = None
        else:
            d = 0.0
            prev_nose = None
        nose_distances.append(d)
        buffer.append((frame_data, d))

        if (idx % frame_step == frame_step - 1) or (idx == len(predictions) - 1):
            first, _ = buffer[0]
            last, _ = buffer[-1]

            avg_disp = 0.0
            bbox_ratio = None
            head_disp = 0.0
            front_disp = 0.0
            action = [] if not last else "Miss"

            if first and last:
                x1, y1, x2, y2 = (last.get("bbox", [0, 0, frame_width, frame_height]) if isinstance(last, dict) else [0,0,frame_width,frame_height])
                h = frame_height
                bh = (y2 - y1)
                bbox_ratio = (h / bh) if bh > 0 else None

                cur_kp = np.array(last.get("keypoints", np.zeros((17, 3)))) if isinstance(last, dict) else np.zeros((17,3))
                # only use confident points to compute displacement
                if cur_kp.ndim == 2 and cur_kp.shape[1] >= 3:
                    vis_mask = cur_kp[:, 2] >= conf_thr
                else:
                    vis_mask = np.zeros((cur_kp.shape[0],), dtype=bool)

                prev_xy = prev_keypoints[:, :2]
                comb = np.concatenate((cur_kp[:, :2], prev_xy), axis=1) if cur_kp.size else np.zeros((0,4))
                pts = comb[vis_mask] if np.any(vis_mask) else comb
                vec = np.mean(pts[:, :2] - pts[:, 2:], axis=0) if pts.size else np.array([0.0,0.0])
                avg_disp = float(np.linalg.norm(vec) / mean_dist_body) if mean_dist_body else 0.0

                hi = [0, 1, 2, 3]
                if cur_kp.shape[0] > 3 and cur_kp.shape[1] > 2:
                    hc = (cur_kp[hi, 2] >= conf_thr)
                    if np.any(hc):
                        vh = np.mean(cur_kp[hi, :2][hc] - prev_keypoints[hi, :2][hc], axis=0)
                        head_disp = float(np.linalg.norm(vh) / mean_dist_body)

                fi = [5, 6, 7, 8, 9, 10]
                if cur_kp.shape[0] > max(fi) and cur_kp.shape[1] > 2:
                    fc = (cur_kp[fi, 2] >= conf_thr)
                    if np.any(fc):
                        vf = np.mean(cur_kp[fi, :2][fc] - prev_keypoints[fi, :2][fc], axis=0)
                        front_disp = float(np.linalg.norm(vf) / mean_dist_body)

                if bbox_ratio is not None and bbox_ratio < 2:
                    action = "Rear"
                else:
                    action = "Research" if avg_disp > 0.18 else "Still"

                prev_keypoints = cur_kp.copy()

            for _fd, _nd in buffer:
                metrics_list.append({
                    "avg_displacement": avg_disp,
                    "bbox_ratio": bbox_ratio,
                    "head_displacement": head_disp,
                    "front_limb_displacement": front_disp,
                })
                actions.append(action)

            buffer = []

    # Still → Freezing/Pausing
    out_actions = []
    i = 0
    n = len(actions)
    while i < n:
        a = actions[i]
        if a != "Still":
            out_actions.append(a)
            i += 1
        else:
            j = i
            while j < n and actions[j] == "Still":
                j += 1
            length = j - i
            label = "Still" if length >= freeze_thresh else "Pausing"
            out_actions.extend([label] * length)
            i = j

    return out_actions, metrics_list, nose_distances


def trim_actions_to_duration(actions, nose_distances, fps, duration_seconds=600):
    start_index = next((i for i, act in enumerate(actions) if act != []), None)
    if start_index is None:
        return [], []
    end_index = start_index + int(fps * duration_seconds)
    return actions[start_index:end_index], nose_distances[start_index:end_index]


def compute_action_stats(actions, nose_distances, fps=30):
    valid_actions = ["Still", "Research", "Rear", "Pausing"]
    segments = {a: [] for a in valid_actions}
    current_action = None
    current_duration = 0
    freeze_to_explore = 0

    for act in actions:
        if act == [] or act not in valid_actions:
            if current_action is not None:
                segments[current_action].append(current_duration)
                current_action = None
                current_duration = 0
            continue
        if current_action is None:
            current_action = act
            current_duration = 1
        elif current_action == act:
            current_duration += 1
        else:
            if current_action == "Still" and act in ("Research", "Rear"):
                freeze_to_explore += 1
            segments[current_action].append(current_duration)
            current_action = act
            current_duration = 1

    if current_action is not None:
        segments[current_action].append(current_duration)

    stats = {}
    stats["nose_distances_sum"] = float(sum(nose_distances))
    stats["freeze_to_explore"] = int(freeze_to_explore)

    for action in valid_actions:
        segs = segments[action]
        count = len(segs)
        abs_frames = int(sum(segs))
        mean_frames = (abs_frames / count) if count > 0 else 0.0
        abs_secs = abs_frames / fps
        mean_secs = mean_frames / fps
        stats[f"{action}_count"] = int(count)
        stats[f"{action}_abs_dur"] = float(abs_secs)
        stats[f"{action}_mean_dur"] = float(mean_secs)

    # merge Rear → Research
    stats["Research_count"] += stats["Rear_count"]
    stats["Research_abs_dur"] += stats["Rear_abs_dur"]
    stats["Research_mean_dur"] = (
        stats["Research_abs_dur"] / stats["Research_count"] if stats["Research_count"] > 0 else 0.0
    )
    return stats


# ======== Pose inference on a single video (single-frame) ========

def process_video_with_model(model_path: str, video_path: str, conf: float = 0.50,
                             device: str = 'cuda', log_cb=None, do_warmup=True) -> Dict:
    """
    Process one video frame-by-frame (no batching). Returns dict like V*_cuted_video_keypoints.json.
    Also returns per-frame bbox and keypoints (first/left-most instance),
    with keypoints filtered by confidence >= conf.
    """
    model = YOLO(model_path)
    try:
        model.to(device)
    except Exception as e:
        if log_cb:
            log_cb(f"Failed to move model to {device}: {e}. Using CPU.")
        device = 'cpu'
        model.to('cpu')

    # Warm-up once
    if do_warmup:
        import cv2
        dummy = np.zeros((256, 256, 3), dtype=np.uint8)
        _ = safe_ultra_predict(model, dummy, conf=conf, device=device, log_cb=log_cb)

    import cv2
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {video_path}")

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    results_per_frame = [[] for _ in range(total_frames)]

    idx = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        preds = safe_ultra_predict(model, frame, conf=conf, device=device, log_cb=log_cb)
        res = preds[0] if len(preds) > 0 else None

        detection = extract_primary_detection(res, conf)
        results_per_frame[idx] = detection
        idx += 1

    cap.release()

    video_data = {
        "video_path": video_path,
        "fps": fps,
        "frame_size": (width, height),
        "rat": results_per_frame,
    }
    return video_data


# ======== Data classes ========
@dataclass
class GroupConfig:
    name: str
    folder: str
    videos: List[str] = field(default_factory=list)


# ======== Video player (keeps aspect ratio, null-pixmap safe) ========
class VideoPlayer(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.label = QLabel("No video")
        self.label.setAlignment(Qt.AlignCenter)
        # prevent pixmap from inflating window min size
        self.label.setSizePolicy(QSizePolicy.Ignored, QSizePolicy.Ignored)
        self.label.setMinimumSize(0, 0)

        self.btn_play = QPushButton("▶")
        self.btn_pause = QPushButton("⏸")
        self.btn_stop = QPushButton("⏹")
        self.speed_combo = QComboBox()
        self.speed_combo.addItems(["0.5x","1x","2x"]) 

        top = QHBoxLayout()
        top.addWidget(self.btn_play)
        top.addWidget(self.btn_pause)
        top.addWidget(self.btn_stop)
        top.addStretch(1)
        top.addWidget(QLabel("Speed:"))
        top.addWidget(self.speed_combo)

        lay = QVBoxLayout(self)
        lay.addLayout(top)
        lay.addWidget(self.label, 1)

        self.cap = None
        self.timer = QTimer(self)
        self.timer.timeout.connect(self._next_frame)
        self.fps = 25

        self.btn_play.clicked.connect(self.play)
        self.btn_pause.clicked.connect(self.pause)
        self.btn_stop.clicked.connect(self.stop)

    def open(self, path: str):
        import cv2
        if self.cap is not None:
            self.cap.release()
            self.cap = None
        self.cap = cv2.VideoCapture(path)
        if not self.cap.isOpened():
            self.label.setText("Failed to open video")
            return
        self.fps = self.cap.get(cv2.CAP_PROP_FPS) or 25
        self.play()

    def play(self):
        sp = self.speed_combo.currentText()
        k = 1.0 if sp == '1x' else (0.5 if sp == '0.5x' else 2.0)
        interval = max(1, int(1000 / (self.fps * k)))
        self.timer.start(interval)

    def pause(self):
        self.timer.stop()

    def stop(self):
        self.timer.stop()
        if self.cap is not None:
            self.cap.set(1, 0)
        # Clear pixmap to avoid null-scaling warnings
        self.label.clear()
        self.label.setText("Stopped")

    def _next_frame(self):
        import cv2
        if self.cap is None:
            return
        ret, frame = self.cap.read()
        if not ret:
            # End of video: stop timer and clear pixmap to avoid null scaling
            self.pause()
            self.label.clear()
            self.label.setText("End of video")
            return
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        h, w, ch = rgb.shape
        qimg = QImage(rgb.data, w, h, ch * w, QImage.Format_RGB888)
        pix = QPixmap.fromImage(qimg)
        # Keep aspect ratio strictly, but only if valid
        if not pix.isNull():
            self.label.setPixmap(pix.scaled(self.label.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation))

    def resizeEvent(self, e):
        super().resizeEvent(e)
        pm = self.label.pixmap()
        if pm is not None and not pm.isNull():
            self.label.setPixmap(pm.scaled(self.label.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation))


# ======== Main window ========
class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Behavior Pipeline UI")
        self.resize(1400, 900)
        # Soft minimums so docking/table can't force huge window
        self.setMinimumSize(800, 500)

        # Work dir
        self.run_dir = os.path.join(os.getcwd(), 'runs', 'pose_app', time.strftime('%Y%m%d_%H%M%S'))
        os.makedirs(self.run_dir, exist_ok=True)

        # State
        self.model_path: Optional[str] = None
        self.device: str = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.conf: float = 0.95
        self.max_workers: int = 1

        self.groups: Dict[str, GroupConfig] = {}
        self.predictions: Dict[str, dict] = {}  # video_path -> video_data
        self.annotated_video_for: Dict[str, str] = {}  # original -> annotated path
        self.table_data: Optional[pd.DataFrame] = None

        # ====== Left panel ======
        left = QWidget()
        left_l = QVBoxLayout(left)

        # Model
        gb_model = QGroupBox("Model & Device")
        fm = QFormLayout(gb_model)
        self.le_model = QLineEdit()
        self.le_model.setPlaceholderText("best.pt")
        btn_pick_model = QPushButton("Pick .pt")
        btn_pick_model.clicked.connect(self.pick_model)
        wrow = QHBoxLayout()
        wrow.addWidget(self.le_model)
        wrow.addWidget(btn_pick_model)
        wroww = QWidget(); wroww.setLayout(wrow)
        fm.addRow("YOLO Pose:", wroww)

        self.cb_device = QComboBox()
        self.cb_device.addItems(['cpu'] + (['cuda'] if torch.cuda.is_available() else []))
        self.cb_device.setCurrentText(self.device)
        fm.addRow("Device:", self.cb_device)

        self.ds_conf = QDoubleSpinBox(); self.ds_conf.setRange(0.01, 1.0); self.ds_conf.setSingleStep(0.01); self.ds_conf.setValue(self.conf)
        fm.addRow("conf:", self.ds_conf)

        self.sb_workers = QSpinBox(); self.sb_workers.setRange(1, max(1, os.cpu_count() or 4)); self.sb_workers.setValue(1)
        fm.addRow("Worker threads:", self.sb_workers)

        left_l.addWidget(gb_model)

        # Groups
        gb_groups = QGroupBox("Groups")
        gl = QVBoxLayout(gb_groups)
        self.tabs_groups = QTabWidget()
        gl.addWidget(self.tabs_groups)
        row = QHBoxLayout()
        self.le_group_name = QLineEdit(); self.le_group_name.setPlaceholderText("e.g. DAT-KO")
        btn_add_group = QPushButton("Add group…")
        btn_add_group.clicked.connect(self.add_group)
        row.addWidget(self.le_group_name)
        row.addWidget(btn_add_group)
        rr = QWidget(); rr.setLayout(row)
        gl.addWidget(rr)
        left_l.addWidget(gb_groups, 1)

        # Run
        btn_run = QPushButton("▶ Run full pipeline")
        btn_run.clicked.connect(self.run_full_pipeline)
        left_l.addWidget(btn_run)

        # Logs
        self.te_log = QTextEdit(); self.te_log.setReadOnly(True)
        self.pb = QProgressBar(); self.pb.setRange(0, 0); self.pb.setVisible(False)
        left_l.addWidget(self.pb)
        left_l.addWidget(self.te_log, 1)

        dock_left = QDockWidget("Settings & Groups")
        dock_left.setWidget(left)
        # Make dock flexible
        dock_left.setMinimumSize(0, 0)
        dock_left.setSizePolicy(QSizePolicy.Preferred, QSizePolicy.Preferred)
        self.addDockWidget(Qt.LeftDockWidgetArea, dock_left)

        # ====== Central ======
        central = QWidget()
        cent_l = QVBoxLayout(central)
        central.setMinimumSize(0, 0)
        self.tabs_central = QTabWidget()

        # 1) Video (tabs inside)
        self.video_tabs = QTabWidget()
        tab_video_container = QWidget(); tvl = QVBoxLayout(tab_video_container); tvl.addWidget(self.video_tabs)
        self.tabs_central.addTab(tab_video_container, "Videos")

        # 2) Table
        tab_table = QWidget(); tbl = QVBoxLayout(tab_table)
        row_table = QHBoxLayout()
        self.btn_export_table = QPushButton("Export table…")
        self.btn_export_table.clicked.connect(self.export_table)
        row_table.addStretch(1)
        row_table.addWidget(self.btn_export_table)
        tbl.addLayout(row_table)
        self.table_widget = QTableWidget()
        # Geometry-safe table header & policies
        hdr = self.table_widget.horizontalHeader()
        hdr.setSectionResizeMode(QHeaderView.Interactive)
        hdr.setStretchLastSection(False)
        hdr.setMinimumSectionSize(60)
        vhr = self.table_widget.verticalHeader()
        vhr.setSectionResizeMode(QHeaderView.Fixed)
        vhr.setDefaultSectionSize(24)
        self.table_widget.setSizeAdjustPolicy(QAbstractScrollArea.AdjustIgnored)
        self.table_widget.setHorizontalScrollMode(QAbstractItemView.ScrollPerPixel)
        self.table_widget.setWordWrap(False)
        self.table_widget.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.table_widget.setMinimumSize(0, 0)
        tbl.addWidget(self.table_widget)
        self.tabs_central.addTab(tab_table, "Table")

        # 3) Classifier + SHAP
        tab_cls = QWidget(); cl = QVBoxLayout(tab_cls)
        top_cls = QHBoxLayout()
        self.cb_cls_a = QComboBox(); self.cb_cls_b = QComboBox()
        top_cls.addWidget(QLabel("Classifier:"))
        top_cls.addWidget(self.cb_cls_a); top_cls.addWidget(QLabel("vs")); top_cls.addWidget(self.cb_cls_b)
        self.btn_fit_cls = QPushButton("Train RF + SHAP")
        self.btn_fit_cls.clicked.connect(self.run_classifier)
        top_cls.addWidget(self.btn_fit_cls)
        top_cls.addStretch(1)
        cl.addLayout(top_cls)
        self.te_cls = QTextEdit(); self.te_cls.setReadOnly(True)
        cl.addWidget(self.te_cls, 1)
        self.lbl_shap = QLabel("SHAP summary will appear here")
        self.lbl_shap.setAlignment(Qt.AlignCenter)
        self.lbl_shap.setSizePolicy(QSizePolicy.Ignored, QSizePolicy.Ignored)
        self.lbl_shap.setMinimumSize(0, 0)
        cl.addWidget(self.lbl_shap, 1)
        self.tabs_central.addTab(tab_cls, "Classifier & SHAP")

        cent_l.addWidget(self.tabs_central)
        self.setCentralWidget(central)

        # Worker handles
        self._worker_threads: List[Thread] = []
        self._worker_lock = Lock()
        self._worker_queue = Queue()
        self._inference_running = False
        self._collector_timer = QTimer(self)
        self._collector_timer.timeout.connect(self._collect_worker_events)

    # ====== UI utils ======
    def log(self, msg: str):
        self.te_log.append(msg)
        self.te_log.moveCursor(QTextCursor.End)

    # ====== Model/parallelism ======
    def pick_model(self):
        path, _ = QFileDialog.getOpenFileName(self, "Pick YOLO .pt", "", "*.pt")
        if path:
            self.model_path = path
            self.le_model.setText(path)
            self.log(f"Model selected: {os.path.basename(path)}")
            # Warm-up once at selection time to compile kernels ahead
            dev = self.cb_device.currentText()
            mdl = YOLO(self.model_path)
            try:
                mdl.to(dev)
            except Exception as e:
                self.log(f"Model.to({dev}) failed: {e}. Using CPU.")
                mdl.to('cpu')
            dummy = np.zeros((256,256,3), dtype=np.uint8)
            _ = mdl(dummy, conf=0.50, verbose=False)
            self.log("Warm-up inference OK.")

    # ====== Groups ======
    def add_group(self):
        name = self.le_group_name.text().strip()
        if not name:
            QMessageBox.warning(self, "Group", "Enter group name")
            return
        folder = QFileDialog.getExistingDirectory(self, f"Pick folder with videos for {name}")
        if not folder:
            return
        vids = list_videos_in_folder(folder)
        if not vids:
            QMessageBox.warning(self, "Group", "No videos in folder")
            return
        gc = GroupConfig(name=name, folder=folder, videos=vids)
        self.groups[name] = gc
        tab = QWidget(); lay = QVBoxLayout(tab)
        lst = QListWidget()
        for v in vids:
            lst.addItem(QListWidgetItem(os.path.basename(v)))
        lst.itemClicked.connect(lambda it, g=name: self.open_video_in_view(g, it.text()))
        lay.addWidget(QLabel(f"Folder: {folder}"))
        lay.addWidget(lst)
        self.tabs_groups.addTab(tab, name)
        self.update_pairs()
        self.log(f"Group '{name}' added: {len(vids)} videos")

    def update_pairs(self):
        groups = list(self.groups.keys())
        for cb in (self.cb_cls_a, self.cb_cls_b):
            old = cb.currentText()
            cb.clear(); cb.addItems(groups)
            if old in groups:
                cb.setCurrentText(old)

    # ====== Video tabs ======
    def open_video_in_view(self, group_name: str, basename: str):
        path = None
        for v in self.groups[group_name].videos:
            if os.path.basename(v) == basename:
                path = v; break
        if not path:
            return
        # Prefer annotated version if available
        display_path = self.annotated_video_for.get(path, path)
        # Check existing tabs
        for i in range(self.video_tabs.count()):
            if self.video_tabs.tabText(i) == os.path.basename(display_path):
                w = self.video_tabs.widget(i)
                if isinstance(w, VideoPlayer):
                    w.open(display_path)
                self.video_tabs.setCurrentIndex(i)
                return
        vp = VideoPlayer()
        vp.open(display_path)
        self.video_tabs.addTab(vp, os.path.basename(display_path))
        self.video_tabs.setCurrentWidget(vp)

    # ====== Run pipeline ======
    def run_full_pipeline(self):
        if not self.model_path or not os.path.isfile(self.model_path):
            QMessageBox.warning(self, "Model", "Pick a valid .pt first")
            return
        if not self.groups:
            QMessageBox.warning(self, "Groups", "Add at least one group")
            return
        self.device = self.cb_device.currentText()
        self.conf = float(self.ds_conf.value())
        self.max_workers = int(self.sb_workers.value())

        # Collect all videos
        all_videos = []
        for g in self.groups.values():
            all_videos.extend(g.videos)
        if not all_videos:
            QMessageBox.warning(self, "Videos", "No videos to process")
            return

        # Start threaded, data-parallel inference
        self.pb.setVisible(True)
        self.te_log.clear()
        self.log("Starting inference...")
        self._start_data_parallel_infer(all_videos)

    def _start_data_parallel_infer(self, videos: List[str]):
        # Split video list into N shards
        n = max(1, self.max_workers)
        shards = [[] for _ in range(n)]
        for i, v in enumerate(videos):
            shards[i % n].append(v)

        self._inference_running = True
        self._worker_threads = []
        self._collector_timer.start(100)  # poll events

        def worker_fn(shard_idx: int, shard_videos: List[str]):
            shard_results = {}
            for v in shard_videos:
                self._worker_queue.put(("log", f"[W{shard_idx+1}] Processing: {os.path.basename(v)}"))
                try:
                    vd = process_video_with_model(self.model_path, v, conf=self.conf, device=self.device,
                                                  log_cb=lambda m: self._worker_queue.put(("log", f"[W{shard_idx+1}] {m}")))
                    shard_results[v] = vd
                    # write annotated mp4
                    out_mp4 = os.path.join(self.run_dir, f"annot_{os.path.basename(v)}")
                    render_overlay(v, vd['rat'], out_mp4, vd['fps'], vd['frame_size'])
                    self._worker_queue.put(("annot", v, out_mp4))
                    self._worker_queue.put(("one_done", v))
                except Exception as e:
                    self._worker_queue.put(("log", f"[W{shard_idx+1}] Error {os.path.basename(v)}: {e}"))
            self._worker_queue.put(("finished", shard_results))

        for i, shard in enumerate(shards):
            if not shard:
                continue
            t = Thread(target=worker_fn, args=(i, shard), daemon=True)
            self._worker_threads.append(t)

        for t in self._worker_threads:
            t.start()

    def _collect_worker_events(self):
        # Gather cross-thread messages, update UI safely in main thread
        any_active = any(t.is_alive() for t in self._worker_threads)
        while not self._worker_queue.empty():
            item = self._worker_queue.get()
            kind = item[0]
            if kind == "log":
                self.log(item[1])
            elif kind == "one_done":
                v = item[1]
                self.log(f"Done: {os.path.basename(v)}")
            elif kind == "annot":
                orig, ann = item[1], item[2]
                self.annotated_video_for[orig] = ann
            elif kind == "finished":
                part = item[1]
                self.predictions.update(part)
            else:
                pass

        if not any_active and self._inference_running:
            self._inference_running = False
            self._collector_timer.stop()
            self.pb.setVisible(False)
            # Save group-wise raw predictions
            for gname, g in self.groups.items():
                gjson = {}
                for v in g.videos:
                    if v in self.predictions:
                        gjson[v] = self.predictions[v]
                if gjson:
                    out = os.path.join(self.run_dir, f"predictions_{gname}.json")
                    with open(out, 'w', encoding='utf-8') as f:
                        json.dump(gjson, f, ensure_ascii=False)
                    self.log(f"Saved: {out}")
            # Build table
            self.build_table()
            self.log("Inference finished.")

    # ====== Postprocess & table ======
    def build_table(self):
        rows = []
        for gname, g in self.groups.items():
            for v in g.videos:
                if v not in self.predictions:
                    continue
                vd = self.predictions[v]
                rat_actions, metrics_list, nose_distances = raw_data_to_actions(
                    vd['rat'], frame_dims=vd['frame_size'], fps=int(round(vd.get('fps', 30))), conf_thr=self.conf
                )
                fps = int(round(vd.get('fps', 30)))
                trimmed, nose_trim = trim_actions_to_duration(rat_actions, nose_distances, fps)
                stats = compute_action_stats(trimmed, nose_trim, fps=fps)
                rows.append({
                    'id': gname,
                    'Still_count': stats['Still_count'],
                    'Still_mean_dur': stats['Still_mean_dur'],
                    'Still_abs_dur': stats['Still_abs_dur'],
                    'Pausing_count': stats['Pausing_count'],
                    'Pausing_mean_dur': stats['Pausing_mean_dur'],
                    'Pausing_abs_dur': stats['Pausing_abs_dur'],
                    'Research_count': stats['Research_count'],
                    'Research_mean_dur': stats['Research_mean_dur'],
                    'Research_abs_dur': stats['Research_abs_dur'],
                    'Rear_count': stats['Rear_count'],
                    'Rear_mean_dur': stats['Rear_mean_dur'],
                    'Rear_abs_dur': stats['Rear_abs_dur'],
                    'Nose-point distance': stats['nose_distances_sum'],
                    'Freeze > research transitions': stats['freeze_to_explore'],
                })
        if not rows:
            QMessageBox.information(self, "Postprocess", "No processed videos")
            return
        df = pd.DataFrame(rows)

        # Rename & order
        rename = {
            'Still_count': 'Freezing bouts frequency',
            'Still_mean_dur': 'Freezing bouts average duration, s',
            'Still_abs_dur': 'Freezing bouts total duration, s',
            'Research_count': 'Research activity bouts frequency',
            'Research_mean_dur': 'Research bouts average duration, s',
            'Rear_count': 'Rearing bouts frequency',
            'Rear_mean_dur': 'Rearing bouts average duration, s',
            'Rear_abs_dur': 'Rearing bouts total duration, s'
        }
        df.rename(columns=rename, inplace=True)

        order = [
            'id',
            'Research activity bouts frequency',
            'Research bouts average duration, s',
            'Nose-point distance',
            'Freezing bouts average duration, s',
            'Freezing bouts total duration, s',
            'Freezing bouts frequency',
            'Rearing bouts frequency',
            'Rearing bouts total duration, s',
            'Rearing bouts average duration, s',
        ]
        for c in order:
            if c not in df.columns:
                df[c] = np.nan
        df = df[order]
        self.table_data = df
        self.populate_table_widget()
        # Export
        out_xlsx = os.path.join(self.run_dir, 'table_data.xlsx')
        try:
            df.to_excel(out_xlsx, index=False)
            self.log(f"table_data.xlsx saved: {out_xlsx}")
        except Exception as exc:
            self.log(f"Failed to save table_data.xlsx: {exc}")
        # Update comboboxes
        self.update_pairs()

    def populate_table_widget(self):
        if self.table_data is None or self.table_data.empty:
            return
        df = self.table_data
        self.table_widget.clear()
        self.table_widget.setRowCount(len(df))
        self.table_widget.setColumnCount(len(df.columns))
        self.table_widget.setHorizontalHeaderLabels(df.columns.tolist())
        for r in range(len(df)):
            for c in range(len(df.columns)):
                val = df.iat[r, c]
                self.table_widget.setItem(r, c, QTableWidgetItem(str(val)))

    def export_table(self):
        if self.table_data is None:
            QMessageBox.information(self, "Export", "Generate the table first")
            return
        path, _ = QFileDialog.getSaveFileName(
            self,
            "Save table_data.xlsx",
            "table_data.xlsx",
            "Excel files (*.xlsx)"
        )
        if not path:
            return
        try:
            self.table_data.to_excel(path, index=False)
            self.log(f"Saved: {path}")
        except Exception as exc:
            QMessageBox.warning(self, "Export", f"Failed to save Excel file:\n{exc}")

    def set_image_to_label(self, path: str, label: QLabel):
        if not os.path.isfile(path):
            return
        pix = QPixmap(path)
        if pix.isNull():
            label.clear()
            label.setText(os.path.basename(path))
            return
        label.setPixmap(pix.scaled(label.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation))
        label.setToolTip(path)

    def _render_shap_summary(self, model, data: pd.DataFrame, title: str):
        """Render SHAP summary plot when dependencies are available."""
        if data is None or data.empty:
            self.te_cls.append("\n[SHAP skipped: insufficient samples]")
            return

        explainer = shap.TreeExplainer(model)
        shap_values = explainer.shap_values(data)
        if isinstance(shap_values, list):
            shap_values = shap_values[0]

        fig = plt.figure(figsize=(8, 6))
        shap.summary_plot(shap_values, data, feature_names=data.columns, plot_type='dot', show=False)
        plt.title(title)
        out = os.path.join(self.run_dir, 'shap_summary.png')
        plt.tight_layout()
        plt.savefig(out, dpi=300)
        plt.close(fig)
        self.set_image_to_label(out, self.lbl_shap)
        self.log(f"SHAP saved: {out}")

    # ====== Classifier + SHAP ======
    def run_classifier(self):
        if self.table_data is None or self.table_data.empty:
            QMessageBox.warning(self, "Table", "No data for classifier")
            return
        g1 = self.cb_cls_a.currentText(); g2 = self.cb_cls_b.currentText()
        if not g1 or not g2 or g1 == g2:
            QMessageBox.warning(self, "Compare", "Pick two different groups")
            return
        df = self.table_data
        pair_df = df[df['id'].isin([g1, g2])].copy()
        labels = pair_df['id']
        features = pair_df.drop(columns=['id'])
        scaler = StandardScaler(); X = pd.DataFrame(scaler.fit_transform(features), columns=features.columns)
        encoder = LabelEncoder(); y = encoder.fit_transform(labels)

        # Robustness for tiny datasets
        counts = pd.Series(y).value_counts()
        if counts.min() < 2 or len(pair_df) < 4:
            # Train on all, no split
            self.te_cls.clear()
            self.te_cls.append("[Warning] Too few samples per class for stratified split.\n"
                               f"Class distribution: {dict(zip(encoder.classes_, counts.reindex(range(len(encoder.classes_))).fillna(0).astype(int).tolist()))}\n"
                               "Training RandomForest on all available data; accuracy on held-out set is not computed.")
            rf = RandomForestClassifier(
                n_estimators=500, min_samples_leaf=1, min_samples_split=2,
                max_features='sqrt', bootstrap=True, random_state=5
            )
            rf.fit(X, y)
            self.te_cls.append("\nModel trained on full data.")
            # SHAP (optional)
            if len(X) >= 2:
                self._render_shap_summary(rf, X, f"{g2} vs {g1} (no test split)")
            return

        # Normal stratified split
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.3, stratify=y, random_state=5
        )
        rf = RandomForestClassifier(
            n_estimators=500, min_samples_leaf=4, min_samples_split=2,
            max_features='sqrt', bootstrap=True, random_state=5
        )
        rf.fit(X_train, y_train)
        preds = rf.predict(X_test)
        acc = accuracy_score(y_test, preds)
        rep = classification_report(y_test, preds, target_names=encoder.classes_, zero_division=0)
        self.te_cls.clear()
        self.te_cls.append(f"Accuracy: {acc:.4f}\n\n{rep}")

        self._render_shap_summary(rf, X_test, f"{g2} vs {g1}")

def main():
    app = QApplication(sys.argv)
    w = MainWindow()
    w.show()
    sys.exit(app.exec())


if __name__ == '__main__':
    main()
