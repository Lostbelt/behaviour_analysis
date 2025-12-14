import os
import sys
import glob
import json
import time
import hashlib
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
    QSizePolicy, QAbstractItemView, QGraphicsView, QGraphicsScene, QGraphicsPixmapItem,
    QGraphicsTextItem
)
import torch

from ultralytics import YOLO

import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import RepeatedStratifiedKFold, ParameterGrid, cross_val_score
from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn.ensemble import RandomForestClassifier

import shap
from joblib import Parallel, delayed


# ======== Helpers & constants ========
VIDEO_EXTS = {".mp4", ".mov", ".avi", ".MP4", ".MOV", ".AVI"}
DEFAULT_DETECTION_CONF = 0.10


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


def safe_ultra_predict(model, frame, conf: float = DEFAULT_DETECTION_CONF, device='cpu',
                       on_cuda_error_fallback_cpu=True, log_cb=None):
    """
    Run single-frame inference with graceful CUDA->CPU fallback when torchvision::nms is missing.
    `conf` here refers to the YOLO detection threshold and defaults to DEFAULT_DETECTION_CONF.
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


def extract_primary_detection(result, conf: Optional[float] = None):
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

    if conf is not None:
        mask = kp_arr[:, 2] >= float(conf)
        filtered = kp_arr[mask]
        keypoints_out = filtered if filtered.size else kp_arr
    else:
        keypoints_out = kp_arr

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
    rear_ratio_threshold: float = 2.0,
    research_disp_threshold: float = 0.18,
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

                if bbox_ratio is not None and bbox_ratio < rear_ratio_threshold:
                    action = "Rear"
                else:
                    action = "Research" if avg_disp > research_disp_threshold else "Still"

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
                             device: str = 'cuda', log_cb=None, do_warmup=True,
                             detect_conf: float = DEFAULT_DETECTION_CONF) -> Dict:
    """
    Process one video frame-by-frame (no batching). Returns dict like V*_cuted_video_keypoints.json.
    Also returns per-frame bbox and keypoints (first/left-most instance),
    while detection itself always runs with detect_conf (defaults to DEFAULT_DETECTION_CONF).
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
        _ = safe_ultra_predict(model, dummy, conf=detect_conf, device=device, log_cb=log_cb)

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

        preds = safe_ultra_predict(model, frame, conf=detect_conf, device=device, log_cb=log_cb)
        res = preds[0] if len(preds) > 0 else None

        detection = extract_primary_detection(res)
        results_per_frame[idx] = detection
        idx += 1

    cap.release()

    video_data = {
        "video_path": video_path,
        "fps": fps,
        "frame_size": (width, height),
        "rat": results_per_frame,
        "analysis_conf": conf,
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


class ZoomableGraphicsView(QGraphicsView):
    """Simple graphics view wrapper that supports wheel-based zoom and text placeholders."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self._scene = QGraphicsScene(self)
        self.setScene(self._scene)
        self._pixmap_item: Optional[QGraphicsPixmapItem] = None
        self._text_item: Optional[QGraphicsTextItem] = None
        self._zoom = 0
        self.setDragMode(QGraphicsView.ScrollHandDrag)
        self.setTransformationAnchor(QGraphicsView.AnchorUnderMouse)
        self.setResizeAnchor(QGraphicsView.AnchorUnderMouse)
        self.setBackgroundBrush(Qt.white)

    def setText(self, text: str):
        self._scene.clear()
        self._pixmap_item = None
        self._text_item = self._scene.addText(text)
        self._text_item.setDefaultTextColor(Qt.darkGray)
        self._scene.setSceneRect(self._text_item.boundingRect())
        self.resetTransform()
        self._zoom = 0

    def set_image(self, path: str):
        pix = QPixmap(path)
        if pix.isNull():
            basename = os.path.basename(path) if path else "Image not available"
            self.setText(basename)
            return
        self._scene.clear()
        self._pixmap_item = self._scene.addPixmap(pix)
        self._scene.setSceneRect(self._pixmap_item.boundingRect())
        self.resetTransform()
        self._zoom = 0
        self._fit_in_view()

    def wheelEvent(self, event):
        if self._pixmap_item is None:
            return super().wheelEvent(event)
        zoom_in_factor = 1.25
        zoom_out_factor = 1 / zoom_in_factor
        if event.angleDelta().y() > 0:
            self._zoom += 1
            factor = zoom_in_factor
        else:
            self._zoom -= 1
            factor = zoom_out_factor
        if self._zoom > 0:
            self.scale(factor, factor)
        else:
            self._zoom = 0
            self._fit_in_view()

    def resizeEvent(self, event):
        super().resizeEvent(event)
        if self._pixmap_item is not None and self._zoom == 0:
            self._fit_in_view()

    def _fit_in_view(self):
        if self._pixmap_item is None:
            return
        pix_rect = self._pixmap_item.boundingRect()
        if pix_rect.isNull():
            return
        self.fitInView(pix_rect, Qt.KeepAspectRatio)


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
        self.cache_dir = os.path.join(self.run_dir, 'cache')
        os.makedirs(self.cache_dir, exist_ok=True)

        # State
        self.model_path: Optional[str] = None
        self.device: str = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.conf: float = 0.95
        self.max_workers: int = 1
        self.rear_ratio_thr: float = 2.0
        self.research_disp_thr: float = 0.18
        self.trim_duration: float = 600.0  # seconds

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
        self.tabs_main = QTabWidget()

        # 1) Videos tab hosts video sub-tabs directly
        self.video_tabs = QTabWidget()
        self.tabs_main.addTab(self.video_tabs, "Videos")

        # 2) Table
        tab_table = QWidget(); tbl = QVBoxLayout(tab_table)
        row_table = QHBoxLayout()
        self.btn_load_json = QPushButton("Load JSON")
        self.btn_load_json.clicked.connect(self.load_predictions_json)
        self.btn_rebuild_table = QPushButton("Rebuild table")
        self.btn_rebuild_table.clicked.connect(self.build_table)
        self.btn_export_table = QPushButton("Export table")
        self.btn_export_table.clicked.connect(self.export_table)
        row_table.addWidget(self.btn_load_json)
        row_table.addWidget(self.btn_rebuild_table)
        row_table.addStretch(1)
        row_table.addWidget(self.btn_export_table)
        tbl.addLayout(row_table)
        thresh_form = QFormLayout()
        self.ds_rear_ratio = QDoubleSpinBox(); self.ds_rear_ratio.setRange(0.5, 10.0); self.ds_rear_ratio.setSingleStep(0.1); self.ds_rear_ratio.setValue(self.rear_ratio_thr)
        thresh_form.addRow("Rear ratio threshold:", self.ds_rear_ratio)
        self.ds_research_disp = QDoubleSpinBox(); self.ds_research_disp.setRange(0.0, 2.0); self.ds_research_disp.setSingleStep(0.01); self.ds_research_disp.setDecimals(3); self.ds_research_disp.setValue(self.research_disp_thr)
        thresh_form.addRow("Research displacement threshold:", self.ds_research_disp)
        self.ds_trim_minutes = QDoubleSpinBox(); self.ds_trim_minutes.setRange(0.5, 120.0); self.ds_trim_minutes.setSingleStep(0.5); self.ds_trim_minutes.setValue(self.trim_duration / 60.0)
        thresh_form.addRow("Trim duration (minutes):", self.ds_trim_minutes)
        tbl.addLayout(thresh_form)
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
        self.tabs_main.addTab(tab_table, "Table")

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
        self.tbl_bootstrap_stats = QTableWidget()
        self.tbl_bootstrap_stats.setColumnCount(0)
        self.tbl_bootstrap_stats.setRowCount(0)
        self.tbl_bootstrap_stats.setMinimumHeight(120)
        cl.addWidget(self.tbl_bootstrap_stats)
        self.tbl_shap_features = QTableWidget()
        self.tbl_shap_features.setColumnCount(0)
        self.tbl_shap_features.setRowCount(0)
        self.tbl_shap_features.setMinimumHeight(150)
        cl.addWidget(self.tbl_shap_features, 1)
        self.lbl_shap = ZoomableGraphicsView()
        self.lbl_shap.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.lbl_shap.setMinimumSize(0, 0)
        self.lbl_shap.setText("SHAP summary will appear here")
        cl.addWidget(self.lbl_shap, 1)
        self.tabs_main.addTab(tab_cls, "Classifier & SHAP")

        cent_l.addWidget(self.tabs_main)
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

    def _cache_prediction_path(self, video_path: str) -> str:
        base = os.path.splitext(os.path.basename(video_path))[0]
        digest = hashlib.sha1(video_path.encode('utf-8')).hexdigest()[:10]
        return os.path.join(self.cache_dir, f"{base}_{digest}.json")

    def _save_prediction_json(self, video_path: str, data: dict) -> str:
        cache_path = self._cache_prediction_path(video_path)
        os.makedirs(os.path.dirname(cache_path), exist_ok=True)
        with open(cache_path, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False)
        return cache_path

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
                    try:
                        cache_path = self._save_prediction_json(v, vd)
                        self._worker_queue.put(("log", f"[W{shard_idx+1}] Cached JSON: {cache_path}"))
                    except Exception as cache_exc:
                        self._worker_queue.put(("log", f"[W{shard_idx+1}] Failed to cache {os.path.basename(v)}: {cache_exc}"))
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
            # Save a single JSON with all predictions
            if self.predictions:
                out = os.path.join(self.run_dir, "predictions_all.json")
                try:
                    with open(out, 'w', encoding='utf-8') as f:
                        json.dump(self.predictions, f, ensure_ascii=False)
                    self.log(f"Saved: {out}")
                except Exception as exc:
                    self.log(f"Failed to save predictions_all.json: {exc}")
            # Build table
            self.build_table()
            self.log("Inference finished.")

    # ====== Postprocess & table ======
    def load_predictions_json(self):
        path, _ = QFileDialog.getOpenFileName(self, "Load cached predictions", "", "JSON files (*.json)")
        if not path:
            return
        try:
            with open(path, 'r', encoding='utf-8') as f:
                data = json.load(f)
        except Exception as exc:
            QMessageBox.warning(self, "Load JSON", f"Failed to read file:\n{exc}")
            return

        loaded = 0
        if isinstance(data, dict):
            if {"video_path", "rat", "frame_size"}.issubset(data.keys()):
                vp = data.get("video_path")
                if vp:
                    self.predictions[vp] = data
                    loaded = 1
            else:
                for vp, vd in data.items():
                    if isinstance(vd, dict):
                        self.predictions[vp] = vd
                        loaded += 1
        elif isinstance(data, list):
            for item in data:
                if isinstance(item, dict) and item.get("video_path"):
                    self.predictions[item["video_path"]] = item
                    loaded += 1

        if not loaded:
            QMessageBox.warning(self, "Load JSON", "File does not contain recognizable prediction data.")
            return

        self.log(f"Loaded {loaded} cached video(s) from {path}")
        self.build_table()

    def build_table(self):
        self.conf = float(self.ds_conf.value())
        self.rear_ratio_thr = float(self.ds_rear_ratio.value())
        self.research_disp_thr = float(self.ds_research_disp.value())
        self.trim_duration = float(self.ds_trim_minutes.value()) * 60.0
        rows = []
        for gname, g in self.groups.items():
            for v in g.videos:
                if v not in self.predictions:
                    continue
                vd = self.predictions[v]
                rat_actions, metrics_list, nose_distances = raw_data_to_actions(
                    vd['rat'],
                    frame_dims=vd['frame_size'],
                    fps=int(round(vd.get('fps', 30))),
                    conf_thr=self.conf,
                    rear_ratio_threshold=self.rear_ratio_thr,
                    research_disp_threshold=self.research_disp_thr,
                )
                fps = int(round(vd.get('fps', 30)))
                trimmed, nose_trim = trim_actions_to_duration(
                    rat_actions,
                    nose_distances,
                    fps,
                    duration_seconds=self.trim_duration,
                )
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

    def _set_table_widget_from_df(self, widget: QTableWidget, df: Optional[pd.DataFrame]):
        widget.clear()
        if df is None or df.empty:
            widget.setRowCount(0)
            widget.setColumnCount(0)
            return
        widget.setRowCount(len(df))
        widget.setColumnCount(len(df.columns))
        widget.setHorizontalHeaderLabels(df.columns.tolist())
        for r in range(len(df)):
            for c in range(len(df.columns)):
                widget.setItem(r, c, QTableWidgetItem(str(df.iat[r, c])))

    def populate_bootstrap_results(self, metrics_df: Optional[pd.DataFrame], shap_df: Optional[pd.DataFrame]):
        self._set_table_widget_from_df(self.tbl_bootstrap_stats, metrics_df)
        self._set_table_widget_from_df(self.tbl_shap_features, shap_df)

    def set_image_to_label(self, path: str, label: QLabel):
        if not os.path.isfile(path):
            if hasattr(label, "setText"):
                label.setText("Image not found")
            return
        if isinstance(label, ZoomableGraphicsView):
            label.set_image(path)
            label.setToolTip(path)
            return
        pix = QPixmap(path)
        if pix.isNull():
            label.clear()
            label.setText(os.path.basename(path))
            return
        label.setPixmap(pix.scaled(label.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation))
        label.setToolTip(path)

    def _render_shap_summary(self, shap_values: np.ndarray, features: pd.DataFrame,
                              title: str, x_limits: Optional[Tuple[float, float]] = None):
        """Render SHAP summary plot when matrix + feature frame are provided."""
        if shap_values is None or shap_values.size == 0 or features is None or features.empty:
            self.lbl_shap.setText("No SHAP data")
            return

        fig = plt.figure(figsize=(7, 6))
        shap.summary_plot(
            shap_values,
            features,
            feature_names=list(features.columns),
            plot_type='dot',
            show=False
        )
        ax = plt.gca()
        ax.axvline(0.0, color='grey', linewidth=1)
        if x_limits:
            ax.set_xlim(*x_limits)
        plt.title(title, fontweight="bold")
        out = os.path.join(self.run_dir, f'shap_summary_{title.replace(" ", "_")}.png')
        plt.tight_layout()
        plt.savefig(out, dpi=350, bbox_inches="tight", facecolor="white")
        plt.close(fig)
        self.set_image_to_label(out, self.lbl_shap)
        self.log(f"SHAP saved: {out}")

    # ====== Classifier + SHAP ======
    def run_classifier(self):
        if self.table_data is None or self.table_data.empty:
            QMessageBox.warning(self, "Table", "No data for classifier")
            return
        g_pos = self.cb_cls_a.currentText(); g_neg = self.cb_cls_b.currentText()
        if not g_pos or not g_neg or g_pos == g_neg:
            QMessageBox.warning(self, "Compare", "Pick two different groups")
            return
        df = self.table_data.copy()
        pair_df = df[df['id'].isin([g_pos, g_neg])].copy()
        if pair_df.empty:
            QMessageBox.warning(self, "Table", "Selected groups are missing in the table")
            return

        labels = pair_df['id']
        features = pair_df.drop(columns=['id'])
        if features.empty:
            QMessageBox.warning(self, "Table", "No feature columns found")
            return

        scaler = StandardScaler()
        X = pd.DataFrame(scaler.fit_transform(features), columns=features.columns)
        encoder = LabelEncoder()
        y = encoder.fit_transform(labels)
        classes = list(encoder.classes_)
        if len(classes) != 2 or g_pos not in classes or g_neg not in classes:
            QMessageBox.warning(self, "Compare", "Classifier currently supports exactly two groups")
            return

        n_samples = len(X)
        if n_samples < 6:
            QMessageBox.warning(self, "Classifier", "Need at least 6 samples for bootstrap routine")
            return

        # Hyperparameter + bootstrap settings (aligned with user-provided snippet)
        RANDOM_STATE = 60
        cv = RepeatedStratifiedKFold(n_splits=5, n_repeats=5, random_state=RANDOM_STATE)
        param_grid = dict(
            n_estimators=[800],
            max_depth=[12],
            min_samples_split=[2, 4],
            min_samples_leaf=[1, 2, 4],
        )
        base_rf_kw = dict(
            max_features='sqrt',
            bootstrap=True,
            n_jobs=-1,
        )
        bootstrap_reps = 800
        min_per_class_oob = 2
        min_oob_size = 4
        max_redraws = 1000
        n_jobs = -1

        X_np = X.values
        classes_idx = [np.where(y == c)[0] for c in np.unique(y)]
        if any(len(idx) == 0 for idx in classes_idx):
            QMessageBox.warning(self, "Classifier", "One of the classes is empty after encoding.")
            return

        # Stage 0: hyperparameter tuning via CV AUC
        best_score, best_params = -np.inf, None
        self.te_cls.clear()
        self.te_cls.append("Running CV-based hyperparameter search...\n")
        for params in ParameterGrid(param_grid):
            rf = RandomForestClassifier(random_state=RANDOM_STATE, **base_rf_kw, **params)
            try:
                scores = cross_val_score(rf, X_np, y, cv=cv, scoring='roc_auc', n_jobs=base_rf_kw.get('n_jobs', None))
            except Exception as exc:
                self.log(f"Grid search failed for {params}: {exc}")
                continue
            mean_score = float(np.mean(scores)) if len(scores) else float('-inf')
            if mean_score > best_score:
                best_score, best_params = mean_score, params
        if best_params is None:
            QMessageBox.warning(self, "Classifier", "Hyperparameter search failed.")
            return
        self.te_cls.append(f"[GridSearch] {g_pos} vs {g_neg}: best CV AUC = {best_score:.3f} with {best_params}\n")

        rf_fixed = dict(**base_rf_kw, **best_params)
        pos_index = classes.index(g_pos)

        def bootstrap_iteration(seed_offset: int):
            rng = np.random.default_rng(RANDOM_STATE + seed_offset)
            redraws = 0
            while redraws < max_redraws:
                boot_indices = []
                for cls_idx in classes_idx:
                    boot_indices.append(rng.choice(cls_idx, size=len(cls_idx), replace=True))
                idx_boot = np.concatenate(boot_indices)
                oob_mask = np.ones(n_samples, dtype=bool)
                oob_mask[idx_boot] = False
                idx_oob = np.where(oob_mask)[0]
                if idx_oob.size < min_oob_size:
                    redraws += 1
                    continue
                ok = True
                for cls_val in np.unique(y):
                    if np.sum(y[idx_oob] == cls_val) < min_per_class_oob:
                        ok = False
                        break
                if ok:
                    break
                redraws += 1
            else:
                return None

            clf = RandomForestClassifier(**rf_fixed, random_state=RANDOM_STATE + seed_offset)
            clf.fit(X_np[idx_boot], y[idx_boot])

            try:
                preds = clf.predict_proba(X_np[idx_oob])[:, 1]
                metric = roc_auc_score(y[idx_oob], preds)
            except Exception:
                preds = clf.predict(X_np[idx_oob])
                metric = accuracy_score(y[idx_oob], preds)

            explainer = shap.TreeExplainer(clf)
            shap_vals = explainer.shap_values(X_np[idx_oob])
            if isinstance(shap_vals, list):
                shap_pos = shap_vals[pos_index]
            else:
                shap_pos = shap_vals
            if shap_pos.ndim == 3:
                shap_pos = shap_pos[:, :, pos_index]

            return {
                "metric": metric,
                "idx_oob": idx_oob,
                "shap_vals": shap_pos,
                "redraws": redraws + 1,
            }

        self.te_cls.append(f"Starting bootstrap ({bootstrap_reps} reps)... this may take a while.\n")
        try:
            results = Parallel(n_jobs=n_jobs, backend='loky')(
                delayed(bootstrap_iteration)(i) for i in range(bootstrap_reps)
            )
        except Exception as exc:
            QMessageBox.warning(self, "Bootstrap", f"Parallel bootstrap failed:\n{exc}")
            return

        rep_metrics = []
        shap_sum = np.zeros((n_samples, X.shape[1]), dtype=float)
        shap_cnt = np.zeros(n_samples, dtype=int)
        oob_hits = np.zeros(n_samples, dtype=int)
        total_attempts = 0
        for res in results:
            if res is None:
                continue
            rep_metrics.append(res["metric"])
            idx_oob = res["idx_oob"]
            shap_sum[idx_oob] += res["shap_vals"]
            shap_cnt[idx_oob] += 1
            oob_hits[idx_oob] += 1
            total_attempts += res["redraws"]

        rep_metrics = np.asarray(rep_metrics, dtype=float)
        accepted_reps = int(rep_metrics.size)
        if accepted_reps == 0:
            QMessageBox.warning(self, "Bootstrap", "No bootstrap replicates met the OOB criteria.")
            return

        perf_mean = float(np.mean(rep_metrics))
        ci_lo, ci_hi = np.quantile(rep_metrics, [0.025, 0.975])
        hits_min = int(np.min(oob_hits))
        hits_med = float(np.median(oob_hits))
        hits_mean = float(np.mean(oob_hits))

        summary_text = (
            f"=== {g_pos} vs {g_neg} ===\n"
            f"AUC (OOB bootstrap): {perf_mean:.3f} [{ci_lo:.3f}, {ci_hi:.3f}] "
            f"over {accepted_reps} reps (requested {bootstrap_reps})\n"
            f"OOB hits - min/med/mean: {hits_min}/{hits_med:.1f}/{hits_mean:.1f}\n"
            f"Tuned params: {best_params}\n"
        )
        self.te_cls.append(summary_text)

        metrics_df = pd.DataFrame([
            {
                "metric": "AUC (bootstrap)",
                "mean": round(perf_mean, 4),
                "ci_low": round(ci_lo, 4),
                "ci_high": round(ci_hi, 4),
                "accepted_reps": accepted_reps,
                "requested_reps": bootstrap_reps,
            },
            {
                "metric": "OOB hits (min/med/mean)",
                "mean": round(hits_mean, 2),
                "ci_low": hits_min,
                "ci_high": round(hits_med, 2),
                "accepted_reps": hits_min,
                "requested_reps": hits_med,
            },
        ])

        has_oob = shap_cnt > 0
        shap_avg = np.zeros_like(shap_sum)
        shap_avg[has_oob] = shap_sum[has_oob] / np.maximum(shap_cnt[has_oob, None], 1)
        shap_for_plot = shap_avg[has_oob]
        X_for_plot = X.iloc[has_oob]

        shap_df = None
        if shap_for_plot.size:
            mean_abs = np.mean(np.abs(shap_for_plot), axis=0)
            mean_signed = np.mean(shap_for_plot, axis=0)
            shap_df = (
                pd.DataFrame({
                    "feature": X.columns,
                    "mean_abs_SHAP": mean_abs,
                    "mean_SHAP": mean_signed,
                })
                .sort_values("mean_abs_SHAP", ascending=False)
                .reset_index(drop=True)
            )

        self.populate_bootstrap_results(metrics_df, shap_df)

        if shap_for_plot.size:
            x_min = float(np.nanmin(shap_for_plot))
            x_max = float(np.nanmax(shap_for_plot))
            self._render_shap_summary(
                shap_for_plot,
                X_for_plot,
                f"{g_pos} vs {g_neg}",
                x_limits=(x_min, x_max),
            )
        else:
            self.lbl_shap.setText("No SHAP data from bootstrap.")

def main():
    app = QApplication(sys.argv)
    w = MainWindow()
    w.show()
    sys.exit(app.exec())


if __name__ == '__main__':
    main()
