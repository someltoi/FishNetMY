import json
import os
import shutil
import sys
import threading
import tkinter as tk
from abc import ABC, abstractmethod
from pathlib import Path
from queue import Empty, Queue
from tkinter import filedialog, messagebox, ttk

import cv2
import numpy as np
import torch
from PIL import Image, ImageTk
from ultralytics import YOLO
from torchvision.models.detection import fasterrcnn_resnet50_fpn, maskrcnn_resnet50_fpn
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models.detection.mask_rcnn import MaskRCNNPredictor
from torchvision.transforms import functional as F


def center_window(window, parent):
    """Centre *window* over *parent* after geometry is known."""
    window.update_idletasks()
    pw = parent.winfo_width()
    ph = parent.winfo_height()
    px = parent.winfo_rootx()
    py = parent.winfo_rooty()
    ww = window.winfo_width()
    wh = window.winfo_height()
    x = px + (pw - ww) // 2
    y = py + (ph - wh) // 2
    window.geometry(f"+{x}+{y}")


class ProgressDialog:
    def __init__(self, parent, title, initial_text="Working..."):
        self.window = tk.Toplevel(parent)
        self.window.title(title)
        self.window.geometry("420x160")
        self.window.resizable(False, False)
        self.window.transient(parent)
        self.window.grab_set()
        center_window(self.window, parent)

        self.lbl_status = tk.Label(self.window, text=initial_text, anchor="w")
        self.lbl_status.pack(fill=tk.X, padx=16, pady=(16, 8))

        self.progress = ttk.Progressbar(self.window, orient=tk.HORIZONTAL, mode="determinate", length=380)
        self.progress.pack(padx=16, pady=6)

        self.lbl_percent = tk.Label(self.window, text="0%", anchor="e")
        self.lbl_percent.pack(fill=tk.X, padx=16, pady=(2, 0))

        self.btn_close = tk.Button(self.window, text="Close", state=tk.DISABLED, command=self.window.destroy)
        self.btn_close.pack(pady=(10, 10))

        self.window.protocol("WM_DELETE_WINDOW", lambda: None)

    def set_determinate(self, value, text):
        self.progress.configure(mode="determinate")
        self.progress["value"] = max(0, min(100, value))
        self.lbl_percent.config(text=f"{int(value)}%")
        self.lbl_status.config(text=text)
        self.window.update_idletasks()

    def set_indeterminate(self, text):
        self.progress.configure(mode="indeterminate")
        self.progress.start(12)
        self.lbl_percent.config(text="")
        self.lbl_status.config(text=text)
        self.window.update_idletasks()

    def finish(self, text, success=True):
        self.progress.stop()
        if success:
            self.progress.configure(mode="determinate")
            self.progress["value"] = 100
            self.lbl_percent.config(text="100%")
        self.lbl_status.config(text=text)
        self.btn_close.config(state=tk.NORMAL)
        self.window.protocol("WM_DELETE_WINDOW", self.window.destroy)
        self.window.update_idletasks()

    def close(self):
        if self.window.winfo_exists():
            self.window.destroy()


class BaseRunner(ABC):
    def __init__(self, model_path):
        self.model_path = model_path

    @abstractmethod
    def predict(self, frame_bgr, conf_threshold):
        raise NotImplementedError


class YOLORunner(BaseRunner):
    def __init__(self, model_path, device, class_names=None, class_colors=None):
        super().__init__(model_path)
        self.device = device
        # Pass the device string directly to Ultralytics so it runs on GPU
        # when available.  torch.device → str e.g. "cuda:0" or "cpu".
        device_str = str(device) if device is not None else "cpu"
        self.model = YOLO(model_path)
        self.model.to(device_str)
        self.class_names = class_names or {}
        self.class_colors = class_colors or {}

    def predict(self, frame_bgr, conf_threshold):
        device_str = str(self.device) if self.device is not None else "cpu"
        results = self.model.predict(frame_bgr, conf=conf_threshold,
                                     device=device_str, verbose=False)
        return results[0].plot()


class FasterRCNNRunner(BaseRunner):
    def __init__(self, model_path, device, class_names=None, class_colors=None):
        super().__init__(model_path)
        self.device = device
        self.class_names = class_names or {}
        self.class_colors = class_colors or {}
        self.model = self._load_model(model_path)

    def _extract_state_dict(self, checkpoint_obj):
        if isinstance(checkpoint_obj, dict) and "model_state_dict" in checkpoint_obj:
            return checkpoint_obj["model_state_dict"]
        if isinstance(checkpoint_obj, dict):
            return checkpoint_obj
        raise ValueError("Unsupported Faster R-CNN checkpoint format")

    def _load_model(self, model_path):
        # Use weights_only=False to support older checkpoint formats
        try:
            checkpoint = torch.load(model_path, map_location=self.device, weights_only=False)
        except TypeError:
            # Fallback for older PyTorch versions that don't support weights_only
            checkpoint = torch.load(model_path, map_location=self.device)
        state_dict = self._extract_state_dict(checkpoint)

        if "roi_heads.box_predictor.cls_score.weight" not in state_dict:
            raise ValueError("Checkpoint missing Faster R-CNN classifier weights")

        num_classes = state_dict["roi_heads.box_predictor.cls_score.weight"].shape[0]
        model = fasterrcnn_resnet50_fpn(weights=None, weights_backbone=None)
        in_features = model.roi_heads.box_predictor.cls_score.in_features
        model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)
        model.load_state_dict(state_dict, strict=False)
        model.to(self.device)
        model.eval()
        return model

    def predict(self, frame_bgr, conf_threshold):
        rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        tensor = F.to_tensor(rgb).to(self.device)
        with torch.no_grad():
            output = self.model([tensor])[0]

        rendered = frame_bgr.copy()
        boxes = output["boxes"].detach().cpu().numpy() if "boxes" in output else np.array([])
        labels = output["labels"].detach().cpu().numpy() if "labels" in output else np.array([])
        scores = output["scores"].detach().cpu().numpy() if "scores" in output else np.array([])

        for box, label, score in zip(boxes, labels, scores):
            if float(score) < conf_threshold:
                continue
            x1, y1, x2, y2 = [int(v) for v in box]
            
            # Get color for this class
            color = self.class_colors.get(int(label), (0, 200, 0))
            # Get class name or default to cls{label}
            class_name = self.class_names.get(int(label), f"cls {int(label)}")
            
            cv2.rectangle(rendered, (x1, y1), (x2, y2), color, 2)
            cv2.putText(
                rendered,
                f"{class_name} {score:.2f}",
                (x1, max(20, y1 - 6)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                color,
                1,
                cv2.LINE_AA,
            )
        return rendered


class MaskRCNNRunner(BaseRunner):
    def __init__(self, model_path, device, class_names=None, class_colors=None):
        super().__init__(model_path)
        self.device = device
        self.class_names = class_names or {}
        self.class_colors = class_colors or {}
        self.model = self._load_model(model_path)

    def _extract_state_dict(self, checkpoint_obj):
        if isinstance(checkpoint_obj, dict) and "model_state_dict" in checkpoint_obj:
            return checkpoint_obj["model_state_dict"]
        if isinstance(checkpoint_obj, dict):
            return checkpoint_obj
        raise ValueError("Unsupported Mask R-CNN checkpoint format")

    def _load_model(self, model_path):
        # Use weights_only=False to support older checkpoint formats
        try:
            checkpoint = torch.load(model_path, map_location=self.device, weights_only=False)
        except TypeError:
            # Fallback for older PyTorch versions that don't support weights_only
            checkpoint = torch.load(model_path, map_location=self.device)
        state_dict = self._extract_state_dict(checkpoint)

        if "roi_heads.box_predictor.cls_score.weight" not in state_dict:
            raise ValueError("Checkpoint missing Mask R-CNN classifier weights")

        num_classes = state_dict["roi_heads.box_predictor.cls_score.weight"].shape[0]
        model = maskrcnn_resnet50_fpn(weights=None, weights_backbone=None)

        in_features_box = model.roi_heads.box_predictor.cls_score.in_features
        model.roi_heads.box_predictor = FastRCNNPredictor(in_features_box, num_classes)

        in_features_mask = model.roi_heads.mask_predictor.conv5_mask.in_channels
        hidden_layer = 256
        model.roi_heads.mask_predictor = MaskRCNNPredictor(in_features_mask, hidden_layer, num_classes)

        model.load_state_dict(state_dict, strict=False)
        model.to(self.device)
        model.eval()
        return model

    @staticmethod
    def _label_to_color(label):
        base = int(label) * 37
        return (50 + (base * 3) % 205, 50 + (base * 5) % 205, 50 + (base * 7) % 205)

    def predict(self, frame_bgr, conf_threshold):
        rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        tensor = F.to_tensor(rgb).to(self.device)
        with torch.no_grad():
            output = self.model([tensor])[0]

        rendered = frame_bgr.copy()
        boxes = output.get("boxes", torch.empty((0, 4))).detach().cpu().numpy()
        labels = output.get("labels", torch.empty((0,))).detach().cpu().numpy()
        scores = output.get("scores", torch.empty((0,))).detach().cpu().numpy()
        masks = output.get("masks", torch.empty((0, 1, frame_bgr.shape[0], frame_bgr.shape[1]))).detach().cpu().numpy()

        for idx, (box, label, score) in enumerate(zip(boxes, labels, scores)):
            if float(score) < conf_threshold:
                continue

            # Get color for this class (use custom or generate default)
            color = self.class_colors.get(int(label), self._label_to_color(label))
            # Get class name or default to cls{label}
            class_name = self.class_names.get(int(label), f"cls {int(label)}")
            
            x1, y1, x2, y2 = [int(v) for v in box]

            if idx < len(masks):
                mask = masks[idx, 0] > 0.5
                rendered[mask] = (0.55 * np.array(color) + 0.45 * rendered[mask]).astype(np.uint8)

            cv2.rectangle(rendered, (x1, y1), (x2, y2), color, 2)
            cv2.putText(
                rendered,
                f"{class_name} {score:.2f}",
                (x1, max(20, y1 - 6)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                color,
                1,
                cv2.LINE_AA,
            )
        return rendered


class ModelRegistry:
    def __init__(self, project_root):
        self.project_root = project_root
        self.models_root = os.path.join(project_root, "models")
        
        # Store user uploads and config in AppData (persistent across app updates)
        if sys.platform == "win32":
            app_data_dir = os.path.join(os.environ.get("APPDATA", os.path.expanduser("~")), "FishNetMy")
        else:
            app_data_dir = os.path.expanduser("~/.fishnetmy")
        
        self.custom_models_root = os.path.join(app_data_dir, "models")
        self.config_path = os.path.join(app_data_dir, "gui_model_registry.json")
        os.makedirs(self.custom_models_root, exist_ok=True)
        os.makedirs(app_data_dir, exist_ok=True)
        self.config = self._load_config()

    def _load_config(self):
        if not os.path.exists(self.config_path):
            return {
                "uploaded_models": [],
                "last_selected_model": "",
                "model_configs": {}
            }
        try:
            with open(self.config_path, "r", encoding="utf-8") as f:
                data = json.load(f)
            data.setdefault("uploaded_models", [])
            data.setdefault("last_selected_model", "")
            data.setdefault("model_configs", {})
            return data
        except Exception:
            return {
                "uploaded_models": [],
                "last_selected_model": "",
                "model_configs": {}
            }

    def save_config(self):
        with open(self.config_path, "w", encoding="utf-8") as f:
            json.dump(self.config, f, indent=2)

    @staticmethod
    def model_type_from_path(path):
        lower = path.lower()
        suffix = Path(path).suffix.lower()
        if suffix == ".pt":
            return "YOLOv11"
        if "mask" in lower:
            return "Mask R-CNN"
        if "fast" in lower or "faster" in lower:
            return "Faster R-CNN"
        if suffix == ".pth":
            return "PyTorch (.pth)"
        return "Unknown"

    def _safe_rel_path(self, path):
        """Return project-relative path when possible; otherwise absolute path."""
        try:
            return os.path.relpath(path, self.project_root)
        except ValueError:
            return path

    def remove_model(self, model_path):
        """Remove a model from the registry and delete the file."""
        model_path = os.path.abspath(model_path)
        
        # Remove from uploaded_models list
        self.config["uploaded_models"] = [
            m for m in self.config["uploaded_models"] 
            if os.path.abspath(m.get("path", "")) != model_path
        ]
        
        # Remove from model_configs
        if model_path in self.config["model_configs"]:
            del self.config["model_configs"][model_path]
        
        # Delete the actual file if it exists
        if os.path.exists(model_path):
            try:
                os.remove(model_path)
            except OSError as e:
                raise Exception(f"Failed to delete model file: {e}")
        
        self.save_config()

    def get_model_config(self, model_path):
        """Get class names and colors config for a model."""
        model_path = os.path.abspath(model_path)
        return self.config["model_configs"].get(model_path, {})

    def save_model_config(self, model_path, class_names, class_colors):
        """Save class names and colors config for a model."""
        model_path = os.path.abspath(model_path)
        if model_path not in self.config["model_configs"]:
            self.config["model_configs"][model_path] = {}
        self.config["model_configs"][model_path].update({
            "class_names": class_names,
            "class_colors": class_colors
        })
        self.save_config()

    def get_model_alias(self, model_path):
        """Return the user-defined display alias for a model, or '' if none set."""
        model_path = os.path.abspath(model_path)
        cfg = self.config["model_configs"].get(model_path, {})
        return cfg.get("alias", "")

    def save_model_alias(self, model_path, alias):
        """Persist a display alias for a model (empty string clears it)."""
        model_path = os.path.abspath(model_path)
        if model_path not in self.config["model_configs"]:
            self.config["model_configs"][model_path] = {}
        self.config["model_configs"][model_path]["alias"] = alias.strip()
        self.save_config()

    def _display_name(self, model_path, file_name=None):
        """Return 'alias' if set, otherwise 'filename [type]'."""
        if file_name is None:
            file_name = os.path.basename(model_path)
        alias = self.get_model_alias(model_path)
        if alias:
            return alias
        return f"{file_name} [{self.model_type_from_path(model_path)}]"

    def discover_models(self):
        discovered = []
        if os.path.isdir(self.models_root):
            for root, _, files in os.walk(self.models_root):
                for file_name in files:
                    if not file_name.lower().endswith((".pt", ".pth")):
                        continue
                    path = os.path.abspath(os.path.join(root, file_name))
                    rel_path = self._safe_rel_path(path)
                    discovered.append(
                        {
                            "name": self._display_name(path, file_name),
                            "path": path,
                            "rel_path": rel_path,
                            "type": self.model_type_from_path(path),
                        }
                    )

        valid_uploaded = []
        for item in self.config.get("uploaded_models", []):
            p = os.path.abspath(item.get("path", ""))
            if os.path.exists(p):
                valid_uploaded.append(item)
        self.config["uploaded_models"] = valid_uploaded
        self.save_config()

        index = {}
        for model in discovered:
            index[os.path.abspath(model["path"])] = model

        for uploaded in valid_uploaded:
            p = os.path.abspath(uploaded["path"])
            if p not in index:
                file_name = os.path.basename(p)
                index[p] = {
                    "name": self._display_name(p, file_name),
                    "path": p,
                    "rel_path": self._safe_rel_path(p),
                    "type": self.model_type_from_path(p),
                }

        return sorted(index.values(), key=lambda x: x["name"].lower())

    def remember_last_selected(self, model_path):
        self.config["last_selected_model"] = os.path.abspath(model_path)
        self.save_config()

    def get_last_selected(self):
        p = os.path.abspath(self.config.get("last_selected_model", ""))
        return p if p and os.path.exists(p) else ""

    def add_uploaded_model(self, model_path):
        model_path = os.path.abspath(model_path)
        already = any(os.path.abspath(x.get("path", "")) == model_path for x in self.config["uploaded_models"])
        if not already:
            self.config["uploaded_models"].append({"path": model_path})
            self.save_config()


class ClassConfigDialog:
    """Dialog for configuring class names and colors."""
    
    DEFAULT_COLORS = {
        0: (255, 0, 0),      # Red
        1: (0, 255, 0),      # Green
        2: (0, 0, 255),      # Blue
        3: (255, 255, 0),    # Cyan
        4: (255, 0, 255),    # Magenta
        5: (0, 255, 255),    # Yellow
        6: (128, 0, 0),      # Maroon
        7: (0, 128, 0),      # Olive
        8: (0, 0, 128),      # Navy
        9: (128, 128, 0),    # Teal
    }
    
    def __init__(self, parent, model_path, existing_config=None):
        self.window = tk.Toplevel(parent)
        self.window.title(f"Configure Classes - {os.path.basename(model_path)}")
        self.window.geometry("600x500")
        self.window.resizable(True, True)
        self.window.transient(parent)
        self.window.grab_set()
        center_window(self.window, parent)
        
        self.model_path = model_path
        self.class_names = existing_config.get("class_names", {}) if existing_config else {}
        self.class_colors = existing_config.get("class_colors", {}) if existing_config else {}
        
        # Get number of classes from model
        self.num_classes = self._detect_num_classes()
        
        tk.Label(self.window, text="Configure Class Names and Colors", font=("Arial", 12, "bold")).pack(pady=10)
        
        # Create scrollable frame
        canvas_frame = tk.Frame(self.window)
        canvas_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=5)
        
        self.canvas = tk.Canvas(canvas_frame)
        scrollbar = ttk.Scrollbar(canvas_frame, orient="vertical", command=self.canvas.yview)
        self.scrollable_frame = tk.Frame(self.canvas)
        
        self.scrollable_frame.bind(
            "<Configure>",
            lambda e: self.canvas.configure(scrollregion=self.canvas.bbox("all"))
        )
        
        self.canvas_window = self.canvas.create_window((0, 0), window=self.scrollable_frame, anchor="nw")
        self.canvas.configure(yscrollcommand=scrollbar.set)
        
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        self.canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        
        self.canvas.bind("<Configure>", self._on_canvas_configure)
        
        # Create class entries
        self.entries = {}
        self.color_buttons = {}
        
        for i in range(self.num_classes):
            frame = tk.Frame(self.scrollable_frame)
            frame.pack(fill=tk.X, pady=5, padx=5)
            
            tk.Label(frame, text=f"Class {i}:", width=10, anchor="w").pack(side=tk.LEFT, padx=5)
            
            name_entry = tk.Entry(frame, width=20)
            name_entry.insert(0, self.class_names.get(i, f"cls {i}"))
            name_entry.pack(side=tk.LEFT, padx=5)
            self.entries[i] = name_entry
            
            # Color button
            color = self.class_colors.get(i, self.DEFAULT_COLORS.get(i % len(self.DEFAULT_COLORS), (0, 200, 0)))
            btn = tk.Button(frame, text=f"Color", width=8, bg=self._rgb_to_tk(color), 
                          command=lambda idx=i, c=color: self._pick_color(idx, c))
            btn.pack(side=tk.LEFT, padx=5)
            self.color_buttons[i] = btn
        
        # Buttons
        btn_frame = tk.Frame(self.window)
        btn_frame.pack(pady=10)
        
        tk.Button(btn_frame, text="Save", command=self._on_save, width=10, bg="#90ee90").pack(side=tk.LEFT, padx=10)
        tk.Button(btn_frame, text="Cancel", command=self.window.destroy, width=10).pack(side=tk.LEFT, padx=10)
    
    def _on_canvas_configure(self, event):
        self.canvas.itemconfig(self.canvas_window, width=event.width)
    
    def _detect_num_classes(self):
        """Detect number of classes from model."""
        try:
            if self.model_path.endswith(".pt"):
                model = YOLO(self.model_path)
                return len(model.names)
            else:
                checkpoint = torch.load(self.model_path, map_location="cpu", weights_only=False)
                state_dict = checkpoint.get("model_state_dict", checkpoint) if isinstance(checkpoint, dict) else checkpoint
                if "roi_heads.box_predictor.cls_score.weight" in state_dict:
                    return state_dict["roi_heads.box_predictor.cls_score.weight"].shape[0]
        except Exception:
            pass
        return 10  # Default
    
    def _rgb_to_tk(self, color):
        """Convert RGB (0-255) to Tkinter color string."""
        return f"#{color[2]:02x}{color[1]:02x}{color[0]:02x}"
    
    def _tk_to_rgb(self, tk_color):
        """Convert Tkinter color string to RGB (0-255)."""
        tk_color = tk_color.lstrip('#')
        return (int(tk_color[4:6], 16), int(tk_color[2:4], 16), int(tk_color[0:2], 16))
    
    def _pick_color(self, class_idx, current_color):
        """Open color picker dialog."""
        from tkinter import colorchooser
        color = colorchooser.askcolor(color=self._rgb_to_tk(current_color), title=f"Choose color for Class {class_idx}")
        if color[1]:
            rgb = self._tk_to_rgb(color[1])
            self.class_colors[class_idx] = rgb
            self.color_buttons[class_idx].config(bg=color[1])
    
    def _on_save(self):
        """Save configuration."""
        self.class_names = {i: entry.get() or f"cls {i}" for i, entry in self.entries.items()}
        self.window.destroy()
    
    def get_config(self):
        """Return the configuration."""
        return {
            "class_names": self.class_names,
            "class_colors": self.class_colors
        }


class BasePage(tk.Frame):
    def __init__(self, parent, app):
        super().__init__(parent)
        self.app = app

    def on_show(self):
        pass

    def on_leave(self):
        pass


def _enumerate_cameras(max_test: int = 10) -> list:
    """
    Probe indices 0..max_test-1 and return (index, backend, label) tuples for
    every camera OpenCV can actually stream from.

    Virtual cameras (Camo Studio, OBS Virtual Camera, DroidCam, etc.) may only
    respond to one specific backend.  We try each index against several backends
    and store whichever backend delivered a real frame.  start_camera() then
    reopens the device with that exact backend — using a different one is the
    primary cause of blank frames from virtual cameras.

    OpenCV prints noisy WARN/ERROR lines to stderr while probing missing indices;
    we redirect stderr to devnull for the duration of the scan.
    """
    if sys.platform == "win32":
        backends = [cv2.CAP_MSMF, cv2.CAP_DSHOW, cv2.CAP_ANY]
    elif sys.platform == "darwin":
        backends = [cv2.CAP_AVFOUNDATION, cv2.CAP_ANY]
    else:
        backends = [cv2.CAP_V4L2, cv2.CAP_ANY]

    # Suppress the WARN/ERROR spam OpenCV emits for every missing index
    devnull_fd = os.open(os.devnull, os.O_WRONLY)
    old_stderr_fd = os.dup(2)
    os.dup2(devnull_fd, 2)   # redirect stderr → /dev/null
    os.close(devnull_fd)

    found = []
    try:
        for idx in range(max_test):
            winning_backend = None
            for backend in backends:
                try:
                    cap = cv2.VideoCapture(idx, backend)
                except Exception:
                    continue
                if not cap.isOpened():
                    cap.release()
                    continue
                # Warmup: drain up to 5 frames; virtual cameras need time to start
                ok = False
                for _ in range(5):
                    ok, _ = cap.read()
                    if ok:
                        break
                cap.release()
                if ok:
                    winning_backend = backend
                    break

            if winning_backend is not None:
                label = f"Camera {idx}" + (" (Default)" if idx == 0 else "")
                found.append((idx, winning_backend, label))
    finally:
        os.dup2(old_stderr_fd, 2)   # restore stderr
        os.close(old_stderr_fd)

    return found


class LiveInferencePage(BasePage):
    def __init__(self, parent, app):
        super().__init__(parent, app)
        self.cap = None
        self.running = False
        self._available_cameras = []   # list of (index, backend, label) tuples

        controls = tk.Frame(self)
        controls.pack(side=tk.TOP, fill=tk.X, padx=10, pady=10)

        self.btn_start = tk.Button(controls, text="Start Camera",
                                   command=self.start_camera, width=16, bg="#90ee90")
        self.btn_start.pack(side=tk.LEFT, padx=5)

        self.btn_stop = tk.Button(controls, text="Stop Camera",
                                  command=self.stop_camera,
                                  state=tk.DISABLED, width=16, bg="#dddddd")
        self.btn_stop.pack(side=tk.LEFT, padx=5)

        # ── Camera selector ───────────────────────────────────────────────────
        tk.Label(controls, text="Camera:").pack(side=tk.LEFT, padx=(20, 4))

        self.camera_var = tk.StringVar(value="Scanning...")
        self.camera_combo = ttk.Combobox(controls, textvariable=self.camera_var,
                                         state="readonly", width=28)
        self.camera_combo.pack(side=tk.LEFT, padx=(0, 4))

        self.btn_refresh_cam = tk.Button(controls, text="🔄",
                                         command=self._scan_cameras_async,
                                         width=3, relief=tk.FLAT, font=("Arial", 11))
        self.btn_refresh_cam.pack(side=tk.LEFT, padx=(0, 10))

        # ── Confidence slider ─────────────────────────────────────────────────
        tk.Label(controls, text="Confidence:").pack(side=tk.LEFT, padx=(10, 5))
        self.conf_slider = tk.Scale(controls, from_=0.0, to=1.0,
                                    resolution=0.05, orient=tk.HORIZONTAL, length=170)
        self.conf_slider.set(0.5)
        self.conf_slider.pack(side=tk.LEFT)

        # ── Display area ──────────────────────────────────────────────────────
        display = tk.Frame(self, bg="black")
        display.pack(expand=True, fill=tk.BOTH, padx=10, pady=10)
        display.grid_rowconfigure(0, weight=1)
        display.grid_columnconfigure(0, weight=1)

        self.image_label = tk.Label(display, bg="black")
        self.image_label.grid(row=0, column=0, sticky="nsew", padx=1, pady=1)

        display.bind("<Configure>", self.on_display_resize)
        self.display_frame = display

        # Scan cameras in background so UI appears instantly
        self.after(200, self._scan_cameras_async)

    # ── Camera discovery ──────────────────────────────────────────────────────

    def _scan_cameras_async(self):
        """Scan for cameras in a background thread so the UI stays responsive."""
        self.camera_combo.config(state=tk.DISABLED)
        self.btn_refresh_cam.config(state=tk.DISABLED)
        self.camera_var.set("Scanning...")

        def _scan():
            cameras = _enumerate_cameras()
            self.after(0, lambda: self._on_cameras_found(cameras))

        threading.Thread(target=_scan, daemon=True).start()

    def _on_cameras_found(self, cameras):
        self._available_cameras = cameras
        self.btn_refresh_cam.config(state=tk.NORMAL)

        if not cameras:
            self.camera_var.set("No cameras found")
            self.camera_combo.config(values=["No cameras found"], state="readonly")
            return

        labels = [label for _, _backend, label in cameras]
        self.camera_combo.config(values=labels, state="readonly")

        # Keep current selection when refreshing; otherwise default to first
        current = self.camera_var.get()
        if current not in labels:
            self.camera_var.set(labels[0])

    def _selected_camera(self):
        """Return (index, backend) for the selected camera, or (0, CAP_ANY) as fallback."""
        selected_label = self.camera_var.get()
        for idx, backend, label in self._available_cameras:
            if label == selected_label:
                return idx, backend
        return 0, cv2.CAP_ANY

    # ── Camera control ────────────────────────────────────────────────────────

    def start_camera(self):
        if self.running:
            return
        if self.app.active_runner is None:
            messagebox.showwarning("Model Required",
                                   "Please select and load a model from the dropdown.")
            return
        if not self._available_cameras:
            messagebox.showwarning("No Camera",
                                   "No cameras detected. Click the refresh button to scan again.")
            return

        cam_idx, backend = self._selected_camera()

        # Open with the exact backend that worked during enumeration.
        # Mismatched backends are the #1 cause of blank frames from virtual
        # cameras like Camo Studio.
        self.cap = cv2.VideoCapture(cam_idx, backend)
        if not self.cap.isOpened():
            messagebox.showerror(
                "Camera Error",
                f"Could not open camera: {self.camera_var.get()}\n\n"
                "If you are using Camo Studio, make sure:\n"
                "  \u2022 Camo is running and your phone is connected\n"
                "  \u2022 The Camo virtual camera driver is installed\n"
                "  \u2022 No other app is currently using the camera"
            )
            return

        # Warmup: drain up to 10 frames silently.  Virtual camera drivers need
        # a moment before they deliver real image data.
        for _ in range(10):
            ok, _ = self.cap.read()
            if ok:
                break

        self.running = True
        self.camera_combo.config(state=tk.DISABLED)
        self.btn_refresh_cam.config(state=tk.DISABLED)
        self.btn_start.config(state=tk.DISABLED, bg="#dddddd")
        self.btn_stop.config(state=tk.NORMAL, bg="#ffcccb")
        self._loop()

    def stop_camera(self):
        self.running = False
        if self.cap is not None:
            self.cap.release()
            self.cap = None
        self.app.clear_label_image(self.image_label)
        self.btn_start.config(state=tk.NORMAL, bg="#90ee90")
        self.btn_stop.config(state=tk.DISABLED, bg="#dddddd")
        self.camera_combo.config(state="readonly")
        self.btn_refresh_cam.config(state=tk.NORMAL)

    def _loop(self):
        if not self.running or self.cap is None:
            return
        ok, frame = self.cap.read()
        if ok:
            conf = float(self.conf_slider.get())
            rendered = self.app.run_inference(frame, conf)
            self.app.show_frame(self.image_label, rendered)
        self.after(10, self._loop)

    def on_display_resize(self, event):
        current = getattr(self.image_label, "current_frame", None)
        if current is not None:
            self.app.show_frame(self.image_label, current)

    def on_leave(self):
        self.stop_camera()


class ImageInferencePage(BasePage):
    def __init__(self, parent, app):
        super().__init__(parent, app)
        self.image_path = ""
        self.original_frame = None
        self.rendered_frame = None

        controls = tk.Frame(self)
        controls.pack(side=tk.TOP, fill=tk.X, padx=10, pady=10)

        self.btn_upload = tk.Button(controls, text="Upload Image", command=self.upload_image, width=16)
        self.btn_upload.pack(side=tk.LEFT, padx=5)

        self.btn_infer = tk.Button(controls, text="Start Inference", command=self.start_inference, width=16, bg="#90ee90")
        self.btn_infer.pack(side=tk.LEFT, padx=5)

        tk.Label(controls, text="Confidence:").pack(side=tk.LEFT, padx=(20, 5))
        self.conf_slider = tk.Scale(controls, from_=0.0, to=1.0, resolution=0.05, orient=tk.HORIZONTAL, length=170)
        self.conf_slider.set(0.5)
        self.conf_slider.pack(side=tk.LEFT)

        self.btn_open_photos = tk.Button(controls, text="🔍 Open in Photos",
                                         command=self.open_in_photos, width=16, state=tk.DISABLED)
        self.btn_open_photos.pack(side=tk.LEFT, padx=(20, 4))

        self.btn_save = tk.Button(controls, text="💾 Save Image",
                                  command=self.save_image, width=14, state=tk.DISABLED)
        self.btn_save.pack(side=tk.LEFT, padx=4)

        self.lbl_path = tk.Label(self, text="No image uploaded", anchor="w", fg="#444444")
        self.lbl_path.pack(fill=tk.X, padx=12)

        # Display area with dynamic sizing
        display = tk.Frame(self, bg="black")
        display.pack(expand=True, fill=tk.BOTH, padx=10, pady=10)
        
        # Use grid with weight to make it responsive
        display.grid_rowconfigure(0, weight=1)
        display.grid_columnconfigure(0, weight=1)

        self.image_label = tk.Label(display, bg="black")
        self.image_label.grid(row=0, column=0, sticky="nsew", padx=1, pady=1)
        
        # Bind resize event
        display.bind("<Configure>", self.on_display_resize)
        self.display_frame = display

    def on_display_resize(self, event):
        """Redraw current frame to fit resized display area."""
        current = getattr(self.image_label, "current_frame", None)
        if current is not None:
            self.app.show_frame(self.image_label, current)

    def upload_image(self):
        path = filedialog.askopenfilename(filetypes=[("Images", "*.jpg *.jpeg *.png *.bmp *.webp")])
        if not path:
            return
        frame = cv2.imread(path)
        if frame is None:
            messagebox.showerror("Image Error", "Failed to read selected image file.")
            return
        self.image_path = path
        self.original_frame = frame
        self.rendered_frame = None
        self.lbl_path.config(text=f"Image: {os.path.basename(path)}")
        self.app.show_frame(self.image_label, frame)
        self.btn_open_photos.config(state=tk.NORMAL)
        self.btn_save.config(state=tk.DISABLED)

    def start_inference(self):
        if self.app.active_runner is None:
            messagebox.showwarning("Model Required", "Please select and load a model from the dropdown.")
            return
        if self.original_frame is None:
            messagebox.showwarning("Image Required", "Please upload an image first.")
            return
        conf = float(self.conf_slider.get())
        self.rendered_frame = self.app.run_inference(self.original_frame, conf)
        self.app.show_frame(self.image_label, self.rendered_frame)
        self.btn_open_photos.config(state=tk.NORMAL)
        self.btn_save.config(state=tk.NORMAL)

    def open_in_photos(self):
        import tempfile, subprocess
        frame = self.rendered_frame if self.rendered_frame is not None else self.original_frame
        if frame is None:
            return
        tmp = tempfile.NamedTemporaryFile(suffix=".png", delete=False)
        tmp_path = tmp.name
        tmp.close()
        cv2.imwrite(tmp_path, frame)
        try:
            if sys.platform == "win32":
                os.startfile(tmp_path)
            elif sys.platform == "darwin":
                subprocess.Popen(["open", tmp_path])
            else:
                subprocess.Popen(["xdg-open", tmp_path])
        except Exception as ex:
            messagebox.showerror("Open Error", f"Could not open image:\n{ex}")

    def save_image(self):
        frame = self.rendered_frame if self.rendered_frame is not None else self.original_frame
        if frame is None:
            return
        stem = os.path.splitext(os.path.basename(self.image_path))[0]
        default_name = f"{stem}_inferred.png"
        save_path = filedialog.asksaveasfilename(
            initialfile=default_name,
            defaultextension=".png",
            filetypes=[("PNG", "*.png"), ("JPEG", "*.jpg"), ("All Files", "*.*")]
        )
        if save_path:
            cv2.imwrite(save_path, frame)
            messagebox.showinfo("Saved", f"Image saved to:\n{save_path}")

    def on_leave(self):
        self.app.clear_label_image(self.image_label)


class FolderInferencePage(BasePage):
    IMAGES_PER_PAGE = 30   # 10 rows × 3 columns
    IMAGES_PER_ROW  = 3

    def __init__(self, parent, app):
        super().__init__(parent, app)
        self.folder_path = ""
        self.image_paths = []
        self.inferenced_images = []   # parallel list of rendered BGR frames (or None)
        self.current_page = 0         # 0-based page index
        self._resize_job = None
        self._last_canvas_width = 0
        self.row_height = 220

        # ── top controls ──────────────────────────────────────────────────────
        controls = tk.Frame(self)
        controls.pack(side=tk.TOP, fill=tk.X, padx=10, pady=10)

        self.btn_upload_folder = tk.Button(controls, text="Upload Folder",
                                           command=self.upload_folder, width=16)
        self.btn_upload_folder.pack(side=tk.LEFT, padx=5)

        self.btn_infer_all = tk.Button(controls, text="Start Inference",
                                       command=self.start_inference_all,
                                       width=16, bg="#90ee90")
        self.btn_infer_all.pack(side=tk.LEFT, padx=5)

        tk.Label(controls, text="Confidence:").pack(side=tk.LEFT, padx=(20, 5))
        self.conf_slider = tk.Scale(controls, from_=0.0, to=1.0,
                                    resolution=0.05, orient=tk.HORIZONTAL, length=170)
        self.conf_slider.set(0.5)
        self.conf_slider.pack(side=tk.LEFT)

        # Save all inferenced images button
        self.btn_save_all = tk.Button(controls, text="💾 Save All",
                                      command=self.save_all_images,
                                      width=12, state=tk.DISABLED)
        self.btn_save_all.pack(side=tk.LEFT, padx=(20, 5))

        self.lbl_path = tk.Label(self, text="No folder selected", anchor="w", fg="#444444")
        self.lbl_path.pack(fill=tk.X, padx=12)

        # ── scrollable canvas for image grid ─────────────────────────────────
        self.canvas_frame = tk.Frame(self)
        self.canvas_frame.pack(expand=True, fill=tk.BOTH, padx=10, pady=(5, 0))

        self.canvas = tk.Canvas(self.canvas_frame, bg="black", highlightthickness=0)
        self.scrollbar = ttk.Scrollbar(self.canvas_frame, orient="vertical",
                                        command=self.canvas.yview)

        self.grid_frame = tk.Frame(self.canvas, bg="black")
        self.canvas_window = self.canvas.create_window((0, 0), window=self.grid_frame, anchor="nw")

        self.canvas.configure(yscrollcommand=self.scrollbar.set)
        self.scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        self.canvas.pack(side=tk.LEFT, expand=True, fill=tk.BOTH)

        self.canvas.bind("<Configure>", self._on_canvas_configure)
        self.grid_frame.bind("<Configure>", self._on_frame_configure)
        self.canvas.bind_all("<MouseWheel>", self._on_mousewheel)

        self.displayed_widgets = []

        # ── bottom pagination bar ─────────────────────────────────────────────
        page_bar = tk.Frame(self)
        page_bar.pack(side=tk.BOTTOM, fill=tk.X, pady=6)

        # Centre the controls inside page_bar
        nav_inner = tk.Frame(page_bar)
        nav_inner.pack(anchor="center")

        self.btn_prev_page = tk.Button(nav_inner, text="◄ Previous",
                                       command=self.prev_page,
                                       width=12, state=tk.DISABLED)
        self.btn_prev_page.pack(side=tk.LEFT, padx=8)

        tk.Label(nav_inner, text="Page:").pack(side=tk.LEFT)
        self.page_entry_var = tk.StringVar(value="1")
        self.page_entry = tk.Entry(nav_inner, textvariable=self.page_entry_var,
                                   width=5, justify="center")
        self.page_entry.pack(side=tk.LEFT, padx=4)
        self.page_entry.bind("<Return>", self._on_page_entry)
        self.page_entry.bind("<FocusOut>", self._on_page_entry)

        self.lbl_total_pages = tk.Label(nav_inner, text="/ 0")
        self.lbl_total_pages.pack(side=tk.LEFT, padx=(0, 8))

        self.btn_next_page = tk.Button(nav_inner, text="Next ►",
                                       command=self.next_page,
                                       width=12, state=tk.DISABLED)
        self.btn_next_page.pack(side=tk.LEFT, padx=8)

    # ── canvas event helpers ──────────────────────────────────────────────────

    def _on_canvas_configure(self, event):
        self.canvas.configure(scrollregion=self.canvas.bbox("all"))
        self.canvas.itemconfig(self.canvas_window, width=event.width)
        if self.inferenced_images and abs(event.width - self._last_canvas_width) > 24:
            self._last_canvas_width = event.width
            if self._resize_job is not None:
                self.after_cancel(self._resize_job)
            self._resize_job = self.after(140, self._display_page)

    def _on_frame_configure(self, event):
        self.canvas.configure(scrollregion=self.canvas.bbox("all"))

    def _on_mousewheel(self, event):
        self.canvas.yview_scroll(int(-1 * (event.delta / 120)), "units")

    # ── folder loading ────────────────────────────────────────────────────────

    def upload_folder(self):
        folder = filedialog.askdirectory(title="Select Folder with Images")
        if not folder:
            return
        valid_ext = {'.jpg', '.jpeg', '.png', '.bmp', '.webp'}
        paths = [os.path.join(folder, f) for f in os.listdir(folder)
                 if Path(f).suffix.lower() in valid_ext]
        if not paths:
            messagebox.showwarning("No Images",
                                   "No image files found in the selected folder.")
            return
        self.folder_path = folder
        self.image_paths = sorted(paths)
        self.lbl_path.config(
            text=f"Folder: {os.path.basename(folder)} ({len(self.image_paths)} images)")
        self._reset_state()

    def _reset_state(self):
        self.inferenced_images = []
        self.current_page = 0
        self._last_canvas_width = 0
        self._clear_grid()
        self._update_page_controls()
        self.btn_save_all.config(state=tk.DISABLED)

    def _clear_grid(self):
        for w in self.displayed_widgets:
            w.destroy()
        self.displayed_widgets = []

    # ── inference ─────────────────────────────────────────────────────────────

    def start_inference_all(self):
        if self.app.active_runner is None:
            messagebox.showwarning("Model Required",
                                   "Please select and load a model from the dropdown.")
            return
        if not self.image_paths:
            messagebox.showwarning("Folder Required",
                                   "Please upload a folder with images first.")
            return

        conf = float(self.conf_slider.get())
        progress = ProgressDialog(self, "Processing Images",
                                  f"Running inference on {len(self.image_paths)} images...")
        progress.set_indeterminate("Loading images and running inference...")

        def process_images():
            results = []
            for idx, img_path in enumerate(self.image_paths):
                frame = cv2.imread(img_path)
                if frame is None:
                    results.append(None)
                else:
                    results.append(self.app.run_inference(frame, conf))
                pct = ((idx + 1) / len(self.image_paths)) * 100
                progress.set_determinate(pct,
                                         f"Processed {idx+1}/{len(self.image_paths)} images")
            self.inferenced_images = results
            self.current_page = 0
            self.after(0, self._on_inference_done)
            progress.finish("Inference complete!", success=True)

        threading.Thread(target=process_images, daemon=True).start()

    def _on_inference_done(self):
        self._update_page_controls()
        self._display_page()
        self.btn_save_all.config(state=tk.NORMAL)

    # ── pagination helpers ────────────────────────────────────────────────────

    @property
    def _total_pages(self):
        if not self.inferenced_images:
            return 0
        return max(1, -(-len(self.inferenced_images) // self.IMAGES_PER_PAGE))  # ceiling div

    def prev_page(self):
        if self.current_page > 0:
            self.current_page -= 1
            self._display_page()

    def next_page(self):
        if self.current_page < self._total_pages - 1:
            self.current_page += 1
            self._display_page()

    def _on_page_entry(self, _event=None):
        try:
            page = int(self.page_entry_var.get()) - 1  # convert to 0-based
            page = max(0, min(page, self._total_pages - 1))
            self.current_page = page
            self._display_page()
        except ValueError:
            pass
        self.page_entry_var.set(str(self.current_page + 1))

    def _update_page_controls(self):
        total = self._total_pages
        self.lbl_total_pages.config(text=f"/ {total}")
        self.page_entry_var.set(str(self.current_page + 1))
        self.btn_prev_page.config(
            state=tk.NORMAL if self.current_page > 0 else tk.DISABLED)
        self.btn_next_page.config(
            state=tk.NORMAL if self.current_page < total - 1 else tk.DISABLED)

    # ── grid rendering ────────────────────────────────────────────────────────

    @staticmethod
    def _resize_cover(img_pil, target_w, target_h):
        src_w, src_h = img_pil.size
        if src_w <= 0 or src_h <= 0:
            return img_pil.resize((target_w, target_h), Image.Resampling.BICUBIC)
        scale = max(target_w / src_w, target_h / src_h)
        rw = max(1, int(round(src_w * scale)))
        rh = max(1, int(round(src_h * scale)))
        resized = img_pil.resize((rw, rh), Image.Resampling.LANCZOS)
        left  = max(0, (rw - target_w) // 2)
        top   = max(0, (rh - target_h) // 2)
        return resized.crop((left, top, left + target_w, top + target_h))

    def _display_page(self):
        self._clear_grid()
        if not self.inferenced_images:
            return

        # Tile dimensions
        canvas_width = self.canvas.winfo_width()
        if canvas_width <= 1:
            canvas_width = 1200
        self._last_canvas_width = canvas_width

        gap = 10
        available = max(360, canvas_width - 16)
        img_w = max(220, int((available - gap * (self.IMAGES_PER_ROW + 1)) / self.IMAGES_PER_ROW))
        img_h = max(170, int(img_w * 0.62))
        self.row_height = img_h + 36

        # Slice for this page
        start = self.current_page * self.IMAGES_PER_PAGE
        end   = min(start + self.IMAGES_PER_PAGE, len(self.inferenced_images))
        page_frames  = self.inferenced_images[start:end]
        page_paths   = self.image_paths[start:end]

        for local_idx, (frame, img_path) in enumerate(zip(page_frames, page_paths)):
            global_idx = start + local_idx
            row = local_idx // self.IMAGES_PER_ROW
            col = local_idx  % self.IMAGES_PER_ROW

            # Configure grid weights
            self.grid_frame.grid_columnconfigure(col, weight=1)
            self.grid_frame.grid_rowconfigure(row, weight=0)

            # Outer tile frame
            tile = tk.Frame(self.grid_frame, bg="#1a1a1a", bd=2, relief=tk.GROOVE)
            tile.grid(row=row, column=col, padx=4, pady=4, sticky="nsew")
            self.displayed_widgets.append(tile)

            # ── image area (relative container for the overlay button) ──────
            img_container = tk.Frame(tile, bg="black", width=img_w, height=img_h)
            img_container.pack(fill=tk.BOTH, expand=True)
            img_container.pack_propagate(False)

            if frame is not None:
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                img_pil   = Image.fromarray(frame_rgb)
                img_pil   = self._resize_cover(img_pil, img_w, img_h)
                img_tk    = ImageTk.PhotoImage(img_pil)

                lbl = tk.Label(img_container, image=img_tk, bg="black", cursor="hand2")
                lbl.image = img_tk
                lbl.place(x=0, y=0, relwidth=1, relheight=1)
                lbl.bind("<Button-1>", lambda e, gi=global_idx: self.view_full_image(gi))

                # ── "Open in Photos" button — top-right corner ──────────────
                btn_open = tk.Button(
                    img_container,
                    text="🔍",
                    font=("Arial", 9),
                    padx=3, pady=1,
                    bg="#222222", fg="white",
                    activebackground="#444444",
                    relief=tk.FLAT, cursor="hand2",
                    command=lambda p=img_path, f=frame: self._open_in_photos(p, f)
                )
                btn_open.place(relx=1.0, rely=0.0, anchor="ne", x=-4, y=4)

                # ── save button — top-left corner ────────────────────────────
                btn_save = tk.Button(
                    img_container,
                    text="💾",
                    font=("Arial", 9),
                    padx=3, pady=1,
                    bg="#222222", fg="white",
                    activebackground="#444444",
                    relief=tk.FLAT, cursor="hand2",
                    command=lambda f=frame, p=img_path: self._save_single(f, p)
                )
                btn_save.place(relx=0.0, rely=0.0, anchor="nw", x=4, y=4)
            else:
                tk.Label(img_container, text="Failed to load",
                         fg="red", bg="black").place(relx=0.5, rely=0.5, anchor="center")

            # Filename label
            filename = os.path.basename(img_path)
            if len(filename) > 44:
                filename = filename[:41] + "..."
            lbl_name = tk.Label(tile, text=filename,
                                fg="white", bg="#1a1a1a", font=("Arial", 8))
            lbl_name.pack(fill=tk.X, pady=(2, 3))
            self.displayed_widgets.append(lbl_name)

        # Scroll back to top when page changes
        self.canvas.yview_moveto(0)
        self.canvas.configure(scrollregion=self.canvas.bbox("all"))
        self._update_page_controls()

    # ── open / save helpers ───────────────────────────────────────────────────

    def _open_in_photos(self, original_path, inferred_frame):
        """
        Save a temporary PNG of the *inferred* result then open it with
        the Windows Photos app (or the system default viewer on other OSes).
        """
        import tempfile, subprocess
        tmp = tempfile.NamedTemporaryFile(suffix=".png", delete=False)
        tmp_path = tmp.name
        tmp.close()
        cv2.imwrite(tmp_path, inferred_frame)
        try:
            if sys.platform == "win32":
                os.startfile(tmp_path)
            elif sys.platform == "darwin":
                subprocess.Popen(["open", tmp_path])
            else:
                subprocess.Popen(["xdg-open", tmp_path])
        except Exception as ex:
            messagebox.showerror("Open Error", f"Could not open image:\n{ex}")

    def _save_single(self, frame, original_path):
        default_name = os.path.splitext(os.path.basename(original_path))[0] + "_inferred.png"
        save_path = filedialog.asksaveasfilename(
            initialfile=default_name,
            defaultextension=".png",
            filetypes=[("PNG", "*.png"), ("JPEG", "*.jpg"), ("All Files", "*.*")]
        )
        if save_path:
            cv2.imwrite(save_path, frame)
            messagebox.showinfo("Saved", f"Image saved to:\n{save_path}")

    def save_all_images(self):
        if not self.inferenced_images:
            return
        folder = filedialog.askdirectory(title="Select folder to save inferenced images")
        if not folder:
            return
        saved = 0
        for frame, src_path in zip(self.inferenced_images, self.image_paths):
            if frame is None:
                continue
            stem = os.path.splitext(os.path.basename(src_path))[0]
            out  = os.path.join(folder, f"{stem}_inferred.png")
            cv2.imwrite(out, frame)
            saved += 1
        messagebox.showinfo("Saved", f"Saved {saved} inferenced images to:\n{folder}")

    def view_full_image(self, image_index):
        if image_index >= len(self.inferenced_images) or \
                self.inferenced_images[image_index] is None:
            return
        frame = self.inferenced_images[image_index]
        top = tk.Toplevel(self)
        top.title(f"Image {image_index+1}: {os.path.basename(self.image_paths[image_index])}")
        top.geometry("900x680")
        center_window(top, self.winfo_toplevel())

        # Toolbar
        toolbar = tk.Frame(top)
        toolbar.pack(side=tk.TOP, fill=tk.X, padx=10, pady=(8, 0))

        def open_in_photos():
            self._open_in_photos(self.image_paths[image_index], frame)

        def save_this():
            self._save_single(frame, self.image_paths[image_index])

        tk.Button(toolbar, text="🔍 Open in Photos", command=open_in_photos).pack(side=tk.LEFT, padx=4)
        tk.Button(toolbar, text="💾 Save Image",     command=save_this).pack(side=tk.LEFT, padx=4)

        orig_h, orig_w = frame.shape[:2]
        info_text = f"Size: {orig_w}×{orig_h} | Confidence: {float(self.conf_slider.get()):.2f}"
        tk.Label(toolbar, text=info_text, fg="#444444").pack(side=tk.RIGHT, padx=8)

        # Image display
        img_frame = tk.Frame(top, bg="black")
        img_frame.pack(expand=True, fill=tk.BOTH, padx=10, pady=10)

        max_w, max_h = 1400, 900
        scale = min(max_w / orig_w, max_h / orig_h)
        new_w, new_h = int(orig_w * scale), int(orig_h * scale)
        frame_rgb  = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        img_pil    = Image.fromarray(frame_rgb).resize((new_w, new_h), Image.Resampling.LANCZOS)
        img_tk     = ImageTk.PhotoImage(img_pil)

        lbl = tk.Label(img_frame, image=img_tk, bg="black")
        lbl.image = img_tk
        lbl.pack(expand=True)

    def on_leave(self):
        self._clear_grid()
        self.canvas.unbind_all("<MouseWheel>")


class VideoInferencePage(BasePage):
    """
    Two modes:
      INFERENCE  — runs the model on each video frame as it arrives.
      PLAYBACK   — once inference is complete, lets the user review the
                   result with play/pause, a seek scrubber, ±10 s skip
                   buttons, and a save-to-file option.
    """

    _SKIP_FRAMES = 10   # frames skipped by the << / >> buttons (overridden by FPS)

    def __init__(self, parent, app):
        super().__init__(parent, app)
        self.video_path   = ""
        self.cap          = None
        self.running      = False          # inference running
        self.max_infer_size   = (1280, 720)
        self.letterbox_color  = (114, 114, 114)

        # ── playback state ────────────────────────────────────────────────────
        self._rendered_frames  = []        # buffered inferred frames
        self._fps              = 25.0      # source video FPS (set at inference start)
        self._pb_playing       = False     # playback is active
        self._pb_index         = 0         # current playback frame index
        self._pb_after_id      = None      # pending after() call id
        self._scrubbing        = False     # user is dragging the scrubber

        # ── top controls row ──────────────────────────────────────────────────
        controls = tk.Frame(self)
        controls.pack(side=tk.TOP, fill=tk.X, padx=10, pady=(10, 4))

        self.btn_upload = tk.Button(controls, text="Upload Video",
                                    command=self.upload_video, width=14)
        self.btn_upload.pack(side=tk.LEFT, padx=5)

        self.btn_start = tk.Button(controls, text="Start Inference",
                                   command=self.start_video, width=14, bg="#90ee90")
        self.btn_start.pack(side=tk.LEFT, padx=5)

        self.btn_stop = tk.Button(controls, text="Stop",
                                  command=self.stop_video,
                                  width=8, bg="#dddddd", state=tk.DISABLED)
        self.btn_stop.pack(side=tk.LEFT, padx=5)

        tk.Label(controls, text="Confidence:").pack(side=tk.LEFT, padx=(16, 4))
        self.conf_slider = tk.Scale(controls, from_=0.0, to=1.0,
                                    resolution=0.05, orient=tk.HORIZONTAL, length=150)
        self.conf_slider.set(0.5)
        self.conf_slider.pack(side=tk.LEFT)

        self.btn_save_video = tk.Button(controls, text="💾 Save Video",
                                        command=self.save_video,
                                        width=13, state=tk.DISABLED)
        self.btn_save_video.pack(side=tk.LEFT, padx=(16, 4))

        self.lbl_path = tk.Label(self, text="No video uploaded", anchor="w", fg="#444444")
        self.lbl_path.pack(fill=tk.X, padx=12)

        # ── inference progress bar (hidden during playback) ───────────────────
        self._infer_bar_frame = tk.Frame(self)
        self._infer_bar_frame.pack(fill=tk.X, padx=12, pady=(0, 2))
        self._infer_bar = ttk.Progressbar(self._infer_bar_frame,
                                           orient=tk.HORIZONTAL, mode="determinate")
        self._infer_bar.pack(fill=tk.X)
        self._lbl_infer_progress = tk.Label(self._infer_bar_frame, text="", anchor="w",
                                             fg="#555555", font=("Arial", 8))
        self._lbl_infer_progress.pack(anchor="w")
        self._infer_bar_frame.pack_forget()   # hidden until inference starts

        # ── video display ─────────────────────────────────────────────────────
        display = tk.Frame(self, bg="black")
        display.pack(expand=True, fill=tk.BOTH, padx=10, pady=(4, 0))
        display.grid_rowconfigure(0, weight=1)
        display.grid_columnconfigure(0, weight=1)

        self.image_label = tk.Label(display, bg="black")
        self.image_label.grid(row=0, column=0, sticky="nsew", padx=1, pady=1)
        display.bind("<Configure>", self.on_display_resize)
        self.display_frame = display

        # ── playback controls (shown only after inference) ────────────────────
        self._pb_frame = tk.Frame(self, pady=4)

        # Scrubber row
        scrub_row = tk.Frame(self._pb_frame)
        scrub_row.pack(fill=tk.X, padx=10)

        self._lbl_cur  = tk.Label(scrub_row, text="0:00", width=5, anchor="e")
        self._lbl_cur.pack(side=tk.LEFT)

        self._scrub_var = tk.IntVar(value=0)
        self._scrubber  = ttk.Scale(scrub_row, from_=0, to=1,
                                    orient=tk.HORIZONTAL, variable=self._scrub_var)
        self._scrubber.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=6)
        self._scrubber.bind("<ButtonPress-1>",   self._on_scrub_start)
        self._scrubber.bind("<ButtonRelease-1>", self._on_scrub_end)
        self._scrubber.bind("<B1-Motion>",       self._on_scrub_drag)

        self._lbl_dur = tk.Label(scrub_row, text="0:00", width=5, anchor="w")
        self._lbl_dur.pack(side=tk.LEFT)

        # Button row
        btn_row = tk.Frame(self._pb_frame)
        btn_row.pack(pady=(2, 4))

        self._btn_rewind = tk.Button(btn_row, text="⏪  -10s",
                                     command=self._skip_backward,
                                     width=9, relief=tk.FLAT, font=("Arial", 10))
        self._btn_rewind.pack(side=tk.LEFT, padx=6)

        self._btn_playpause = tk.Button(btn_row, text="▶  Play",
                                        command=self._toggle_play,
                                        width=10, bg="#90ee90",
                                        font=("Arial", 10, "bold"))
        self._btn_playpause.pack(side=tk.LEFT, padx=6)

        self._btn_forward = tk.Button(btn_row, text="+10s  ⏩",
                                      command=self._skip_forward,
                                      width=9, relief=tk.FLAT, font=("Arial", 10))
        self._btn_forward.pack(side=tk.LEFT, padx=6)

        # Frame counter label
        self._lbl_frame_count = tk.Label(btn_row, text="", fg="#888888",
                                          font=("Arial", 8))
        self._lbl_frame_count.pack(side=tk.LEFT, padx=12)

        # Playback speed selector
        tk.Label(btn_row, text="Speed:", font=("Arial", 9)).pack(side=tk.LEFT, padx=(12, 2))
        self._speed_var = tk.StringVar(value="1x")
        speed_combo = ttk.Combobox(btn_row, textvariable=self._speed_var,
                                   values=["0.25x", "0.5x", "1x", "1.5x", "2x"],
                                   state="readonly", width=6)
        speed_combo.pack(side=tk.LEFT)
        speed_combo.bind("<<ComboboxSelected>>", lambda _e: None)  # value read in _pb_tick

        # playback panel hidden until inference done
        self._pb_frame.pack_forget()

    # ── helpers ───────────────────────────────────────────────────────────────

    @staticmethod
    def _fmt_time(seconds):
        s = max(0, int(seconds))
        return f"{s // 60}:{s % 60:02d}"

    def _letterbox_preprocess(self, frame_bgr):
        target_w, target_h = self.max_infer_size
        h, w = frame_bgr.shape[:2]
        if h <= 0 or w <= 0:
            return frame_bgr
        if w <= target_w and h <= target_h:
            return frame_bgr
        scale  = min(target_w / w, target_h / h)
        new_w  = max(1, int(round(w * scale)))
        new_h  = max(1, int(round(h * scale)))
        resized = cv2.resize(frame_bgr, (new_w, new_h), interpolation=cv2.INTER_AREA)
        letterboxed = np.full((target_h, target_w, 3), self.letterbox_color, dtype=np.uint8)
        pad_x = (target_w - new_w) // 2
        pad_y = (target_h - new_h) // 2
        letterboxed[pad_y:pad_y + new_h, pad_x:pad_x + new_w] = resized
        return letterboxed

    # ── inference mode ────────────────────────────────────────────────────────

    def upload_video(self):
        path = filedialog.askopenfilename(
            filetypes=[("Videos", "*.mp4 *.avi *.mov *.mkv *.wmv")])
        if not path:
            return
        self.video_path = path
        self.lbl_path.config(text=f"Video: {os.path.basename(path)}")
        # Reset playback if a new video is chosen
        self._stop_playback()
        self._rendered_frames = []
        self._pb_frame.pack_forget()
        self.btn_save_video.config(state=tk.DISABLED)
        self.app.clear_label_image(self.image_label)

    def start_video(self):
        if self.running:
            return
        if self.app.active_runner is None:
            messagebox.showwarning("Model Required",
                                   "Please select and load a model from the dropdown.")
            return
        if not self.video_path:
            messagebox.showwarning("Video Required", "Please upload a video first.")
            return

        self.cap = cv2.VideoCapture(self.video_path)
        if not self.cap.isOpened():
            messagebox.showerror("Video Error", "Could not open the selected video file.")
            return

        # Capture FPS now so playback uses the correct speed
        raw_fps = self.cap.get(cv2.CAP_PROP_FPS)
        self._fps = raw_fps if raw_fps and raw_fps > 0 else 25.0
        total_frames = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 1

        self._rendered_frames = []
        self._pb_frame.pack_forget()
        self._stop_playback()
        self.btn_save_video.config(state=tk.DISABLED)

        # Show inference progress bar
        self._infer_bar_frame.pack(fill=tk.X, padx=12, pady=(0, 2))
        self._infer_bar["value"] = 0
        self._infer_bar["maximum"] = 100
        self._lbl_infer_progress.config(text="Starting inference...")

        self.running = True
        self.btn_start.config(state=tk.DISABLED, bg="#dddddd")
        self.btn_stop.config(state=tk.NORMAL, bg="#ffcccb")
        self.btn_upload.config(state=tk.DISABLED)
        self._loop(total_frames)

    def stop_video(self):
        """Stop inference mid-way (user pressed Stop)."""
        self.running = False
        if self.cap is not None:
            self.cap.release()
            self.cap = None
        self._infer_bar_frame.pack_forget()
        self.btn_start.config(state=tk.NORMAL, bg="#90ee90")
        self.btn_stop.config(state=tk.DISABLED, bg="#dddddd")
        self.btn_upload.config(state=tk.NORMAL)
        if self._rendered_frames:
            self._enter_playback_mode()

    def _loop(self, total_frames):
        if not self.running or self.cap is None:
            return
        ok, frame = self.cap.read()
        if not ok:
            # Inference finished naturally — release cap and enter playback
            self.cap.release()
            self.cap = None
            self.running = False
            self._infer_bar_frame.pack_forget()
            self.btn_start.config(state=tk.NORMAL, bg="#90ee90")
            self.btn_stop.config(state=tk.DISABLED, bg="#dddddd")
            self.btn_upload.config(state=tk.NORMAL)
            self._enter_playback_mode()
            return

        conf        = float(self.conf_slider.get())
        preprocessed = self._letterbox_preprocess(frame)
        rendered    = self.app.run_inference(preprocessed, conf)
        self._rendered_frames.append(rendered.copy())
        self.app.show_frame(self.image_label, rendered)

        # Update progress bar
        done = len(self._rendered_frames)
        pct  = int(done / max(total_frames, 1) * 100)
        self._infer_bar["value"] = pct
        self._lbl_infer_progress.config(
            text=f"Inferencing frame {done} / {total_frames}  ({pct}%)")

        self.after(15, lambda: self._loop(total_frames))

    # ── playback mode ─────────────────────────────────────────────────────────

    def _enter_playback_mode(self):
        """Switch the UI from inference mode to playback mode."""
        if not self._rendered_frames:
            return

        total = len(self._rendered_frames)
        self._scrubber.config(to=total - 1)
        self._scrub_var.set(0)
        self._pb_index   = 0
        self._pb_playing = False

        dur = total / self._fps
        self._lbl_dur.config(text=self._fmt_time(dur))
        self._lbl_cur.config(text=self._fmt_time(0))
        self._lbl_frame_count.config(text=f"Frame 1 / {total}")
        self._btn_playpause.config(text="▶  Play", bg="#90ee90")

        # Show playback panel and save button
        self._pb_frame.pack(fill=tk.X, padx=0, pady=(0, 6))
        self.btn_save_video.config(state=tk.NORMAL)

        # Show first frame
        self._show_pb_frame(0)

    def _show_pb_frame(self, idx):
        """Render a specific frame index in the display."""
        if not self._rendered_frames:
            return
        idx = max(0, min(idx, len(self._rendered_frames) - 1))
        self._pb_index = idx
        self.app.show_frame(self.image_label, self._rendered_frames[idx])

        # Update scrubber and time labels (only when not being dragged)
        if not self._scrubbing:
            self._scrub_var.set(idx)
        cur = idx / self._fps
        self._lbl_cur.config(text=self._fmt_time(cur))
        self._lbl_frame_count.config(
            text=f"Frame {idx + 1} / {len(self._rendered_frames)}")

    def _toggle_play(self):
        if self._pb_playing:
            self._pb_playing = False
            self._btn_playpause.config(text="▶  Play", bg="#90ee90")
            if self._pb_after_id is not None:
                self.after_cancel(self._pb_after_id)
                self._pb_after_id = None
        else:
            # If at the end, restart from beginning
            if self._pb_index >= len(self._rendered_frames) - 1:
                self._pb_index = 0
            self._pb_playing = True
            self._btn_playpause.config(text="⏸  Pause", bg="#ffcccb")
            self._pb_tick()

    def _pb_tick(self):
        """Advance one frame during playback."""
        if not self._pb_playing:
            return
        self._show_pb_frame(self._pb_index)
        if self._pb_index >= len(self._rendered_frames) - 1:
            # Reached the end — stop playback
            self._pb_playing = False
            self._btn_playpause.config(text="▶  Play", bg="#90ee90")
            self._pb_after_id = None
            return

        self._pb_index += 1
        # Compute delay from FPS and playback speed multiplier
        speed_str = self._speed_var.get().replace("x", "")
        try:
            speed = float(speed_str)
        except ValueError:
            speed = 1.0
        delay_ms = max(1, int(1000 / (self._fps * speed)))
        self._pb_after_id = self.after(delay_ms, self._pb_tick)

    def _stop_playback(self):
        self._pb_playing = False
        if self._pb_after_id is not None:
            self.after_cancel(self._pb_after_id)
            self._pb_after_id = None
        if hasattr(self, "_btn_playpause"):
            self._btn_playpause.config(text="▶  Play", bg="#90ee90")

    def _skip_backward(self):
        skip = max(1, int(self._fps * 10))
        self._show_pb_frame(self._pb_index - skip)

    def _skip_forward(self):
        skip = max(1, int(self._fps * 10))
        self._show_pb_frame(self._pb_index + skip)

    # ── scrubber events ───────────────────────────────────────────────────────

    def _on_scrub_start(self, _event):
        self._scrubbing = True
        if self._pb_playing:
            self._toggle_play()   # pause while scrubbing

    def _on_scrub_drag(self, _event):
        idx = int(self._scrub_var.get())
        self._show_pb_frame(idx)

    def _on_scrub_end(self, _event):
        self._scrubbing = False
        idx = int(self._scrub_var.get())
        self._show_pb_frame(idx)

    # ── save ──────────────────────────────────────────────────────────────────

    def save_video(self):
        if not self._rendered_frames:
            return
        stem = os.path.splitext(os.path.basename(self.video_path))[0]
        save_path = filedialog.asksaveasfilename(
            initialfile=f"{stem}_inferred.mp4",
            defaultextension=".mp4",
            filetypes=[("MP4 Video", "*.mp4"), ("AVI Video", "*.avi"),
                       ("All Files", "*.*")]
        )
        if not save_path:
            return

        # Show a progress dialog while writing — can be slow for long videos
        prog = ProgressDialog(self.app.root, "Saving Video", "Writing frames...")
        prog.set_indeterminate("Encoding inferenced video...")

        def _write():
            try:
                h, w    = self._rendered_frames[0].shape[:2]
                fourcc  = cv2.VideoWriter_fourcc(*"mp4v")
                writer  = cv2.VideoWriter(save_path, fourcc, self._fps, (w, h))
                total   = len(self._rendered_frames)
                for i, f in enumerate(self._rendered_frames):
                    writer.write(f)
                    pct = int((i + 1) / total * 100)
                    prog.set_determinate(pct, f"Writing frame {i+1} / {total}")
                writer.release()
                self.after(0, lambda: prog.finish("Video saved successfully!", success=True))
                self.after(0, lambda: messagebox.showinfo(
                    "Saved", f"Inferenced video saved to:\n{save_path}"))
            except Exception as ex:
                self.after(0, lambda: prog.finish(f"Save failed: {ex}", success=False))

        threading.Thread(target=_write, daemon=True).start()

    # ── misc ──────────────────────────────────────────────────────────────────

    def on_display_resize(self, event):
        current = getattr(self.image_label, "current_frame", None)
        if current is not None:
            self.app.show_frame(self.image_label, current)

    def on_leave(self):
        self.stop_video()
        self._stop_playback()
        self.app.clear_label_image(self.image_label)


class ModelManagerDialog:
    """
    Unified model management dialog.
    Left panel  — list of all registered models with checkboxes for bulk delete.
    Right panel — class name / colour configuration for the selected model.
    """

    def __init__(self, parent, app):
        self.app = app
        self.parent = parent

        self.window = tk.Toplevel(parent)
        self.window.title("Configure Models")
        self.window.geometry("860x560")
        self.window.resizable(True, True)
        self.window.transient(parent)
        self.window.grab_set()
        center_window(self.window, parent)

        # Track per-model class config edits {model_path: {class_names, class_colors}}
        self._pending_config = {}

        self._build_ui()
        self._refresh_model_list()

    # ── UI construction ───────────────────────────────────────────────────────

    def _build_ui(self):
        # ── main horizontal split ─────────────────────────────────────────────
        pane = tk.PanedWindow(self.window, orient=tk.HORIZONTAL, sashwidth=5,
                              sashrelief=tk.RIDGE)
        pane.pack(expand=True, fill=tk.BOTH, padx=8, pady=8)

        # ── LEFT: model list ──────────────────────────────────────────────────
        left = tk.Frame(pane, width=280)
        pane.add(left, minsize=220)

        tk.Label(left, text="Registered Models", font=("Arial", 10, "bold")).pack(
            anchor="w", padx=6, pady=(6, 2))

        list_frame = tk.Frame(left)
        list_frame.pack(expand=True, fill=tk.BOTH, padx=4)

        self.lb_scroll = ttk.Scrollbar(list_frame)
        self.lb_scroll.pack(side=tk.RIGHT, fill=tk.Y)

        self.listbox = tk.Listbox(list_frame, selectmode=tk.EXTENDED,
                                   yscrollcommand=self.lb_scroll.set,
                                   activestyle="dotbox", font=("Arial", 9))
        self.listbox.pack(expand=True, fill=tk.BOTH)
        self.lb_scroll.config(command=self.listbox.yview)
        self.listbox.bind("<<ListboxSelect>>", self._on_model_select)

        btn_row = tk.Frame(left)
        btn_row.pack(fill=tk.X, padx=4, pady=6)
        tk.Button(btn_row, text="Select All",   command=self._select_all,   width=10).pack(side=tk.LEFT, padx=2)
        tk.Button(btn_row, text="Deselect All", command=self._deselect_all, width=10).pack(side=tk.LEFT, padx=2)
        tk.Button(btn_row, text="🗑 Delete Selected", command=self._delete_selected,
                  bg="#ffcccc", width=16).pack(side=tk.LEFT, padx=2)

        # ── RIGHT: class config ───────────────────────────────────────────────
        right = tk.Frame(pane)
        pane.add(right, minsize=360)

        self.cfg_title = tk.Label(right, text="Select a model to configure",
                                  font=("Arial", 10, "bold"))
        self.cfg_title.pack(anchor="w", padx=8, pady=(6, 2))

        # ── Display-name (alias) row ──────────────────────────────────────────
        alias_row = tk.Frame(right)
        alias_row.pack(fill=tk.X, padx=8, pady=(0, 6))

        tk.Label(alias_row, text="Display name:", width=13, anchor="w").pack(side=tk.LEFT)

        self._alias_var = tk.StringVar()
        self._alias_entry = tk.Entry(alias_row, textvariable=self._alias_var, width=28,
                                     state=tk.DISABLED)
        self._alias_entry.pack(side=tk.LEFT, padx=(0, 6))

        self._btn_save_alias = tk.Button(
            alias_row, text="Apply", width=8,
            command=self._apply_alias, state=tk.DISABLED, bg="#d0e8ff"
        )
        self._btn_save_alias.pack(side=tk.LEFT, padx=(0, 4))

        self._btn_clear_alias = tk.Button(
            alias_row, text="Clear", width=6,
            command=self._clear_alias, state=tk.DISABLED
        )
        self._btn_clear_alias.pack(side=tk.LEFT)

        tk.Label(alias_row, text="(leave blank to use filename)",
                 fg="#999999", font=("Arial", 8)).pack(side=tk.LEFT, padx=(8, 0))

        # Scrollable area for class rows
        cfg_outer = tk.Frame(right)
        cfg_outer.pack(expand=True, fill=tk.BOTH, padx=4)

        self.cfg_canvas = tk.Canvas(cfg_outer, highlightthickness=0)
        cfg_sb = ttk.Scrollbar(cfg_outer, orient="vertical",
                                command=self.cfg_canvas.yview)
        self.cfg_inner = tk.Frame(self.cfg_canvas)
        self.cfg_canvas_win = self.cfg_canvas.create_window(
            (0, 0), window=self.cfg_inner, anchor="nw")

        self.cfg_canvas.configure(yscrollcommand=cfg_sb.set)
        cfg_sb.pack(side=tk.RIGHT, fill=tk.Y)
        self.cfg_canvas.pack(side=tk.LEFT, expand=True, fill=tk.BOTH)

        self.cfg_inner.bind("<Configure>", lambda e: self.cfg_canvas.configure(
            scrollregion=self.cfg_canvas.bbox("all")))
        self.cfg_canvas.bind("<Configure>", lambda e: self.cfg_canvas.itemconfig(
            self.cfg_canvas_win, width=e.width))

        # Import config + Save config buttons
        btn_row_cfg = tk.Frame(right)
        btn_row_cfg.pack(pady=(4, 8))

        self.btn_import_cfg = tk.Button(
            btn_row_cfg, text="📂  Import Config File",
            command=self._import_class_config,
            width=20, state=tk.DISABLED
        )
        self.btn_import_cfg.pack(side=tk.LEFT, padx=6)

        self.btn_save_cfg = tk.Button(
            btn_row_cfg, text="💾  Save Class Config",
            command=self._save_class_config,
            bg="#90ee90", state=tk.DISABLED, width=20
        )
        self.btn_save_cfg.pack(side=tk.LEFT, padx=6)

        # Internal state for class editor
        self._cfg_model_path = None
        self._cfg_entries = {}       # {class_idx: Entry widget}
        self._cfg_colors  = {}       # {class_idx: (r,g,b)}
        self._cfg_color_btns = {}    # {class_idx: Button widget}
        # alias widgets initialised above; bind Enter key
        self._alias_entry.bind("<Return>", lambda _e: self._apply_alias())

    # ── model list helpers ────────────────────────────────────────────────────

    def _refresh_model_list(self):
        self.listbox.delete(0, tk.END)
        self._models = self.app.registry.discover_models()
        for m in self._models:
            self.listbox.insert(tk.END, m["name"])

    def _select_all(self):
        self.listbox.select_set(0, tk.END)

    def _deselect_all(self):
        self.listbox.select_clear(0, tk.END)

    def _delete_selected(self):
        indices = list(self.listbox.curselection())
        if not indices:
            messagebox.showinfo("Nothing selected",
                                "Select one or more models in the list first.")
            return
        names = [self._models[i]["name"] for i in indices]
        msg = "Delete the following model(s)?\n\n" + "\n".join(f"• {n}" for n in names)
        if not messagebox.askyesno("Confirm Delete", msg, icon="warning"):
            return

        # Stop inference if any deleted model is active
        for i in indices:
            path = self._models[i]["path"]
            if os.path.abspath(path) == os.path.abspath(self.app.active_model_path):
                self.app._stop_all_pages()

        errors = []
        for i in indices:
            try:
                self.app.registry.remove_model(self._models[i]["path"])
            except Exception as ex:
                errors.append(f"{self._models[i]['name']}: {ex}")

        if errors:
            messagebox.showerror("Delete Errors", "\n".join(errors))

        # Clear class editor if deleted model was being configured
        if self._cfg_model_path and not os.path.exists(self._cfg_model_path):
            self._clear_class_editor()

        self._refresh_model_list()

    # ── class config helpers ──────────────────────────────────────────────────

    def _on_model_select(self, _event=None):
        sel = self.listbox.curselection()
        if len(sel) != 1:
            return  # Only configure when exactly one model is selected
        model = self._models[sel[0]]
        self._load_class_editor(model["path"], model["name"])

    def _clear_class_editor(self):
        for w in self.cfg_inner.winfo_children():
            w.destroy()
        self._cfg_entries.clear()
        self._cfg_colors.clear()
        self._cfg_color_btns.clear()
        self._cfg_model_path = None
        self.cfg_title.config(text="Select a model to configure")
        self.btn_save_cfg.config(state=tk.DISABLED)
        self.btn_import_cfg.config(state=tk.DISABLED)
        # Reset alias widgets
        self._alias_var.set("")
        self._alias_entry.config(state=tk.DISABLED)
        self._btn_save_alias.config(state=tk.DISABLED)
        self._btn_clear_alias.config(state=tk.DISABLED)

    def _load_class_editor(self, model_path, model_name):
        self._cfg_model_path = model_path
        self.cfg_title.config(text=f"Classes — {model_name}")
        self.btn_save_cfg.config(state=tk.NORMAL)
        self.btn_import_cfg.config(state=tk.NORMAL)

        # Populate alias field with current alias (if any)
        current_alias = self.app.registry.get_model_alias(model_path)
        self._alias_var.set(current_alias)
        self._alias_entry.config(state=tk.NORMAL)
        self._btn_save_alias.config(state=tk.NORMAL)
        self._btn_clear_alias.config(state=tk.NORMAL)

        existing = self.app.registry.get_model_config(model_path)
        saved_names  = existing.get("class_names",  {})
        saved_colors = existing.get("class_colors", {})

        # Detect number of classes
        num_classes = self._detect_num_classes(model_path)

        # Clear old rows
        for w in self.cfg_inner.winfo_children():
            w.destroy()
        self._cfg_entries.clear()
        self._cfg_colors.clear()
        self._cfg_color_btns.clear()

        DEFAULT_COLORS = [
            (255, 0, 0), (0, 200, 0), (0, 0, 255), (255, 200, 0),
            (255, 0, 255), (0, 200, 255), (128, 0, 0), (0, 128, 0),
            (0, 0, 128), (128, 128, 0),
        ]

        header = tk.Frame(self.cfg_inner)
        header.pack(fill=tk.X, padx=4, pady=(4, 0))
        tk.Label(header, text="ID",    width=4,  anchor="w", font=("Arial", 8, "bold")).pack(side=tk.LEFT)
        tk.Label(header, text="Class Name", width=18, anchor="w", font=("Arial", 8, "bold")).pack(side=tk.LEFT)
        tk.Label(header, text="Colour", width=8, anchor="w", font=("Arial", 8, "bold")).pack(side=tk.LEFT)
        ttk.Separator(self.cfg_inner, orient="horizontal").pack(fill=tk.X, pady=2)

        for i in range(num_classes):
            row = tk.Frame(self.cfg_inner)
            row.pack(fill=tk.X, padx=4, pady=2)

            tk.Label(row, text=str(i), width=4, anchor="w").pack(side=tk.LEFT)

            entry = tk.Entry(row, width=18)
            entry.insert(0, saved_names.get(i, saved_names.get(str(i), f"cls {i}")))
            entry.pack(side=tk.LEFT, padx=4)
            self._cfg_entries[i] = entry

            default_color = DEFAULT_COLORS[i % len(DEFAULT_COLORS)]
            color = saved_colors.get(i, saved_colors.get(str(i), default_color))
            if isinstance(color, list):
                color = tuple(color)
            self._cfg_colors[i] = color

            btn = tk.Button(row, text="  ", width=4,
                            bg=self._rgb_to_tk(color),
                            relief=tk.GROOVE,
                            command=lambda idx=i: self._pick_color(idx))
            btn.pack(side=tk.LEFT)
            self._cfg_color_btns[i] = btn

    def _apply_alias(self):
        """Save the alias entered in the display-name field and refresh the list."""
        if not self._cfg_model_path:
            return
        new_alias = self._alias_var.get().strip()
        self.app.registry.save_model_alias(self._cfg_model_path, new_alias)
        # Refresh both the left-panel list and the main dropdown
        self._refresh_model_list()
        self.app.refresh_model_dropdown()
        # Keep the same model selected in the listbox after refresh
        for i, m in enumerate(self._models):
            if os.path.abspath(m["path"]) == os.path.abspath(self._cfg_model_path):
                self.listbox.select_clear(0, tk.END)
                self.listbox.select_set(i)
                self.listbox.see(i)
                break
        # Update title to reflect new display name
        display = new_alias if new_alias else os.path.basename(self._cfg_model_path)
        self.cfg_title.config(text=f"Classes — {display}")
        # Update main status label if this is the active model
        if os.path.abspath(self._cfg_model_path) == os.path.abspath(self.app.active_model_path):
            active_info = next(
                (m for m in self._models
                 if os.path.abspath(m["path"]) == os.path.abspath(self._cfg_model_path)),
                None
            )
            if active_info:
                self.app.model_var.set(active_info["name"])
                self.app.lbl_status.config(
                    text=f"Loaded: {active_info['name']}", fg="green")

    def _clear_alias(self):
        """Clear the alias field (does not save until Apply is clicked)."""
        self._alias_var.set("")
        self._alias_entry.focus_set()

    def _detect_num_classes(self, model_path):
        try:
            if model_path.endswith(".pt"):
                mdl = YOLO(model_path)
                return len(mdl.names)
            ckpt = torch.load(model_path, map_location="cpu", weights_only=False)
            state = ckpt.get("model_state_dict", ckpt) if isinstance(ckpt, dict) else {}
            key = "roi_heads.box_predictor.cls_score.weight"
            if key in state:
                return state[key].shape[0]
        except Exception:
            pass
        return 10

    def _rgb_to_tk(self, color):
        return f"#{color[2]:02x}{color[1]:02x}{color[0]:02x}"

    def _pick_color(self, class_idx):
        from tkinter import colorchooser
        current = self._cfg_colors.get(class_idx, (0, 200, 0))
        result = colorchooser.askcolor(color=self._rgb_to_tk(current),
                                       title=f"Choose colour for Class {class_idx}")
        if result[1]:
            hex_color = result[1]
            # Convert hex #RRGGBB → (B, G, R) for OpenCV
            r = int(hex_color[1:3], 16)
            g = int(hex_color[3:5], 16)
            b = int(hex_color[5:7], 16)
            self._cfg_colors[class_idx] = (b, g, r)
            self._cfg_color_btns[class_idx].config(bg=hex_color)

    def _import_class_config(self):
        """
        Let the user pick a COCO JSON or YAML config file and auto-fill
        the class name entries from it.  Supported formats:
          • COCO annotation JSON  — categories[].name
          • YOLO / Ultralytics YAML — names: list or names: {id: name} dict
          • Generic YAML          — classes: list  /  labels: list
        """
        path = filedialog.askopenfilename(
            title="Select Config File",
            filetypes=[
                ("Config files", "*.json *.yaml *.yml"),
                ("COCO JSON",    "*.json"),
                ("YAML",         "*.yaml *.yml"),
                ("All files",    "*.*"),
            ]
        )
        if not path:
            return

        ext = os.path.splitext(path)[1].lower()
        try:
            if ext == ".json":
                names = self._parse_coco_json(path)
            elif ext in (".yaml", ".yml"):
                names = self._parse_yaml(path)
            else:
                messagebox.showerror("Unsupported File",
                                     "Please choose a .json, .yaml, or .yml file.")
                return
        except Exception as ex:
            messagebox.showerror("Parse Error",
                                 f"Could not read the config file:\n{ex}")
            return

        if not names:
            messagebox.showwarning("No Classes Found",
                                   "No class names were found in the selected file.\n\n"
                                   "Supported keys: categories (COCO JSON), "
                                   "names / classes / labels (YAML).")
            return

        # Fill in however many entries exist — extras are left unchanged
        filled = 0
        for i, name in enumerate(names):
            if i in self._cfg_entries:
                self._cfg_entries[i].delete(0, tk.END)
                self._cfg_entries[i].insert(0, name)
                filled += 1

        extra = len(names) - filled
        msg = f"Filled {filled} class name(s) from:\n{os.path.basename(path)}"
        if extra > 0:
            msg += f"\n\n({extra} name(s) in the file exceeded the model's class count and were ignored.)"
        messagebox.showinfo("Import Successful", msg)

    @staticmethod
    def _parse_coco_json(path):
        """Return ordered list of class names from a COCO annotation JSON."""
        import json
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)

        categories = data.get("categories", [])
        if not categories:
            raise ValueError("No 'categories' key found in JSON.")

        # Sort by id so index order matches model class IDs
        categories = sorted(categories, key=lambda c: c.get("id", 0))
        return [c["name"] for c in categories]

    @staticmethod
    def _parse_yaml(path):
        """
        Return ordered list of class names from a YAML file.
        Handles:
          names: [cat, dog, fish]          # YOLO list form
          names: {0: cat, 1: dog}          # YOLO dict form
          classes: [cat, dog]              # generic list
          labels: [cat, dog]              # generic list
        """
        import yaml
        with open(path, "r", encoding="utf-8") as f:
            data = yaml.safe_load(f)

        # Try common keys in priority order
        for key in ("names", "classes", "labels"):
            value = data.get(key)
            if value is None:
                continue
            if isinstance(value, list):
                return [str(v) for v in value]
            if isinstance(value, dict):
                # Sort by numeric key
                ordered = sorted(value.items(), key=lambda kv: int(kv[0]))
                return [str(v) for _, v in ordered]

        raise ValueError("No recognised class-list key found (names / classes / labels).")

    def _save_class_config(self):
        if not self._cfg_model_path:
            return
        class_names  = {i: e.get() or f"cls {i}" for i, e in self._cfg_entries.items()}
        class_colors = dict(self._cfg_colors)
        self.app.registry.save_model_config(self._cfg_model_path,
                                             class_names, class_colors)
        # Reload runner if this is the active model so colours take effect immediately
        if os.path.abspath(self._cfg_model_path) == os.path.abspath(self.app.active_model_path):
            self.app.load_model_with_progress(self._cfg_model_path)
        messagebox.showinfo("Saved", "Class configuration saved successfully.")


class FishNetApp:
    def __init__(self, root):
        self.root = root
        self.root.title("FishNetMy")
        self.root.geometry("1200x850")
        self.root.minsize(1020, 720)

        self.project_root = os.path.dirname(os.path.abspath(__file__))
        self.registry = ModelRegistry(self.project_root)
        # Select compute device — prefer CUDA (NVIDIA GPU), fall back to CPU.
        # torch.cuda.is_available() returns False when:
        #   • No NVIDIA GPU is present
        #   • The GPU driver or CUDA toolkit is not installed
        #   • PyTorch was installed without CUDA support (e.g. the CPU-only wheel)
        if torch.cuda.is_available():
            self.device = torch.device("cuda")
            _gpu_name   = torch.cuda.get_device_name(0)
        else:
            self.device = torch.device("cpu")
            _gpu_name   = None
        self._gpu_name = _gpu_name

        self.models = []
        self.model_map = {}
        self.active_runner = None
        self.active_model_path = ""

        self.current_page = None
        self.pages = {}

        self._build_layout()
        self.refresh_model_dropdown()
        self._try_restore_last_model()
        self.show_page("live")

        self.root.protocol("WM_DELETE_WINDOW", self.on_close)

    def _build_layout(self):
        nav = tk.Frame(self.root, bd=1, relief=tk.GROOVE)
        nav.pack(side=tk.TOP, fill=tk.X, padx=10, pady=(10, 5))

        tk.Button(nav, text="Live Inference", width=14, command=lambda: self.show_page("live")).pack(side=tk.LEFT, padx=5, pady=6)
        tk.Button(nav, text="Image Inference", width=14, command=lambda: self.show_page("image")).pack(side=tk.LEFT, padx=5, pady=6)
        tk.Button(nav, text="Folder Inference", width=14, command=lambda: self.show_page("folder")).pack(side=tk.LEFT, padx=5, pady=6)
        tk.Button(nav, text="Video Inference", width=14, command=lambda: self.show_page("video")).pack(side=tk.LEFT, padx=5, pady=6)

        tk.Label(nav, text="Model:").pack(side=tk.LEFT, padx=(18, 5))
        self.model_var = tk.StringVar()
        self.model_combo = ttk.Combobox(nav, textvariable=self.model_var, state="readonly", width=44)
        self.model_combo.pack(side=tk.LEFT, padx=4, pady=6)
        self.model_combo.bind("<<ComboboxSelected>>", self.on_model_selected)

        tk.Button(nav, text="Upload Model", command=self.upload_model).pack(side=tk.LEFT, padx=(8, 2), pady=6)
        tk.Button(nav, text="Configure", command=self.open_configure_dialog).pack(side=tk.LEFT, padx=(2, 8), pady=6)

        self.lbl_status = tk.Label(nav, text="No model loaded", fg="red")
        self.lbl_status.pack(side=tk.LEFT, padx=12)

        # GPU / CPU indicator — right-aligned in the nav bar
        if self._gpu_name:
            gpu_text  = f"🟢 GPU: {self._gpu_name}"
            gpu_color = "#006600"
        else:
            gpu_text  = "🔴 CPU only (no CUDA GPU detected)"
            gpu_color = "#993300"
        self.lbl_device = tk.Label(nav, text=gpu_text, fg=gpu_color,
                                   font=("Arial", 8))
        self.lbl_device.pack(side=tk.RIGHT, padx=12)

        self.page_container = tk.Frame(self.root)
        self.page_container.pack(expand=True, fill=tk.BOTH, padx=10, pady=(5, 10))
        self.page_container.grid_rowconfigure(0, weight=1)
        self.page_container.grid_columnconfigure(0, weight=1)

        self.pages["live"] = LiveInferencePage(self.page_container, self)
        self.pages["image"] = ImageInferencePage(self.page_container, self)
        self.pages["folder"] = FolderInferencePage(self.page_container, self)
        self.pages["video"] = VideoInferencePage(self.page_container, self)

        for page in self.pages.values():
            page.grid(row=0, column=0, sticky="nsew")

    def refresh_model_dropdown(self):
        self.models = self.registry.discover_models()
        self.model_map = {m["name"]: m for m in self.models}
        names = list(self.model_map.keys())
        self.model_combo["values"] = names
        if not names:
            self.model_var.set("")

    def _try_restore_last_model(self):
        last_model = self.registry.get_last_selected()
        if not last_model:
            return
        match = None
        for model in self.models:
            if os.path.abspath(model["path"]) == os.path.abspath(last_model):
                match = model
                break
        if match:
            self.model_var.set(match["name"])
            self.load_model_with_progress(match["path"])

    def on_model_selected(self, _event=None):
        selected = self.model_var.get()
        model_info = self.model_map.get(selected)
        if not model_info:
            return
        self.load_model_with_progress(model_info["path"])

    def _stop_all_pages(self):
        for page in self.pages.values():
            page.on_leave()

    def _detect_runner_type(self, model_path):
        suffix = Path(model_path).suffix.lower()
        lower = model_path.lower()
        if suffix == ".pt":
            return "yolo"
        if "mask" in lower:
            return "mask"
        if "fast" in lower or "faster" in lower:
            return "faster"

        # For ambiguous .pth files, infer from checkpoint keys.
        if suffix == ".pth":
            try:
                ckpt = torch.load(model_path, map_location="cpu", weights_only=False)
            except TypeError:
                # Fallback for older PyTorch versions that don't support weights_only
                ckpt = torch.load(model_path, map_location="cpu")
            state = ckpt.get("model_state_dict", ckpt) if isinstance(ckpt, dict) else {}
            keys = state.keys() if isinstance(state, dict) else []
            if any("mask_predictor" in k for k in keys):
                return "mask"
            return "faster"
        raise ValueError("Unsupported model extension. Use .pt or .pth")

    def _load_runner(self, model_path):
        runner_type = self._detect_runner_type(model_path)
        config = self.registry.get_model_config(model_path)
        
        # JSON serializes int keys as strings — normalize them back to int
        raw_names  = config.get("class_names",  {}) if config else {}
        raw_colors = config.get("class_colors", {}) if config else {}
        class_names  = {int(k): v for k, v in raw_names.items()}
        class_colors = {int(k): tuple(v) if isinstance(v, list) else v
                        for k, v in raw_colors.items()}
        
        if runner_type == "yolo":
            return YOLORunner(model_path, self.device, class_names, class_colors)
        if runner_type == "faster":
            return FasterRCNNRunner(model_path, self.device, class_names, class_colors)
        if runner_type == "mask":
            return MaskRCNNRunner(model_path, self.device, class_names, class_colors)
        raise ValueError("Could not determine model type")

    def load_model_with_progress(self, model_path):
        model_path = os.path.abspath(model_path)
        self._stop_all_pages()

        dialog = ProgressDialog(self.root, "Loading Model", "Preparing model load...")
        queue = Queue()

        def worker():
            try:
                queue.put(("indeterminate", "Loading model into memory..."))
                runner = self._load_runner(model_path)
                queue.put(("done", runner, None))
            except Exception as ex:
                queue.put(("done", None, str(ex)))

        thread = threading.Thread(target=worker, daemon=True)
        thread.start()

        def poll():
            try:
                while True:
                    msg = queue.get_nowait()
                    kind = msg[0]
                    if kind == "indeterminate":
                        dialog.set_indeterminate(msg[1])
                    elif kind == "done":
                        runner, error = msg[1], msg[2]
                        if error:
                            dialog.finish(f"Failed to load model: {error}", success=False)
                            self.lbl_status.config(text="Model load failed", fg="red")
                            messagebox.showerror("Model Load Error", error)
                        else:
                            self.active_runner = runner
                            self.active_model_path = model_path
                            self.registry.remember_last_selected(model_path)
                            self.lbl_status.config(text=f"Loaded: {os.path.basename(model_path)}", fg="green")
                            dialog.finish("Model loaded successfully", success=True)
                        self.root.after(700, dialog.close)
                        return
            except Empty:
                pass
            self.root.after(80, poll)

        poll()

    def upload_model(self):
        src_path = filedialog.askopenfilename(filetypes=[("Model files", "*.pt *.pth")])
        if not src_path:
            return

        src_path = os.path.abspath(src_path)
        filename = os.path.basename(src_path)
        dest_path = os.path.join(self.registry.custom_models_root, filename)

        if os.path.abspath(src_path) == os.path.abspath(dest_path):
            messagebox.showinfo("Upload Model", "This model is already in the persistent upload folder.")
            self.registry.add_uploaded_model(dest_path)
            self.refresh_model_dropdown()
            return

        if os.path.exists(dest_path):
            overwrite = messagebox.askyesno("Overwrite Model", f"{filename} already exists. Overwrite it?")
            if not overwrite:
                return

        dialog = ProgressDialog(self.root, "Uploading Model", "Preparing upload...")
        queue = Queue()

        def copy_worker():
            try:
                total_size = os.path.getsize(src_path)
                copied = 0
                chunk = 1024 * 1024

                os.makedirs(self.registry.custom_models_root, exist_ok=True)
                with open(src_path, "rb") as src, open(dest_path, "wb") as dst:
                    while True:
                        data = src.read(chunk)
                        if not data:
                            break
                        dst.write(data)
                        copied += len(data)
                        percent = int((copied / max(total_size, 1)) * 100)
                        queue.put(("progress", percent, f"Uploading {filename}..."))

                queue.put(("indeterminate", "Validating uploaded model..."))
                _ = self._load_runner(dest_path)

                self.registry.add_uploaded_model(dest_path)
                queue.put(("done", None))
            except Exception as ex:
                queue.put(("error", str(ex)))

        thread = threading.Thread(target=copy_worker, daemon=True)
        thread.start()

        def poll():
            try:
                while True:
                    msg = queue.get_nowait()
                    kind = msg[0]
                    if kind == "progress":
                        dialog.set_determinate(msg[1], msg[2])
                    elif kind == "indeterminate":
                        dialog.set_indeterminate(msg[1])
                    elif kind == "done":
                        dialog.finish("Model uploaded and validated", success=True)
                        self.refresh_model_dropdown()
                        model_candidates = [m for m in self.models if os.path.abspath(m["path"]) == os.path.abspath(dest_path)]
                        if model_candidates:
                            self.model_var.set(model_candidates[0]["name"])
                            self.load_model_with_progress(dest_path)
                        self.root.after(800, dialog.close)
                        return
                    elif kind == "error":
                        if os.path.exists(dest_path):
                            try:
                                os.remove(dest_path)
                            except OSError:
                                pass
                        dialog.finish(f"Upload failed: {msg[1]}", success=False)
                        messagebox.showerror("Upload Error", msg[1])
                        self.root.after(1500, dialog.close)
                        return
            except Empty:
                pass
            self.root.after(80, poll)

        poll()

    def open_configure_dialog(self):
        """Open the unified model configuration & management dialog."""
        dialog = ModelManagerDialog(self.root, self)
        self.root.wait_window(dialog.window)
        self.refresh_model_dropdown()
        # If the active model was removed, clear status
        if self.active_model_path and not os.path.exists(self.active_model_path):
            self.active_runner = None
            self.active_model_path = ""
            self.lbl_status.config(text="No model loaded", fg="red")
            self.model_var.set("")

    def on_model_right_click(self, event):
        """Show context menu for model management."""
        if not self.model_var.get():
            return
        
        # Create context menu
        self.model_context_menu = tk.Menu(self.root, tearoff=0)
        self.model_context_menu.add_command(label="Configure Classes...", command=self.configure_model_classes)
        self.model_context_menu.add_command(label="Remove Model", command=self.remove_selected_model)
        
        try:
            self.model_context_menu.post(event.x_root, event.y_root)
        except Exception:
            pass
    
    def configure_model_classes(self):
        """Open dialog to configure class names and colors."""
        selected = self.model_var.get()
        model_info = self.model_map.get(selected)
        if not model_info:
            return
        
        existing_config = self.registry.get_model_config(model_info["path"])
        dialog = ClassConfigDialog(self.root, model_info["path"], existing_config)
        self.root.wait_window(dialog.window)
        
        config = dialog.get_config()
        self.registry.save_model_config(model_info["path"], config["class_names"], config["class_colors"])
        
        # Reload model with new config
        self.load_model_with_progress(model_info["path"])
    
    def remove_selected_model(self):
        """Remove the selected model."""
        selected = self.model_var.get()
        model_info = self.model_map.get(selected)
        if not model_info:
            return
        
        # Confirm removal
        result = messagebox.askyesno(
            "Remove Model",
            f"Are you sure you want to remove '{model_info['name']}'?\n\n"
            f"This will delete the model file from your computer.",
            icon='warning'
        )
        
        if not result:
            return
        
        try:
            # Stop any running inference
            self._stop_all_pages()
            
            # Remove from registry and delete file
            self.registry.remove_model(model_info["path"])
            
            # Clear active model if it was the removed one
            if self.active_model_path == model_info["path"]:
                self.active_runner = None
                self.active_model_path = ""
                self.lbl_status.config(text="No model loaded", fg="red")
            
            # Refresh dropdown
            self.refresh_model_dropdown()
            self.model_var.set("")
            
            messagebox.showinfo("Model Removed", f"Model '{model_info['name']}' has been removed.")
        except Exception as e:
            messagebox.showerror("Remove Failed", f"Failed to remove model:\n{e}")

    def run_inference(self, frame_bgr, conf_threshold):
        if self.active_runner is None:
            return frame_bgr
        try:
            return self.active_runner.predict(frame_bgr, conf_threshold)
        except Exception as ex:
            self.lbl_status.config(text=f"Inference error: {ex}", fg="red")
            return frame_bgr

    @staticmethod
    def show_frame(label, frame_bgr):
        label.current_frame = frame_bgr
        rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        image = Image.fromarray(rgb)
        
        # Get display area size
        display_width = label.winfo_width()
        display_height = label.winfo_height()
        
        # If display size not available yet, use original size
        if display_width <= 1 or display_height <= 1:
            display_width = max(640, display_width)
            display_height = max(480, display_height)
        
        # Calculate aspect ratio and resize
        orig_width, orig_height = image.size
        scale = min(display_width / orig_width, display_height / orig_height)
        new_width = max(1, int(orig_width * scale))
        new_height = max(1, int(orig_height * scale))

        # Resize for both downscale and upscale while preserving aspect ratio.
        if new_width != orig_width or new_height != orig_height:
            resample = Image.Resampling.LANCZOS if scale < 1.0 else Image.Resampling.BICUBIC
            image = image.resize((new_width, new_height), resample)
        
        imgtk = ImageTk.PhotoImage(image=image)
        label.imgtk = imgtk
        label.configure(image=imgtk, anchor="center")

    @staticmethod
    def clear_label_image(label):
        label.configure(image="")
        label.imgtk = None
        label.current_frame = None

    def show_page(self, page_name):
        if self.current_page is not None:
            self.current_page.on_leave()
        page = self.pages[page_name]
        page.tkraise()
        page.on_show()
        self.current_page = page

    def on_close(self):
        self._stop_all_pages()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        self.root.destroy()


if __name__ == "__main__":
    root = tk.Tk()
    app = FishNetApp(root)
    root.mainloop()
