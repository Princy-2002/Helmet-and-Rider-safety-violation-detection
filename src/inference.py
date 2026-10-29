"""
inference.py
------------
Real-Time Inference and Dynamic Bounding-Box Rendering Pipeline
for Helmet & Rider-Safety Violation Detection.

Features:
- Ultralytics YOLOv8 inference wrapper with configurable device (CPU / CUDA).
- Interactive confidence and IoU threshold controls.
- Dynamic color-coded annotation:
    * Green: 'With Helmet' (Compliant)
    * Red: 'Without Helmet' (Violation)
- Real-time video frame overlay with rider counts, violation counters, and FPS.
- Direct integration with ViolationLogger to record temporal events.
"""

import os
import time
from typing import List, Dict, Any, Optional, Tuple, Union
import cv2
import numpy as np
from PIL import Image

try:
    from ultralytics import YOLO
except ImportError:
    YOLO = None

try:
    from violation_logger import ViolationLogger
except ImportError:
    from src.violation_logger import ViolationLogger


# Default class configuration matching trained model
CLASS_NAMES = {
    0: "With Helmet",
    1: "Without Helmet"
}

# Color palette: BGR format for OpenCV
CLASS_COLORS = {
    0: (46, 204, 113),   # Emerald Green (Compliant)
    1: (46, 46, 231)     # Bright Crimson Red (Violation)
}


class HelmetDetector:
    """
    YOLOv8-based inference engine for detecting motorcycle rider safety compliance.
    """

    def __init__(
        self,
        model_path: Optional[str] = None,
        conf_threshold: float = 0.35,
        iou_threshold: float = 0.45,
        device: Optional[str] = None
    ):
        """
        Initialize the detector.

        Args:
            model_path: Path to YOLOv8 weights (.pt). If None or not found,
                        attempts to load standard fallback.
            conf_threshold: Initial confidence threshold (0.05 to 1.0).
            iou_threshold: Initial IoU / NMS threshold.
            device: 'cuda', 'cpu', or None (auto-detect).
        """
        self.conf_threshold = conf_threshold
        self.iou_threshold = iou_threshold
        self.device = device
        self.model = None
        self.model_path = model_path

        self._load_model(model_path)

    def _load_model(self, model_path: Optional[str]):
        """Load YOLO model safely with fallback options."""
        if YOLO is None:
            print("[WARNING] Ultralytics is not installed. Inference will be disabled.")
            return

        candidates = []
        if model_path:
            candidates.append(model_path)
        candidates.extend([
            "models/best.pt",
            "models/baseline_yolov8n.pt",
            "yolov8n.pt"
        ])

        loaded = False
        for path in candidates:
            if os.path.exists(path):
                try:
                    print(f"[INFO] Loading YOLO model from '{path}'...")
                    self.model = YOLO(path)
                    self.model_path = path
                    loaded = True
                    break
                except Exception as e:
                    print(f"[WARNING] Could not load model from '{path}': {e}")

        if not loaded:
            print("[INFO] No local weights found. Initializing 'yolov8n.pt' directly...")
            try:
                self.model = YOLO("yolov8n.pt")
                self.model_path = "yolov8n.pt"
            except Exception as e:
                print(f"[ERROR] Failed to initialize default YOLO model: {e}")

    def update_thresholds(self, conf: Optional[float] = None, iou: Optional[float] = None):
        """
        Update confidence and IoU thresholds dynamically (useful for UI sliders).
        """
        if conf is not None:
            self.conf_threshold = float(np.clip(conf, 0.01, 1.0))
        if iou is not None:
            self.iou_threshold = float(np.clip(iou, 0.01, 1.0))

    def enhance_contrast(self, image: np.ndarray) -> np.ndarray:
        """
        Apply CLAHE (Contrast Limited Adaptive Histogram Equalization)
        for night-vision, rain, and low-light traffic feeds.
        """
        if image is None or image.size == 0:
            return image
        lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)
        clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
        cl = clahe.apply(l)
        limg = cv2.merge((cl, a, b))
        return cv2.cvtColor(limg, cv2.COLOR_LAB2BGR)

    def detect(self, image: Union[np.ndarray, Image.Image, str], track: bool = False) -> List[Dict[str, Any]]:
        """
        Run inference on an input frame or image with optional ByteTrack object tracking.

        Args:
            image: OpenCV BGR image, PIL Image, or file path.
            track: Whether to enable persistent multi-object tracking (MOT).

        Returns:
            List of parsed detection dictionaries.
        """
        if self.model is None:
            return []

        # Run YOLO inference with or without persistent tracking
        if track and hasattr(self.model, "track"):
            try:
                results = self.model.track(
                    source=image,
                    conf=self.conf_threshold,
                    iou=self.iou_threshold,
                    device=self.device,
                    persist=True,
                    agnostic_nms=True,
                    verbose=False
                )
            except Exception:
                # Fallback to standard predict if tracking tracker config fails
                results = self.model.predict(
                    source=image,
                    conf=self.conf_threshold,
                    iou=self.iou_threshold,
                    device=self.device,
                    agnostic_nms=True,
                    verbose=False
                )
        else:
            results = self.model.predict(
                source=image,
                conf=self.conf_threshold,
                iou=self.iou_threshold,
                device=self.device,
                agnostic_nms=True,
                verbose=False
            )

        detections = []
        if not results:
            return detections

        r = results[0]
        boxes = r.boxes

        if boxes is None or len(boxes) == 0:
            return detections

        for box in boxes:
            cls_id = int(box.cls[0].item())
            conf = float(box.conf[0].item())
            xyxy = box.xyxy[0].cpu().numpy().astype(int).tolist()
            track_id = int(box.id[0].item()) if (hasattr(box, "id") and box.id is not None) else None

            # Map class name using model names or fallback
            if hasattr(self.model, "names") and cls_id in self.model.names:
                cls_name = self.model.names[cls_id]
            else:
                cls_name = CLASS_NAMES.get(cls_id, f"Class {cls_id}")

            detections.append({
                "class_id": cls_id,
                "class_name": cls_name,
                "confidence": round(conf, 3),
                "box": xyxy,
                "track_id": track_id
            })

        return detections

    @property
    def is_custom_helmet_model(self) -> bool:
        """Return True if model was specifically fine-tuned for Helmet vs No-Helmet classes."""
        if self.model and hasattr(self.model, "names") and isinstance(self.model.names, dict):
            names_vals = [str(v).lower() for v in self.model.names.values()]
            return any("helmet" in v for v in names_vals)
        return False

    def draw_annotations(
        self,
        frame: np.ndarray,
        detections: List[Dict[str, Any]],
        show_labels: bool = True,
        show_fps: Optional[float] = None
    ) -> np.ndarray:
        """
        Render dynamic bounding boxes, violation tags, and HUD overlay on the frame.
        """
        annotated = frame.copy()
        h, w = annotated.shape[:2]

        with_helmet_count = 0
        without_helmet_count = 0

        for det in detections:
            cls_id = det["class_id"]
            cls_name = det["class_name"]
            conf = det["confidence"]
            box = det["box"]

            track_prefix = f"ID #{det['track_id']} | " if det.get("track_id") is not None else ""

            if self.is_custom_helmet_model:
                if cls_id == 1 or "Without" in cls_name or "no helmet" in cls_name.lower():
                    without_helmet_count += 1
                    color = CLASS_COLORS[1]
                    label_text = f"{track_prefix}VIOLATION: No Helmet ({conf:.2f})"
                else:
                    with_helmet_count += 1
                    color = CLASS_COLORS[0]
                    label_text = f"{track_prefix}Helmet ({conf:.2f})"
            else:
                # Generic 80-Class COCO model:
                # Do NOT falsely mark generic "person" detections as compliant helmets!
                if cls_name.lower() in ["motorcycle", "bicycle"]:
                    color = (46, 204, 113)  # Green for vehicle
                    label_text = f"{track_prefix}{cls_name.upper()} ({conf:.2f})"
                elif cls_name.lower() == "person":
                    without_helmet_count += 1
                    color = (46, 46, 231)   # Red violation for unverified rider
                    label_text = f"{track_prefix}UNVERIFIED RIDER ({conf:.2f})"
                else:
                    color = (180, 180, 180)
                    label_text = f"{track_prefix}{cls_name} ({conf:.2f})"

            x1, y1, x2, y2 = box

            thickness = max(2, int(w / 400))
            cv2.rectangle(annotated, (x1, y1), (x2, y2), color, thickness)

            if show_labels:
                font = cv2.FONT_HERSHEY_SIMPLEX
                font_scale = 0.5
                font_thick = 1
                (tw, th), baseline = cv2.getTextSize(label_text, font, font_scale, font_thick)

                label_y1 = max(0, y1 - th - 6)
                label_y2 = y1
                label_x1 = x1
                label_x2 = min(w, x1 + tw + 8)

                cv2.rectangle(annotated, (label_x1, label_y1), (label_x2, label_y2), color, -1)
                cv2.putText(
                    annotated,
                    label_text,
                    (label_x1 + 4, label_y2 - 4),
                    font,
                    font_scale,
                    (255, 255, 255),
                    font_thick,
                    cv2.LINE_AA
                )

        banner_height = 42
        overlay = annotated.copy()
        cv2.rectangle(overlay, (0, 0), (w, banner_height), (20, 24, 33), -1)
        cv2.addWeighted(overlay, 0.75, annotated, 0.25, 0, annotated)

        font = cv2.FONT_HERSHEY_SIMPLEX
        total_riders = with_helmet_count + without_helmet_count
        comp_rate = (with_helmet_count / total_riders * 100) if total_riders > 0 else 100.0

        hud_text = f"Riders: {total_riders}  |  Compliant: {with_helmet_count}  |  Violations: {without_helmet_count}"
        cv2.putText(annotated, hud_text, (15, 28), font, 0.6, (255, 255, 255), 2, cv2.LINE_AA)

        if without_helmet_count > 0:
            badge_color = (46, 46, 231)
            status_text = f"ALERT: {without_helmet_count} VIOLATIONS"
        else:
            badge_color = (46, 204, 113)
            status_text = "ALL COMPLIANT"

        (bw, bh), _ = cv2.getTextSize(status_text, font, 0.55, 2)
        cv2.rectangle(annotated, (w - bw - 30, 8), (w - 10, 34), badge_color, -1)
        cv2.putText(annotated, status_text, (w - bw - 20, 26), font, 0.55, (255, 255, 255), 2, cv2.LINE_AA)

        if show_fps is not None:
            fps_str = f"FPS: {show_fps:.1f}"
            cv2.putText(annotated, fps_str, (15, h - 15), font, 0.55, (0, 255, 255), 2, cv2.LINE_AA)

        return annotated

    def process_frame(
        self,
        frame: np.ndarray,
        frame_id: int = 0,
        logger: Optional[ViolationLogger] = None,
        camera_id: str = "CAM-01"
    ) -> Tuple[np.ndarray, List[Dict[str, Any]], Optional[Dict[str, Any]]]:
        """
        Process a single video frame: run detection, annotate, and log if requested.

        Returns:
            (annotated_frame, detections_list, logged_record)
        """
        t0 = time.time()
        detections = self.detect(frame)
        latency = time.time() - t0
        fps = 1.0 / latency if latency > 0 else 30.0

        annotated = self.draw_annotations(frame, detections, show_fps=fps)

        record = None
        if logger is not None:
            record = logger.log_detection(
                frame_id=frame_id,
                detections=detections,
                camera_id=camera_id
            )

        return annotated, detections, record

    def process_video_stream(
        self,
        source: Union[str, int],
        output_path: Optional[str] = None,
        logger: Optional[ViolationLogger] = None,
        camera_id: str = "CAM-01",
        max_frames: Optional[int] = None
    ):
        """
        Process a video file or webcam stream.

        Args:
            source: Video file path or integer webcam index (e.g. 0).
            output_path: Optional path to save annotated MP4/AVI.
            logger: Optional ViolationLogger instance.
            camera_id: Camera perspective name.
            max_frames: Max frames to process before stopping.
        """
        cap = cv2.VideoCapture(source)
        if not cap.isOpened():
            print(f"[ERROR] Unable to open video source: {source}")
            return

        w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = cap.get(cv2.CAP_PROP_FPS) or 25.0

        writer = None
        if output_path:
            fourcc = cv2.VideoWriter_fourcc(*"mp4v")
            writer = cv2.VideoWriter(output_path, fourcc, fps, (w, h))

        frame_idx = 0
        try:
            while True:
                ret, frame = cap.read()
                if not ret:
                    break

                frame_idx += 1
                annotated, detections, _ = self.process_frame(
                    frame=frame,
                    frame_id=frame_idx,
                    logger=logger,
                    camera_id=camera_id
                )

                if writer:
                    writer.write(annotated)

                if max_frames and frame_idx >= max_frames:
                    break

        finally:
            cap.release()
            if writer:
                writer.release()
            print(f"[INFO] Processed {frame_idx} frames.")


if __name__ == "__main__":
    print("Testing HelmetDetector inference pipeline...")
    detector = HelmetDetector(conf_threshold=0.30)
    print("Model initialized:", detector.model_path)

    dummy_frame = np.zeros((480, 640, 3), dtype=np.uint8)
    dummy_frame[:] = (70, 70, 70)

    mock_detections = [
        {"class_id": 0, "class_name": "With Helmet", "confidence": 0.92, "box": [50, 80, 180, 240]},
        {"class_id": 1, "class_name": "Without Helmet", "confidence": 0.88, "box": [300, 100, 430, 260]}
    ]

    annotated = detector.draw_annotations(dummy_frame, mock_detections, show_fps=28.5)
    print("Annotation executed successfully. Frame shape:", annotated.shape)

    test_logger = ViolationLogger()
    _, _, record = detector.process_frame(dummy_frame, frame_id=1, logger=test_logger)
    print("Logged event:", record)
