#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
YOLOv11 Detector for Volleyball Hawkeye System
"""

import cv2
import numpy as np
import logging
from typing import List, Dict, Optional, Tuple
from pathlib import Path
import torch

try:
    from ultralytics import YOLO
    YOLO_AVAILABLE = True
except ImportError:
    YOLO_AVAILABLE = False
    logging.warning("Ultralytics not available. Install with: pip install ultralytics")

class YOLOv11Detector:
    """
    YOLOv11-specific detector
    """
    def __init__(self, model_path: str = "yolo11n.pt", confidence_threshold: float = 0.5, device: str = "auto"):
        self.logger = logging.getLogger(__name__)
        self.model_path = model_path
        self.confidence_threshold = confidence_threshold
        self.device = device
        if not YOLO_AVAILABLE:
            raise ImportError("Ultralytics not available. Install with: pip install ultralytics")
        self.model = None
        self._load_model()

    def _load_model(self):
        try:
            self.model = YOLO(self.model_path)
            if hasattr(self.model, 'fuse'):
                self.model.fuse()
            self.logger.info(f"Loaded YOLOv11 model: {self.model_path}")
        except Exception as e:
            self.logger.error(f"Failed to load YOLOv11 model: {str(e)}")
            raise

    def detect_ball(self, frame: np.ndarray, camera_id: str = "") -> List[np.ndarray]:
        if self.model is None:
            self.logger.error("Model not loaded")
            return []
        try:
            results = self.model(frame, conf=self.confidence_threshold, verbose=False)
            detections = []
            for result in results:
                boxes = result.boxes
                if boxes is not None:
                    for box in boxes:
                        x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                        confidence = box.conf[0].cpu().numpy()
                        class_id = int(box.cls[0].cpu().numpy())
                        if self._is_ball_class(class_id):
                            center_x = (x1 + x2) / 2
                            center_y = (y1 + y2) / 2
                            detections.append(np.array([center_x, center_y, confidence, class_id]))
            if detections:
                self.logger.debug(f"Camera {camera_id}: Detected {len(detections)} balls")
            return detections
        except Exception as e:
            self.logger.error(f"Ball detection failed: {str(e)}")
            return []

    def _is_ball_class(self, class_id: int) -> bool:
        ball_classes = {
            32,  # sports ball
            37,  # tennis ball
            38,  # baseball
            39,  # softball
            40,  # basketball
            41,  # volleyball
            42,  # soccer ball
            43,  # football
            44,  # bowling ball
        }
        return class_id in ball_classes

    def detect_volleyball_specific(self, frame: np.ndarray, camera_id: str = "") -> List[np.ndarray]:
        yolo_detections = self.detect_ball(frame, camera_id)
        volleyball_detections = []
        for detection in yolo_detections:
            x, y, confidence, class_id = detection
            if self._is_in_court_area(x, y, frame.shape):
                volleyball_detections.append(detection)
        return volleyball_detections

    def _is_in_court_area(self, x: float, y: float, frame_shape: Tuple[int, int, int]) -> bool:
        height, width = frame_shape[:2]
        court_x_min = width * 0.1
        court_x_max = width * 0.9
        court_y_min = height * 0.1
        court_y_max = height * 0.9
        return (court_x_min <= x <= court_x_max and court_y_min <= y <= court_y_max)

    def get_model_info(self) -> Dict:
        if self.model is None:
            return {"error": "Model not loaded"}
        try:
            info = self.model.info()
            return {
                "model_path": self.model_path,
                "layers": info[0] if len(info) > 0 else "Unknown",
                "parameters": info[1] if len(info) > 1 else "Unknown",
                "gradients": info[2] if len(info) > 2 else "Unknown",
                "gflops": info[3] if len(info) > 3 else "Unknown"
            }
        except Exception as e:
            return {"error": str(e)}

    def update_model(self, model_path: str):
        try:
            self.model_path = model_path
            self._load_model()
            self.logger.info(f"Updated to YOLOv11 model: {model_path}")
        except Exception as e:
            self.logger.error(f"Failed to update model: {str(e)}")
            raise

    def get_available_models(self) -> List[str]:
        return [
            "yolo11n.pt", "yolo11s.pt", "yolo11m.pt", "yolo11l.pt", "yolo11x.pt"
        ] 