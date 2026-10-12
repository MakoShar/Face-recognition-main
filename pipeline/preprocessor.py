"""
Face Preprocessing Module.

Handles cropping detected faces from frames, validating crop quality,
and preparing face images for the embedding model.
"""

import numpy as np
import cv2
import logging

logger = logging.getLogger(__name__)


class FacePreprocessor:
    """
    Preprocesses detected face regions for the embedding model.

    Responsibilities:
      - Crop face from frame using bounding box (with margin)
      - Validate crop quality (minimum size, not mostly padding)
      - Resize to embedding model input size
      - Normalize pixel values for the embedding model
      - Convert color space as needed
    """

    def __init__(self, target_size=112, margin_ratio=0.2, min_face_size=20):
        """
        Args:
            target_size: Output crop size (square) for embedding model
            margin_ratio: Extra margin around face bbox (fraction of face size)
            min_face_size: Minimum acceptable face dimension in pixels
        """
        self.target_size = target_size
        self.margin_ratio = margin_ratio
        self.min_face_size = min_face_size

    def extract_face(self, frame, bbox):
        """
        Extract and preprocess a single face crop from a frame.

        Args:
            frame: BGR image (H, W, 3) uint8
            bbox: [x1, y1, x2, y2] in pixel coordinates

        Returns:
            face_tensor: (1, 3, 112, 112) float32 normalized for embedding model,
                         or None if the crop is invalid
            face_crop: The raw BGR crop (for display purposes), or None
        """
        if frame is None or frame.size == 0:
            return None, None

        h, w = frame.shape[:2]
        x1, y1, x2, y2 = bbox

        # Validate bounding box
        face_w = x2 - x1
        face_h = y2 - y1

        if face_w < self.min_face_size or face_h < self.min_face_size:
            logger.debug(f"Face too small: {face_w:.0f}x{face_h:.0f}")
            return None, None

        # Add margin around the face
        margin_x = face_w * self.margin_ratio
        margin_y = face_h * self.margin_ratio

        x1_m = int(max(0, x1 - margin_x))
        y1_m = int(max(0, y1 - margin_y))
        x2_m = int(min(w, x2 + margin_x))
        y2_m = int(min(h, y2 + margin_y))

        # Crop
        face_crop = frame[y1_m:y2_m, x1_m:x2_m]

        if face_crop.size == 0:
            logger.debug("Empty face crop after clipping")
            return None, None

        # Prepare for embedding model
        face_tensor = self._normalize_for_embedding(face_crop)

        return face_tensor, face_crop

    def _normalize_for_embedding(self, face_bgr):
        """
        Normalize a face crop for the MobileFaceNet embedding model.

        MobileFaceNet expects:
          - Input: (1, 3, 112, 112) float32
          - Channel order: RGB
          - Normalization: (pixel / 255.0 - 0.5) / 0.5 = pixel / 127.5 - 1.0

        Args:
            face_bgr: BGR image (H, W, 3) uint8

        Returns:
            tensor: (1, 3, 112, 112) float32
        """
        # Convert BGR to RGB
        face_rgb = cv2.cvtColor(face_bgr, cv2.COLOR_BGR2RGB)

        # Resize to target size
        face_resized = cv2.resize(
            face_rgb,
            (self.target_size, self.target_size),
            interpolation=cv2.INTER_LINEAR
        )

        # Normalize to [-1, 1]
        face_float = face_resized.astype(np.float32) / 127.5 - 1.0

        # Transpose from (H, W, C) to (C, H, W) — ONNX expects NCHW
        face_transposed = np.transpose(face_float, (2, 0, 1))

        # Add batch dimension
        face_tensor = np.expand_dims(face_transposed, axis=0)

        return face_tensor

    def extract_faces_batch(self, frame, bboxes):
        """
        Extract multiple face crops from a single frame.

        Args:
            frame: BGR image (H, W, 3) uint8
            bboxes: List of [x1, y1, x2, y2] bounding boxes

        Returns:
            valid_tensors: List of (1, 3, 112, 112) tensors
            valid_crops: List of raw BGR crops
            valid_indices: List of original indices (for mapping back)
        """
        valid_tensors = []
        valid_crops = []
        valid_indices = []

        for i, bbox in enumerate(bboxes):
            tensor, crop = self.extract_face(frame, bbox)
            if tensor is not None:
                valid_tensors.append(tensor)
                valid_crops.append(crop)
                valid_indices.append(i)

        return valid_tensors, valid_crops, valid_indices
