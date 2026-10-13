"""
Face Preprocessing Module.

Handles cropping detected faces from frames, validating crop quality,
and preparing face images for the embedding model.
"""

import numpy as np
import cv2
import logging

logger = logging.getLogger(__name__)

# Standard canonical coordinates for 112x112 ArcFace / GlintR100 / AdaFace alignment
CANONICAL_POINTS_112 = np.array([
    [38.2946, 51.6963],   # Right eye (viewer left)
    [73.5318, 51.6963],   # Left eye (viewer right)
    [56.0252, 71.7366],   # Nose tip
    [56.1396, 92.3655],   # Mouth center
], dtype=np.float32)


class FacePreprocessor:
    """
    Preprocesses detected face regions for the embedding model.

    Responsibilities:
      - Align face to canonical coordinates using landmark similarity transform
      - Crop face from frame using bounding box (with margin) as fallback
      - Validate crop quality (minimum size, not mostly padding)
      - Resize to embedding model input size
      - Normalize pixel values for the embedding model
      - Convert color space as needed
    """

    def __init__(self, target_size=112, margin_ratio=0.2, min_face_size=50):
        """
        Args:
            target_size: Output crop size (square) for embedding model
            margin_ratio: Extra margin around face bbox (fraction of face size)
            min_face_size: Minimum acceptable face dimension in pixels (default: 50)
        """
        self.target_size = target_size
        self.margin_ratio = margin_ratio
        self.min_face_size = min_face_size

    def align_face(self, frame, keypoints=None, bbox=None, target_size=None):
        """
        Extract an aligned face crop using similarity transformation.

        Uses BlazeFace's keypoints (right eye, left eye, nose tip, mouth center)
        mapped to canonical ArcFace 112x112 reference landmarks via
        cv2.estimateAffinePartial2D (scale, rotation, translation).

        Falls back to 2-point eye rotation if partial keypoints,
        and finally to bounding box crop if keypoints are unavailable.

        Args:
            frame: BGR image (H, W, 3) uint8
            keypoints: List of 6 (x, y) keypoints from BlazeFace
            bbox: [x1, y1, x2, y2] bounding box
            target_size: Output square size (default: self.target_size)

        Returns:
            aligned_bgr: (target_size, target_size, 3) BGR uint8 aligned image, or None
        """
        if frame is None or frame.size == 0:
            return None

        if target_size is None:
            target_size = self.target_size

        # Validate bounding box coordinates and resolution before alignment
        if bbox is not None:
            x1, y1, x2, y2 = bbox
            if not (np.isfinite(x1) and np.isfinite(y1) and np.isfinite(x2) and np.isfinite(y2)):
                return None
            face_w = x2 - x1
            face_h = y2 - y1
            if face_w < self.min_face_size or face_h < self.min_face_size:
                logger.debug(f"Face too small for alignment: {face_w:.0f}x{face_h:.0f} < {self.min_face_size}")
                return None

        h, w = frame.shape[:2]

        # 1. 4-point similarity transform (eyes, nose, mouth)
        if keypoints is not None and len(keypoints) >= 4:
            src_kps = np.array([keypoints[0], keypoints[1], keypoints[2], keypoints[3]], dtype=np.float32)
            if np.all(np.isfinite(src_kps)):
                scale_factor = target_size / 112.0
                dst_kps = CANONICAL_POINTS_112 * scale_factor
                M, _ = cv2.estimateAffinePartial2D(src_kps, dst_kps)
                if M is not None:
                    return cv2.warpAffine(
                        frame, M, (target_size, target_size),
                        flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REFLECT
                    )

        # 2. Fallback: 2-point eye alignment
        if keypoints is not None and len(keypoints) >= 2:
            re = np.array(keypoints[0], dtype=np.float32)
            le = np.array(keypoints[1], dtype=np.float32)
            if np.all(np.isfinite(re)) and np.all(np.isfinite(le)):
                dx = le[0] - re[0]
                dy = le[1] - re[1]
                dist = np.sqrt(dx * dx + dy * dy)
                if dist > 2.0:
                    scale_factor = target_size / 112.0
                    desired_dist = (73.5318 - 38.2946) * scale_factor
                    desired_center_x = 55.9132 * scale_factor
                    desired_center_y = 51.6963 * scale_factor

                    angle = np.degrees(np.arctan2(dy, dx))
                    scale = desired_dist / dist
                    center = ((re[0] + le[0]) * 0.5, (re[1] + le[1]) * 0.5)
                    M = cv2.getRotationMatrix2D(center, angle, scale)
                    M[0, 2] += (desired_center_x - center[0])
                    M[1, 2] += (desired_center_y - center[1])
                    return cv2.warpAffine(
                        frame, M, (target_size, target_size),
                        flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REFLECT
                    )

        # 3. Fallback: Bounding box crop
        if bbox is not None:
            x1, y1, x2, y2 = [int(v) for v in bbox]
            crop = frame[max(0, y1):min(h, y2), max(0, x1):min(w, x2)]
            if crop.size > 0:
                return cv2.resize(crop, (target_size, target_size), interpolation=cv2.INTER_LINEAR)

        return None

    def extract_face(self, frame, bbox, keypoints=None):
        """
        Extract and preprocess a single face crop from a frame.

        Args:
            frame: BGR image (H, W, 3) uint8
            bbox: [x1, y1, x2, y2] in pixel coordinates
            keypoints: Optional list of (x, y) keypoints for canonical alignment

        Returns:
            face_tensor: (1, 3, 112, 112) float32 normalized for embedding model,
                         or None if the crop is invalid
            face_crop: The raw or aligned BGR crop, or None
        """
        if frame is None or frame.size == 0 or bbox is None:
            return None, None

        # Validate bounding box coordinates and minimum resolution
        x1, y1, x2, y2 = bbox
        if not (np.isfinite(x1) and np.isfinite(y1) and np.isfinite(x2) and np.isfinite(y2)):
            return None, None

        face_w = x2 - x1
        face_h = y2 - y1
        if face_w < self.min_face_size or face_h < self.min_face_size:
            logger.debug(f"Face too small: {face_w:.0f}x{face_h:.0f} < {self.min_face_size}")
            return None, None

        if keypoints is not None:
            aligned = self.align_face(frame, keypoints, bbox)
            if aligned is not None and aligned.size > 0:
                tensor = self._normalize_for_embedding(aligned)
                return tensor, aligned

        h, w = frame.shape[:2]

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
