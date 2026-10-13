"""
BlazeFace TFLite Face Detector.

Wraps the BlazeFace Full Range Sparse model for multi-face detection.
Handles anchor generation, output decoding, NMS, and bounding box extraction.
"""

import threading
import numpy as np
import cv2
import logging

logger = logging.getLogger(__name__)


class BlazeFaceDetector:
    """
    Face detector using the BlazeFace Full Range TFLite model.

    The model outputs 2304 anchor predictions, each with:
      - 4 bounding box regressors (center_x, center_y, width, height)
      - 12 keypoint coordinates (6 keypoints × 2)
      - 1 confidence score

    This class handles:
      - Anchor grid generation
      - Input preprocessing (resize + normalize)
      - Output decoding (de-anchor bounding boxes)
      - Non-Maximum Suppression (NMS)
    """

    def __init__(self, model_path, confidence_threshold=0.65, nms_iou_threshold=0.3):
        """
        Args:
            model_path: Path to blaze_face_full_range_sparse.tflite
            confidence_threshold: Minimum face score to keep
            nms_iou_threshold: IoU threshold for NMS
        """
        self.confidence_threshold = confidence_threshold
        self.nms_iou_threshold = nms_iou_threshold
        self.input_size = 192
        self._lock = threading.Lock()

        # Load TFLite model cleanly (suppressing numpy 1.x vs 2.x stderr warnings)
        try:
            import sys, os, warnings
            warnings.filterwarnings('ignore')
            devnull = os.open(os.devnull, os.O_WRONLY)
            saved_stderr = os.dup(sys.stderr.fileno())
            os.dup2(devnull, sys.stderr.fileno())
            try:
                from tensorflow.lite.python.interpreter import Interpreter
            finally:
                os.dup2(saved_stderr, sys.stderr.fileno())
                os.close(saved_stderr)
                os.close(devnull)
            self.interpreter = Interpreter(model_path=model_path)
        except Exception:
            try:
                import tflite_runtime.interpreter as tflite
                self.interpreter = tflite.Interpreter(model_path=model_path)
            except Exception:
                import tensorflow as tf
                self.interpreter = tf.lite.Interpreter(model_path=model_path)

        self.interpreter.allocate_tensors()
        self.input_details = self.interpreter.get_input_details()
        self.output_details = self.interpreter.get_output_details()

        # Generate anchors for the BlazeFace Full Range model
        self.anchors = self._generate_anchors()

        logger.info(
            f"BlazeFace loaded: input={self.input_details[0]['shape']}, "
            f"anchors={len(self.anchors)}, "
            f"confidence_threshold={self.confidence_threshold}"
        )

    def _generate_anchors(self):
        """
        Generate SSD-style anchors for BlazeFace Full Range.

        The Full Range model uses these strides and anchor configurations:
        - Strides: [4, 4, 4, 4, 4, 4, 4, 4] with layers producing
          48x48, 24x24, 12x12, etc. grids.

        The total should sum to 2304 anchors matching the model output.
        """
        # BlazeFace Full Range anchor configuration
        # Based on MediaPipe's face detection full range config
        strides = [4]
        # The full range model uses a simpler anchor scheme:
        # 48x48 grid with 1 anchor per position = 2304 anchors
        anchors = []
        input_size = self.input_size

        # For the full-range variant, all 2304 anchors come from a single
        # 48×48 feature map with stride 4
        grid_size = input_size // strides[0]  # 192/4 = 48
        for y in range(grid_size):
            for x in range(grid_size):
                # Each anchor is at center of grid cell, normalized to [0, 1]
                cx = (x + 0.5) / grid_size
                cy = (y + 0.5) / grid_size
                anchors.append([cx, cy])

        anchors = np.array(anchors, dtype=np.float32)

        # If we don't have exactly 2304, we need to replicate
        # Full range model: 48×48 = 2304 with 1 anchor per cell
        if len(anchors) != 2304:
            logger.warning(
                f"Anchor count mismatch: generated {len(anchors)}, expected 2304. "
                "Adjusting anchor generation."
            )
            anchors = self._generate_anchors_multi_stride()

        return anchors

    def _generate_anchors_multi_stride(self):
        """
        Fallback multi-stride anchor generation for BlazeFace Full Range.
        Matches the MediaPipe configuration for 2304 total anchors.
        """
        # MediaPipe full range face detection anchor config
        # Uses multiple feature map sizes with 1 anchor per position
        anchor_configs = [
            # (stride, num_anchors_per_position)
            (4, 1),   # 48×48 = 2304 anchors
        ]

        anchors = []
        input_size = self.input_size

        total = 0
        # Try single 48x48 grid first
        for stride, num_per_pos in anchor_configs:
            grid_size = input_size // stride
            for y in range(grid_size):
                for x in range(grid_size):
                    for _ in range(num_per_pos):
                        cx = (x + 0.5) / grid_size
                        cy = (y + 0.5) / grid_size
                        anchors.append([cx, cy])
                        total += 1

        return np.array(anchors, dtype=np.float32)

    def _preprocess(self, frame):
        """
        Preprocess a BGR frame for BlazeFace input.

        Args:
            frame: BGR image (H, W, 3) uint8

        Returns:
            input_tensor: (1, 192, 192, 3) float32, normalized to [-1, 1]
            scale_x: horizontal scale factor for mapping back to original
            scale_y: vertical scale factor for mapping back to original
            pad_x: horizontal padding offset
            pad_y: vertical padding offset
        """
        h, w = frame.shape[:2]

        # Convert BGR to RGB
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Letterbox resize to maintain aspect ratio
        scale = min(self.input_size / w, self.input_size / h)
        new_w = int(w * scale)
        new_h = int(h * scale)
        resized = cv2.resize(rgb, (new_w, new_h), interpolation=cv2.INTER_LINEAR)

        # Create padded image
        padded = np.full((self.input_size, self.input_size, 3), 128, dtype=np.uint8)
        pad_x = (self.input_size - new_w) // 2
        pad_y = (self.input_size - new_h) // 2
        padded[pad_y:pad_y + new_h, pad_x:pad_x + new_w] = resized

        # Normalize to [-1, 1]
        input_tensor = (padded.astype(np.float32) / 127.5) - 1.0
        input_tensor = np.expand_dims(input_tensor, axis=0)

        return input_tensor, scale, pad_x, pad_y, w, h

    def _decode_boxes(self, raw_boxes, anchors=None):
        """
        Decode raw bounding box predictions using anchors.

        Args:
            raw_boxes: (N, 16) raw regressors
            anchors: Optional (N, 2) anchor centers. If None, uses self.anchors.

        Returns:
            boxes: (N, 4) decoded boxes as [x1, y1, x2, y2] normalized
            keypoints: (N, 6, 2) decoded keypoint coordinates
        """
        use_anchors = self.anchors if anchors is None else anchors

        # Extract box center offsets and size
        cx = raw_boxes[:, 0] / self.input_size + use_anchors[:, 0]
        cy = raw_boxes[:, 1] / self.input_size + use_anchors[:, 1]
        bw = raw_boxes[:, 2] / self.input_size
        bh = raw_boxes[:, 3] / self.input_size

        # Convert to corner format [x1, y1, x2, y2]
        x1 = cx - bw / 2
        y1 = cy - bh / 2
        x2 = cx + bw / 2
        y2 = cy + bh / 2

        boxes = np.stack([x1, y1, x2, y2], axis=-1)

        # Decode keypoints (6 keypoints × 2 coords = 12 values)
        keypoints = np.zeros((len(raw_boxes), 6, 2), dtype=np.float32)
        for k in range(6):
            kp_offset = 4 + k * 2
            keypoints[:, k, 0] = raw_boxes[:, kp_offset] / self.input_size + use_anchors[:, 0]
            keypoints[:, k, 1] = raw_boxes[:, kp_offset + 1] / self.input_size + use_anchors[:, 1]

        return boxes, keypoints

    def _nms(self, boxes, scores):
        """
        Weighted Non-Maximum Suppression.

        Args:
            boxes: (N, 4) boxes in [x1, y1, x2, y2] format
            scores: (N,) confidence scores

        Returns:
            indices: List of kept detection indices
        """
        if len(boxes) == 0:
            return []

        # Convert to pixel coordinates for NMS
        x1 = boxes[:, 0]
        y1 = boxes[:, 1]
        x2 = boxes[:, 2]
        y2 = boxes[:, 3]
        areas = (x2 - x1) * (y2 - y1)

        # Sort by score descending
        order = scores.argsort()[::-1]
        keep = []

        while len(order) > 0:
            i = order[0]
            keep.append(i)

            if len(order) == 1:
                break

            # Compute IoU with remaining boxes
            xx1 = np.maximum(x1[i], x1[order[1:]])
            yy1 = np.maximum(y1[i], y1[order[1:]])
            xx2 = np.minimum(x2[i], x2[order[1:]])
            yy2 = np.minimum(y2[i], y2[order[1:]])

            inter_w = np.maximum(0.0, xx2 - xx1)
            inter_h = np.maximum(0.0, yy2 - yy1)
            intersection = inter_w * inter_h
            union = areas[i] + areas[order[1:]] - intersection
            iou = intersection / (union + 1e-6)

            # Keep boxes with IoU below threshold
            remaining = np.where(iou <= self.nms_iou_threshold)[0]
            order = order[remaining + 1]

        return keep

    def detect(self, frame):
        """
        Detect faces in a BGR frame.

        Args:
            frame: BGR image (H, W, 3) uint8

        Returns:
            detections: List of dicts, each containing:
                - 'bbox': [x1, y1, x2, y2] in pixel coordinates
                - 'score': confidence score
                - 'keypoints': [(x, y), ...] 6 keypoints in pixel coords
        """
        if frame is None or frame.size == 0:
            return []

        # Preprocess
        input_tensor, scale, pad_x, pad_y, orig_w, orig_h = self._preprocess(frame)

        # Run thread-safe inference
        with self._lock:
            self.interpreter.set_tensor(self.input_details[0]['index'], input_tensor)
            self.interpreter.invoke()
            raw_boxes = self.interpreter.get_tensor(self.output_details[0]['index'])[0].copy()   # (2304, 16)
            raw_scores = self.interpreter.get_tensor(self.output_details[1]['index'])[0].copy()  # (2304, 1)

        # Apply sigmoid to scores (clipped to avoid numerical overflow)
        raw_scores_clamped = np.clip(raw_scores.squeeze(), -20.0, 20.0)
        scores = 1.0 / (1.0 + np.exp(-raw_scores_clamped))

        # Filter by confidence
        mask = scores >= self.confidence_threshold
        if not np.any(mask):
            return []

        filtered_boxes_raw = raw_boxes[mask]
        filtered_scores = scores[mask]

        # Decode boxes using matched anchors without mutating self.anchors
        filtered_anchors = self.anchors[mask]
        decoded_boxes, decoded_keypoints = self._decode_boxes(filtered_boxes_raw, anchors=filtered_anchors)

        # NMS
        keep_indices = self._nms(decoded_boxes, filtered_scores)

        # Convert to pixel coordinates and build results
        detections = []
        for idx in keep_indices:
            box = decoded_boxes[idx]
            score = float(filtered_scores[idx])
            kps = decoded_keypoints[idx]

            # Map from normalized [0, 1] in padded space to original pixel coords
            x1 = (box[0] * self.input_size - pad_x) / scale
            y1 = (box[1] * self.input_size - pad_y) / scale
            x2 = (box[2] * self.input_size - pad_x) / scale
            y2 = (box[3] * self.input_size - pad_y) / scale

            # Clamp to image bounds
            x1 = max(0, min(orig_w, x1))
            y1 = max(0, min(orig_h, y1))
            x2 = max(0, min(orig_w, x2))
            y2 = max(0, min(orig_h, y2))

            # Skip degenerate boxes
            if (x2 - x1) < 2 or (y2 - y1) < 2:
                continue

            # Convert keypoints to pixel coords
            pixel_kps = []
            for k in range(6):
                kx = (kps[k, 0] * self.input_size - pad_x) / scale
                ky = (kps[k, 1] * self.input_size - pad_y) / scale
                pixel_kps.append((float(kx), float(ky)))

            detections.append({
                'bbox': [float(x1), float(y1), float(x2), float(y2)],
                'score': score,
                'keypoints': pixel_kps,
            })

        logger.debug(f"Detected {len(detections)} faces (from {np.sum(mask)} candidates)")
        return detections
