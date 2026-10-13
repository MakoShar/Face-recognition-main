"""
Head Pose Estimator and Face Quality Assessment Module.

Uses geometric relationships between BlazeFace's 6 facial keypoints
(right eye, left eye, nose tip, mouth center, right ear tragion, left ear tragion)
to estimate head pose (Front, Left, Right, Up, Down) and assesses face quality.
"""

import cv2
import math
import numpy as np
import logging
import os

logger = logging.getLogger(__name__)

# Pose categories required by the specification
POSE_CATEGORIES = ["Front", "Left", "Right", "Up", "Down"]


class HeadPoseEstimator:
    """
    Estimates 3D-relative head pose from 2D facial keypoints or ResNet-50 6D rotation matrix.

    Keypoints provided by BlazeFace (in image pixel space):
      0: Right Eye (viewer left)
      1: Left Eye (viewer right)
      2: Nose Tip
      3: Mouth Center
      4: Right Ear Tragion (viewer left)
      5: Left Ear Tragion (viewer right)
    """

    def __init__(self,
                 model_path=None,
                 yaw_threshold=0.32,
                 pitch_threshold=0.28,
                 max_roll_threshold=60.0,
                 yaw_threshold_deg=20.0,
                 pitch_threshold_deg=15.0):
        """
        Args:
            model_path: Optional path to ResNet-50 6D rotation matrix ONNX model
            yaw_threshold: geometric threshold for Left/Right classification
            pitch_threshold: geometric threshold for Up/Down classification
            max_roll_threshold: maximum allowable in-plane tilt angle in degrees
            yaw_threshold_deg: neural threshold in degrees for Left/Right
            pitch_threshold_deg: neural threshold in degrees for Up/Down
        """
        self.model_path = model_path
        self.yaw_threshold = yaw_threshold
        self.pitch_threshold = pitch_threshold
        self.max_roll_threshold = max_roll_threshold
        self.yaw_threshold_deg = yaw_threshold_deg
        self.pitch_threshold_deg = pitch_threshold_deg
        self.session = None

        if model_path and os.path.exists(model_path):
            try:
                import onnxruntime as ort
                sess_opts = ort.SessionOptions()
                sess_opts.intra_op_num_threads = 2
                sess_opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
                providers = ['CUDAExecutionProvider', 'CPUExecutionProvider']
                self.session = ort.InferenceSession(model_path, sess_options=sess_opts, providers=providers)
                logger.info(f"HeadPoseEstimator initialized with ResNet-50 model from {model_path} ({self.session.get_providers()})")
            except Exception as e:
                logger.warning(f"Could not load ONNX head pose model from {model_path}: {e}")

    def estimate_pose(self, keypoints=None, bbox=None, frame=None, face_crop=None):
        """
        Classify head pose into Front, Left, Right, Up, Down.

        Args:
            keypoints: Optional list of 6 (x, y) tuples in pixel space.
            bbox: Optional [x1, y1, x2, y2] bounding box.
            frame: Optional full frame BGR image.
            face_crop: Optional cropped face BGR image.

        Returns:
            pose: One of "Front", "Left", "Right", "Up", "Down", or "Uncertain"
            metrics: Dict with computed ratios and angles
        """
        # 1. Neural 3D Head Pose estimation (ResNet-50 6D rotation matrix)
        if self.session is not None:
            crop = face_crop
            if crop is None and frame is not None and bbox is not None:
                h, w = frame.shape[:2]
                x1, y1, x2, y2 = [int(v) for v in bbox]
                crop = frame[max(0, y1):min(h, y2), max(0, x1):min(w, x2)]

            if crop is not None and crop.size > 0:
                try:
                    face_resized = cv2.resize(crop, (224, 224), interpolation=cv2.INTER_LINEAR)
                    face_rgb = cv2.cvtColor(face_resized, cv2.COLOR_BGR2RGB)
                    mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
                    std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
                    norm = (face_rgb.astype(np.float32) / 255.0 - mean) / std
                    tensor = np.transpose(norm, (2, 0, 1))[np.newaxis, ...].astype(np.float32)
                    out = self.session.run(None, {"input": tensor})
                    R = out[0][0]

                    sy = math.sqrt(R[0, 0] * R[0, 0] + R[1, 0] * R[1, 0])
                    if sy > 1e-6:
                        pitch = math.degrees(math.atan2(R[2, 1], R[2, 2]))
                        yaw = math.degrees(math.atan2(-R[2, 0], sy))
                        roll = math.degrees(math.atan2(R[1, 0], R[0, 0]))
                    else:
                        pitch = math.degrees(math.atan2(-R[1, 2], R[1, 1]))
                        yaw = math.degrees(math.atan2(-R[2, 0], sy))
                        roll = 0.0

                    # 5-class categorization
                    if abs(yaw) >= self.yaw_threshold_deg:
                        pose = "Left" if yaw < 0 else "Right"
                    elif abs(pitch) >= self.pitch_threshold_deg:
                        pose = "Up" if pitch < 0 else "Down"
                    else:
                        pose = "Front"

                    metrics = {
                        "pitch_deg": pitch,
                        "yaw_deg": yaw,
                        "roll_deg": roll,
                        "method": "neural_resnet50"
                    }
                    return pose, metrics
                except Exception as e:
                    logger.debug(f"Neural pose estimation failed, falling back to geometric: {e}")

        # 2. Geometric fallback
        return self._estimate_pose_geometric(keypoints, bbox)

    def _estimate_pose_geometric(self, keypoints, bbox=None):
        """Geometric fallback estimation from 2D keypoints."""
        if not keypoints or len(keypoints) < 6:
            return "Uncertain", {"error": "Insufficient keypoints"}

        re = np.array(keypoints[0], dtype=np.float32)     # Right eye
        le = np.array(keypoints[1], dtype=np.float32)     # Left eye
        nose = np.array(keypoints[2], dtype=np.float32)   # Nose
        mouth = np.array(keypoints[3], dtype=np.float32)  # Mouth
        rear = np.array(keypoints[4], dtype=np.float32)   # Right ear
        lear = np.array(keypoints[5], dtype=np.float32)   # Left ear

        # 1. Roll (eye tilt angle in degrees)
        dx_eyes = le[0] - re[0]
        dy_eyes = le[1] - re[1]
        roll_deg = float(np.degrees(np.arctan2(dy_eyes, dx_eyes)))

        # 2. Yaw (horizontal asymmetry)
        dx_right_eye = nose[0] - re[0]
        dx_left_eye = le[0] - nose[0]
        denom_yaw = abs(dx_right_eye) + abs(dx_left_eye) + 1e-6
        yaw_ratio = float((dx_right_eye - dx_left_eye) / denom_yaw)

        dx_rear = re[0] - rear[0]
        dx_lear = lear[0] - le[0]
        denom_ear = abs(dx_rear) + abs(dx_lear) + 1e-6
        ear_ratio = float((dx_rear - dx_lear) / denom_ear)

        yaw_score = 0.7 * yaw_ratio + 0.3 * ear_ratio

        # 3. Pitch (vertical asymmetry)
        eye_mid_y = (re[1] + le[1]) / 2.0
        dy_eye_nose = nose[1] - eye_mid_y
        dy_nose_mouth = mouth[1] - nose[1]
        denom_pitch = abs(dy_eye_nose) + abs(dy_nose_mouth) + 1e-6
        pitch_score = float((dy_eye_nose - dy_nose_mouth) / denom_pitch)

        metrics = {
            "yaw_score": yaw_score,
            "yaw_ratio": yaw_ratio,
            "ear_ratio": ear_ratio,
            "pitch_score": pitch_score,
            "roll_deg": roll_deg,
            "method": "geometric_landmarks"
        }

        if abs(roll_deg) > self.max_roll_threshold:
            return "Uncertain", metrics

        if abs(yaw_score) >= self.yaw_threshold:
            pose = "Left" if yaw_score < 0 else "Right"
        elif abs(pitch_score) >= self.pitch_threshold:
            pose = "Up" if pitch_score < 0 else "Down"
        else:
            pose = "Front"

        return pose, metrics


class FaceQualityAssessor:
    """
    Evaluates face image quality for enrollment filtering.
    """

    def __init__(self,
                 min_face_size=50,
                 min_blur_score=60.0,
                 min_detection_score=0.45):
        """
        Args:
            min_face_size: Minimum width and height of face bounding box
            min_blur_score: Minimum Laplacian variance (sharpness score)
            min_detection_score: Minimum BlazeFace detection confidence
        """
        self.min_face_size = min_face_size
        self.min_blur_score = min_blur_score
        self.min_detection_score = min_detection_score

    def assess_crop(self, frame, bbox, det_score):
        """
        Check if a face detection meets quality requirements for enrollment.

        Args:
            frame: BGR frame
            bbox: [x1, y1, x2, y2] in pixel coords
            det_score: Confidence score from detector

        Returns:
            is_valid: bool
            blur_score: float (Laplacian variance)
            reason: str (None if valid, rejection reason otherwise)
        """
        if det_score < self.min_detection_score:
            return False, 0.0, f"Detection score too low: {det_score:.2f} < {self.min_detection_score:.2f}"

        x1, y1, x2, y2 = [int(v) for v in bbox]
        h, w = frame.shape[:2]

        # Clamp
        x1 = max(0, min(w - 1, x1))
        y1 = max(0, min(h - 1, y1))
        x2 = max(0, min(w, x2))
        y2 = max(0, min(h, y2))

        fw = x2 - x1
        fh = y2 - y1

        if fw < self.min_face_size or fh < self.min_face_size:
            return False, 0.0, f"Face resolution too small: {fw}x{fh} < {self.min_face_size}px"

        crop = frame[y1:y2, x1:x2]
        if crop.size == 0:
            return False, 0.0, "Empty crop"

        # Laplacian variance for blur/sharpness
        gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
        blur_score = float(cv2.Laplacian(gray, cv2.CV_64F).var())

        if blur_score < self.min_blur_score:
            return False, blur_score, f"Blurry frame: sharpness {blur_score:.1f} < {self.min_blur_score:.1f}"

        return True, blur_score, None
