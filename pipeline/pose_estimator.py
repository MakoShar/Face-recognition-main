"""
Head Pose Estimator and Face Quality Assessment Module.

Uses geometric relationships between BlazeFace's 6 facial keypoints
(right eye, left eye, nose tip, mouth center, right ear tragion, left ear tragion)
to estimate head pose (Front, Left, Right, Up, Down) and assesses face quality.
"""

import cv2
import numpy as np
import logging

logger = logging.getLogger(__name__)

# Pose categories required by the specification
POSE_CATEGORIES = ["Front", "Left", "Right", "Up", "Down"]


class HeadPoseEstimator:
    """
    Estimates 3D-relative head pose from 2D facial keypoints.

    Keypoints provided by BlazeFace (in image pixel space):
      0: Right Eye (viewer left)
      1: Left Eye (viewer right)
      2: Nose Tip
      3: Mouth Center
      4: Right Ear Tragion (viewer left)
      5: Left Ear Tragion (viewer right)
    """

    def __init__(self,
                 yaw_threshold=0.32,
                 pitch_threshold=0.28,
                 max_roll_threshold=60.0):
        """
        Args:
            yaw_threshold: threshold for Left/Right classification
            pitch_threshold: threshold for Up/Down classification
            max_roll_threshold: maximum allowable in-plane tilt angle in degrees
        """
        self.yaw_threshold = yaw_threshold
        self.pitch_threshold = pitch_threshold
        self.max_roll_threshold = max_roll_threshold

    def estimate_pose(self, keypoints, bbox=None):
        """
        Classify head pose into Front, Left, Right, Up, Down.

        Args:
            keypoints: List of 6 (x, y) tuples in pixel space.
            bbox: Optional [x1, y1, x2, y2] bounding box.

        Returns:
            pose: One of "Front", "Left", "Right", "Up", "Down", or "Uncertain"
            metrics: Dict with computed ratios and angles
        """
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
        # Distance from nose to right eye vs nose to left eye
        dx_right_eye = nose[0] - re[0]
        dx_left_eye = le[0] - nose[0]
        denom_yaw = abs(dx_right_eye) + abs(dx_left_eye) + 1e-6
        yaw_ratio = float((dx_right_eye - dx_left_eye) / denom_yaw)

        # Distance from ears to eyes provides additional lateral support
        dx_rear = re[0] - rear[0]
        dx_lear = lear[0] - le[0]
        denom_ear = abs(dx_rear) + abs(dx_lear) + 1e-6
        ear_ratio = float((dx_rear - dx_lear) / denom_ear)

        # Combined yaw score (primary eye-nose with ear confirmation)
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
        }

        # Check excessive roll
        if abs(roll_deg) > self.max_roll_threshold:
            return "Uncertain", metrics

        # Pose classification
        # Left vs Right has priority if lateral rotation is significant
        if abs(yaw_score) >= self.yaw_threshold:
            # Note: in mirror/camera view, negative yaw_score means subject facing Left (viewer right/subject left)
            pose = "Left" if yaw_score < 0 else "Right"
        elif abs(pitch_score) >= self.pitch_threshold:
            # Negative pitch_score indicates nose is closer to eye line (looking Up)
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
