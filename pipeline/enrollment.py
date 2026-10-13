"""
Pose-Aware Video Enrollment Module.

Decodes video, samples frames, detects faces with BlazeFace,
classifies head pose (Front, Left, Right, Up, Down), applies quality filters,
removes near-duplicates, extracts embeddings, and updates the gallery.
"""

import cv2
import numpy as np
import logging
import os

from pipeline.preprocessor import FacePreprocessor

logger = logging.getLogger(__name__)

REQUIRED_POSES = ["Front", "Left", "Right", "Up", "Down"]


class VideoEnrollmentManager:
    """
    Manages five-angle face enrollment from recorded video.
    """

    def __init__(self,
                 target_samples_per_pose=3,
                 dedup_similarity_threshold=0.92,
                 frame_sample_step=3):
        """
        Args:
            target_samples_per_pose: Desired number of reference images per pose
            dedup_similarity_threshold: Min cosine similarity to consider duplicate
            frame_sample_step: Decode every Nth frame from the video
        """
        self.target_samples_per_pose = target_samples_per_pose
        self.dedup_similarity_threshold = dedup_similarity_threshold
        self.frame_sample_step = frame_sample_step

    def enroll_from_video(self,
                           video_path,
                           identity_name,
                           gallery,
                           detector,
                           embedder,
                           pose_estimator,
                           quality_assessor,
                           preprocessor=None):
        """
        Process an enrollment video and register reference embeddings.

        Args:
            video_path: Path to video file
            identity_name: Person's name
            gallery: FaceGallery instance
            detector: BlazeFaceDetector instance
            embedder: FaceEmbedder instance
            pose_estimator: HeadPoseEstimator instance
            quality_assessor: FaceQualityAssessor instance
            preprocessor: FacePreprocessor instance (optional)

        Returns:
            report: Dictionary summarizing enrollment results
        """
        if preprocessor is None:
            preprocessor = FacePreprocessor()
        if not os.path.exists(video_path):
            return {
                "success": False,
                "error": f"Video file not found: {video_path}"
            }

        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            return {
                "success": False,
                "error": f"Failed to open video: {video_path}"
            }

        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = cap.get(cv2.CAP_PROP_FPS) or 30.0

        # Storage for candidate frames per pose
        # {pose: list of (sharpness, crop, embedding)}
        pose_candidates = {pose: [] for pose in REQUIRED_POSES}

        stats = {
            "total_frames": total_frames,
            "processed_frames": 0,
            "rejected_no_face": 0,
            "rejected_multi_face": 0,
            "rejected_low_quality": 0,
            "rejected_duplicate": 0,
        }

        frame_idx = 0
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            frame_idx += 1
            if frame_idx % self.frame_sample_step != 0:
                continue

            stats["processed_frames"] += 1

            # 1. Detect faces
            detections = detector.detect(frame)
            if not detections:
                stats["rejected_no_face"] += 1
                continue

            # Reject if multiple plausible faces (to prevent identity contamination)
            if len(detections) > 1:
                # If secondary face has significant score, reject
                if detections[1]["score"] > 0.40:
                    stats["rejected_multi_face"] += 1
                    continue

            best_det = detections[0]
            bbox = best_det["bbox"]
            score = best_det["score"]
            keypoints = best_det["keypoints"]

            # 2. Quality assessment
            is_valid, sharpness, reason = quality_assessor.assess_crop(frame, bbox, score)
            if not is_valid:
                stats["rejected_low_quality"] += 1
                continue

            # 3. Pose estimation
            x1, y1, x2, y2 = [int(v) for v in bbox]
            h, w = frame.shape[:2]
            raw_crop = frame[max(0, y1):min(h, y2), max(0, x1):min(w, x2)]
            pose, metrics = pose_estimator.estimate_pose(keypoints=keypoints, bbox=bbox, face_crop=raw_crop)
            if pose not in REQUIRED_POSES:
                continue

            # Extract aligned crop
            if preprocessor is not None:
                crop = preprocessor.align_face(frame, keypoints, bbox)
            else:
                x1, y1, x2, y2 = [int(v) for v in bbox]
                h, w = frame.shape[:2]
                crop = frame[max(0, y1):min(h, y2), max(0, x1):min(w, x2)]

            if crop is None or crop.size == 0:
                continue

            # 4. Extract embedding
            emb = embedder.get_embedding(crop)
            if emb is None:
                continue

            # 5. Deduplication check within this pose category
            is_dup = False
            for existing_sharpness, _, existing_emb in pose_candidates[pose]:
                sim = float(np.dot(emb, existing_emb))
                if sim >= self.dedup_similarity_threshold:
                    is_dup = True
                    break

            if is_dup:
                stats["rejected_duplicate"] += 1
                continue

            # Add candidate
            pose_candidates[pose].append((sharpness, crop, emb))

        cap.release()

        # 6. Select top candidates per pose and enroll
        enrolled_summary = {}
        missing_poses = []
        total_enrolled = 0

        for pose in REQUIRED_POSES:
            candidates = pose_candidates[pose]
            # Sort by sharpness descending
            candidates.sort(key=lambda c: c[0], reverse=True)
            top_candidates = candidates[:self.target_samples_per_pose]

            count = 0
            for sharpness, crop, emb in top_candidates:
                success = gallery.enroll(identity_name, emb, pose=pose, quality_score=sharpness)
                if success:
                    count += 1
                    total_enrolled += 1

            enrolled_summary[pose] = count
            if count == 0:
                missing_poses.append(pose)

        status = "Complete" if len(missing_poses) == 0 else ("Partial" if total_enrolled > 0 else "Failed")

        report = {
            "success": total_enrolled > 0,
            "status": status,
            "identity": identity_name,
            "total_enrolled_references": total_enrolled,
            "enrolled_per_pose": enrolled_summary,
            "missing_poses": missing_poses,
            "stats": stats
        }

        logger.info(
            f"Video enrollment for '{identity_name}': status={status}, "
            f"enrolled={total_enrolled}, missing={missing_poses}"
        )
        return report
