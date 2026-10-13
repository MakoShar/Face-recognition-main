"""
Face Tracker — Frame-to-Frame Face Association.

Simple IoU-based tracker that associates detected faces across frames
to stabilize identity labels and reduce redundant recognition calls.

This tracker does NOT replace identity verification — it only provides
temporal consistency between recognition runs.
"""

import time
import numpy as np
import logging
from collections import OrderedDict

import config

logger = logging.getLogger(__name__)


class TrackedFace:
    """Represents a tracked face across frames with multi-frame observation fusion and state confirmation."""

    def __init__(self, track_id, bbox, name="Unknown", similarity=0.0, confidence=0.0, pose="None", window_size=6):
        self.track_id = track_id
        self.bbox = bbox
        self.name = name
        self.similarity = float(similarity)
        self.confidence = float(confidence)
        self.pose = pose
        self.disappeared_count = 0
        self.frames_since_recognition = 0
        self.total_frames = 1
        self.window_size = window_size
        self.observations = []  # list of recent frame observation dicts

        # State confirmation tracking (Section 5)
        self.raw_name = name
        self.confirmed_name = "Unknown"
        self.is_confirmed = False
        self.candidate_name = None
        self.consistent_observations = 0
        self.contradiction_count = 0

        # Trajectory & motion direction
        cx = (bbox[0] + bbox[2]) * 0.5
        cy = (bbox[1] + bbox[3]) * 0.5
        self.centroid_history = [(cx, cy, time.time())]
        self.direction = "Stationary"

        # Best-quality observation retained across entire track lifecycle
        self.best_quality_score = 0.0
        self.best_observation = None  # {name, confidence, similarity, pose, quality, timestamp}

        # Attendance tracking per individual in recognition zone
        self.attendance_logged = False
        self.attendance_event_id = None
        self.start_time = time.time()

    def update_bbox(self, new_bbox):
        """Update track bounding box and record trajectory centroid."""
        self.bbox = new_bbox
        cx = (new_bbox[0] + new_bbox[2]) * 0.5
        cy = (new_bbox[1] + new_bbox[3]) * 0.5
        self.centroid_history.append((cx, cy, time.time()))
        if len(self.centroid_history) > 30:
            self.centroid_history.pop(0)

        # Calculate motion direction along vertical axis
        if len(self.centroid_history) >= 3:
            first_cy = self.centroid_history[0][1]
            last_cy = self.centroid_history[-1][1]
            dy = last_cy - first_cy
            if dy > 20.0:
                self.direction = "Entry"
            elif dy < -20.0:
                self.direction = "Exit"
            else:
                self.direction = "Stationary"

    def add_observation(self, name, similarity, confidence, pose="None", quality=100.0):
        """
        Record a frame observation, validate stability, detect contradictions,
        and combine observations across the temporal window.
        """
        self.frames_since_recognition = 0
        self.raw_name = name
        now = time.time()

        # Frame Redundancy Check:
        # Do not treat adjacent frames as independent proof when image position and timing are virtually identical
        is_redundant = False
        if self.observations:
            last_obs = self.observations[-1]
            time_delta = now - last_obs.get('timestamp', 0.0)
            if time_delta < 0.05 and len(self.centroid_history) >= 2:
                c_now = self.centroid_history[-1]
                c_prev = self.centroid_history[-2]
                c_dist = np.hypot(c_now[0] - c_prev[0], c_now[1] - c_prev[1])
                if c_dist < 4.0:
                    is_redundant = True

        min_conf_obs = getattr(config, 'TRACK_CONFIRMATION_MIN_OBSERVATIONS', 3)
        max_contr = getattr(config, 'TRACK_MAX_CONTRADICTIONS', 1)

        composite_quality = float(quality) * max(0.5, float(confidence) / 100.0)

        obs = {
            'name': name,
            'confidence': float(confidence),
            'similarity': float(similarity),
            'pose': pose,
            'quality': float(quality),
            'quality_score': float(quality),
            'composite_quality': float(composite_quality),
            'timestamp': now
        }
        self.observations.append(obs)
        if len(self.observations) > self.window_size:
            self.observations.pop(0)

        # Retain best-quality observation across the individual's track lifecycle
        if composite_quality > self.best_quality_score:
            self.best_quality_score = composite_quality
            self.best_observation = dict(obs)

        # ─── Confirmation & Contradiction Logic (Section 5) ───
        if name != "Unknown" and confidence >= getattr(config, 'CONFIDENCE_THRESHOLD_YELLOW', 60.0):
            if self.candidate_name is None:
                self.candidate_name = name
                self.consistent_observations = 1 if not is_redundant else 0
                self.contradiction_count = 0
            elif self.candidate_name == name:
                if not is_redundant:
                    self.consistent_observations += 1
                self.contradiction_count = 0
                if self.consistent_observations >= min_conf_obs:
                    self.is_confirmed = True
                    self.confirmed_name = name
            else:
                # Contradiction: confident evidence for a competing identity
                self.contradiction_count += 1
                logger.info(
                    f"[Tracker] Track #{self.track_id} contradiction: candidate '{self.candidate_name}' "
                    f"vs new '{name}' (contradictions={self.contradiction_count})"
                )
                if self.contradiction_count > max_contr:
                    # Revoke confirmed identity immediately!
                    logger.warning(
                        f"[Tracker] Revoking match on track #{self.track_id} "
                        f"('{self.confirmed_name}') due to persistent contradictions"
                    )
                    self.is_confirmed = False
                    self.confirmed_name = "Unknown"
                    self.candidate_name = name
                    self.consistent_observations = 1
                    self.contradiction_count = 0
                    self.observations = [obs]
                    self.attendance_logged = False
        else:
            # Observation was Unknown or low confidence
            if self.candidate_name is not None and confidence < 40.0:
                self.contradiction_count += 1
                if self.contradiction_count > max_contr * 2:
                    self.is_confirmed = False
                    self.confirmed_name = "Unknown"
                    self.candidate_name = None
                    self.consistent_observations = 0

        self._combine_observations()

        # Enforce confirmed naming
        if self.is_confirmed:
            self.name = self.confirmed_name
        elif self.candidate_name is not None and self.consistent_observations >= 1:
            self.name = self.candidate_name
        else:
            self.name = "Unknown"

        return self.name, self.confidence, self.similarity, self.pose

    def _combine_observations(self):
        """
        Combine observations across frames:
        1. Weighted voting across the temporal window
        2. Recency-weighted smoothing of confidence and similarity
        """
        if not self.observations:
            return

        # Tally weighted votes per identity
        weights = {}
        for obs in self.observations:
            id_name = obs['name']
            w = max(1.0, obs['confidence'])
            weights[id_name] = weights.get(id_name, 0.0) + w

        winning_name = max(weights.keys(), key=lambda k: weights[k])

        # Calculate recency-weighted confidence and similarity for winning identity
        win_obs = [o for o in self.observations if o['name'] == winning_name]
        exp_weights = [1.25 ** i for i in range(len(win_obs))]
        sum_w = sum(exp_weights)

        smoothed_conf = sum(o['confidence'] * w for o, w in zip(win_obs, exp_weights)) / sum_w
        smoothed_sim = sum(o['similarity'] * w for o, w in zip(win_obs, exp_weights)) / sum_w

        self.confidence = float(smoothed_conf)
        self.similarity = float(smoothed_sim)
        self.pose = win_obs[-1].get('pose', 'Front')


class FaceTracker:
    """
    Simple IoU-based face tracker for temporal consistency.

    Associates detected faces across frames using bounding box IoU.
    Maintains identity labels between recognition runs to reduce
    flickering and redundant embedding computation.

    This tracker does NOT replace identity verification.
    """

    def __init__(self, iou_threshold=0.3, max_disappeared=5,
                 recognition_interval=1):
        """
        Args:
            iou_threshold: Minimum IoU to associate a detection with a track
            max_disappeared: Frames before dropping a lost track
            recognition_interval: Re-run recognition every N frames per track
        """
        self.iou_threshold = iou_threshold
        self.max_disappeared = max_disappeared
        self.recognition_interval = recognition_interval

        self.tracks = OrderedDict()
        self.next_track_id = 0

    def _compute_iou(self, box_a, box_b):
        """Compute IoU between two boxes [x1, y1, x2, y2]."""
        x1 = max(box_a[0], box_b[0])
        y1 = max(box_a[1], box_b[1])
        x2 = min(box_a[2], box_b[2])
        y2 = min(box_a[3], box_b[3])

        inter_area = max(0, x2 - x1) * max(0, y2 - y1)
        area_a = (box_a[2] - box_a[0]) * (box_a[3] - box_a[1])
        area_b = (box_b[2] - box_b[0]) * (box_b[3] - box_b[1])
        union_area = area_a + area_b - inter_area

        if union_area <= 0:
            return 0.0
        return inter_area / union_area

    def _compute_match_score(self, box_a, box_b, disappeared_count=0):
        """
        Compute association score combining IoU and normalized centroid proximity.
        Ensures continuous tracking as individuals move through the recognition zone.
        """
        iou = self._compute_iou(box_a, box_b)
        if iou >= self.iou_threshold:
            return 1.0 + iou

        # Centroid distance fallback: ONLY apply if track was visible in the immediate last frame (disappeared_count == 0)
        # If the track was lost, require actual IoU overlap to avoid latching onto a new person!
        if disappeared_count == 0:
            cx_a = (box_a[0] + box_a[2]) * 0.5
            cy_a = (box_a[1] + box_a[3]) * 0.5
            cx_b = (box_b[0] + box_b[2]) * 0.5
            cy_b = (box_b[1] + box_b[3]) * 0.5

            w_a = max(1.0, box_a[2] - box_a[0])
            h_a = max(1.0, box_a[3] - box_a[1])
            diag = np.sqrt(w_a * w_a + h_a * h_a)

            c_dist = np.sqrt((cx_a - cx_b) ** 2 + (cy_a - cy_b) ** 2) / diag
            if c_dist < 0.40:
                return 0.5 + (0.40 - c_dist)

        return 0.0

    def update(self, detections):
        """
        Update tracks with new detections.

        Args:
            detections: List of dicts with 'bbox' key [x1, y1, x2, y2]

        Returns:
            assignments: List of (detection_index, track_id, needs_recognition) tuples
                - detection_index: Index into the detections list
                - track_id: Associated track ID
                - needs_recognition: Whether this face needs fresh recognition
        """
        current_bboxes = [d['bbox'] for d in detections]

        # If no existing tracks, register all detections as new tracks
        if not self.tracks:
            assignments = []
            for i, bbox in enumerate(current_bboxes):
                track_id = self._register(bbox)
                assignments.append((i, track_id, True))
            return assignments

        # If no detections, mark all tracks as disappeared
        if not current_bboxes:
            disappeared_ids = []
            for track_id, track in self.tracks.items():
                track.disappeared_count += 1
                if track.disappeared_count >= self.max_disappeared:
                    disappeared_ids.append(track_id)
            for tid in disappeared_ids:
                self._deregister(tid)
            return []

        # Compute match score matrix between existing tracks and new detections
        track_ids = list(self.tracks.keys())
        track_bboxes = [self.tracks[tid].bbox for tid in track_ids]

        num_tracks = len(track_bboxes)
        num_dets = len(current_bboxes)

        # Greedy assignment: match highest score pairs first
        pairs = []
        for t in range(num_tracks):
            tid = track_ids[t]
            disp = self.tracks[tid].disappeared_count
            for d in range(num_dets):
                score = self._compute_match_score(track_bboxes[t], current_bboxes[d], disappeared_count=disp)
                if score > 0.0:
                    pairs.append((score, t, d))

        pairs.sort(key=lambda x: x[0], reverse=True)

        assignments = []
        matched_tracks = set()
        matched_dets = set()

        for iou_val, t_idx, d_idx in pairs:
            if t_idx in matched_tracks or d_idx in matched_dets:
                continue

            track_id = track_ids[t_idx]
            track = self.tracks[track_id]

            # Update track
            track.update_bbox(current_bboxes[d_idx])
            track.disappeared_count = 0
            track.frames_since_recognition += 1
            track.total_frames += 1

            # Determine if recognition is needed
            needs_recognition = (
                track.frames_since_recognition >= self.recognition_interval
                or track.name == "Unknown"
            )

            assignments.append((d_idx, track_id, needs_recognition))
            matched_tracks.add(t_idx)
            matched_dets.add(d_idx)

        # Handle unmatched tracks (disappeared)
        disappeared_ids = []
        for t_idx in range(num_tracks):
            if t_idx not in matched_tracks:
                track_id = track_ids[t_idx]
                track = self.tracks[track_id]
                track.disappeared_count += 1
                if track.disappeared_count >= self.max_disappeared:
                    disappeared_ids.append(track_id)

        for tid in disappeared_ids:
            self._deregister(tid)

        # Handle unmatched detections (new faces)
        for d_idx in range(num_dets):
            if d_idx not in matched_dets:
                track_id = self._register(current_bboxes[d_idx])
                assignments.append((d_idx, track_id, True))

        return assignments

    def update_identity(self, track_id, name, similarity, confidence=0.0, pose="None", quality=100.0):
        """
        Record a new observation and combine observations across frames for this track.

        Returns:
            (combined_name, combined_conf, combined_sim, combined_pose)
        """
        if track_id in self.tracks:
            return self.tracks[track_id].add_observation(name, similarity, confidence, pose, quality=quality)
        return name, confidence, similarity, pose

    def get_track(self, track_id):
        """Get a tracked face by ID."""
        return self.tracks.get(track_id)

    def _register(self, bbox):
        """Register a new track."""
        track_id = self.next_track_id
        self.tracks[track_id] = TrackedFace(track_id, bbox)
        self.next_track_id += 1
        return track_id

    def _deregister(self, track_id):
        """Remove a track."""
        if track_id in self.tracks:
            del self.tracks[track_id]

    def reset(self):
        """Clear all tracks."""
        self.tracks.clear()
        self.next_track_id = 0
