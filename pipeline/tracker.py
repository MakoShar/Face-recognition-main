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

logger = logging.getLogger(__name__)


class TrackedFace:
    """Represents a tracked face across frames with multi-frame observation fusion."""

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

        # Best-quality observation retained across entire track lifecycle
        self.best_quality_score = 0.0
        self.best_observation = None  # {name, confidence, similarity, pose, quality, timestamp}

        # Attendance tracking per individual in recognition zone
        self.attendance_logged = False
        self.attendance_event_id = None
        self.start_time = time.time()

    def add_observation(self, name, similarity, confidence, pose="None", quality=100.0):
        """
        Record a frame observation, retain best-quality observations,
        and combine observations across the temporal window.
        """
        self.frames_since_recognition = 0
        composite_quality = float(quality) * max(0.5, float(confidence) / 100.0)
        obs = {
            'name': name,
            'confidence': float(confidence),
            'similarity': float(similarity),
            'pose': pose,
            'quality': float(quality),
            'quality_score': float(quality),
            'composite_quality': float(composite_quality),
            'timestamp': time.time()
        }
        self.observations.append(obs)
        if len(self.observations) > self.window_size:
            self.observations.pop(0)

        # Retain best-quality observation across the individual's track lifecycle
        if composite_quality > self.best_quality_score:
            self.best_quality_score = composite_quality
            self.best_observation = dict(obs)

        self._combine_observations()
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

        self.name = winning_name
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

    def __init__(self, iou_threshold=0.3, max_disappeared=15,
                 recognition_interval=10):
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

    def _compute_match_score(self, box_a, box_b):
        """
        Compute association score combining IoU and normalized centroid proximity.
        Ensures continuous tracking as individuals move through the recognition zone.
        """
        iou = self._compute_iou(box_a, box_b)
        if iou >= self.iou_threshold:
            return 1.0 + iou

        # Centroid distance fallback
        cx_a = (box_a[0] + box_a[2]) * 0.5
        cy_a = (box_a[1] + box_a[3]) * 0.5
        cx_b = (box_b[0] + box_b[2]) * 0.5
        cy_b = (box_b[1] + box_b[3]) * 0.5

        w_a = max(1.0, box_a[2] - box_a[0])
        h_a = max(1.0, box_a[3] - box_a[1])
        diag = np.sqrt(w_a * w_a + h_a * h_a)

        c_dist = np.sqrt((cx_a - cx_b) ** 2 + (cy_a - cy_b) ** 2) / diag
        if c_dist < 0.60:
            return 0.5 + (0.60 - c_dist)

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
                if track.disappeared_count > self.max_disappeared:
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
            for d in range(num_dets):
                score = self._compute_match_score(track_bboxes[t], current_bboxes[d])
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
            track.bbox = current_bboxes[d_idx]
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
                if track.disappeared_count > self.max_disappeared:
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
