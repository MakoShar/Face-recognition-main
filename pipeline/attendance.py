"""
Attendance Event Manager with Duplicate Suppression.

Handles recording attendance events when verified individuals move through
the recognition zone, with multi-level duplicate suppression:
  1. In-Track Suppression: Emits at most ONE attendance event per track session.
  2. Cooldown Suppression: Suppresses duplicate events for the same individual within a cooldown window.
  3. Quality-Threshold Gating: Requires high confidence and minimum stable observations.
"""

import os
import time
import json
import random
import logging
import threading
from datetime import datetime

logger = logging.getLogger(__name__)


class AttendanceManager:
    """
    Manages confirmed attendance events and enforces duplicate suppression.
    """

    def __init__(self,
                 log_path=None,
                 min_confidence=85.0,
                 min_stable_frames=3,
                 cooldown_seconds=300):
        """
        Args:
            log_path: Path to persist attendance events JSON file
            min_confidence: Minimum calibrated confidence percentage
            min_stable_frames: Consecutive observations needed before logging
            cooldown_seconds: Cooldown period before the same person can log again
        """
        self.log_path = log_path
        self.min_confidence = min_confidence
        self.min_stable_frames = min_stable_frames
        self.cooldown_seconds = cooldown_seconds

        self._lock = threading.Lock()
        self.events = []
        self.logged_tracks = set()  # set of (camera_id, track_id) tuples logged in current session
        self.last_attendance_time = {}  # {(person_name, direction): timestamp}
        self.track_identities_history = {}  # {(camera_id, track_id): {name: count}}

        if self.log_path:
            self._load_from_disk()

    def process_observation(self,
                            track_id,
                            name,
                            confidence,
                            similarity=0.0,
                            pose="Front",
                            quality_score=100.0,
                            camera_id=0,
                            best_observation=None,
                            is_confirmed=True,
                            direction="Entry"):
        """
        Evaluate an observation for an attendance event with multi-camera isolation.

        Args:
            track_id: Integer track ID
            name: Recognized person name
            confidence: Calibrated confidence percentage
            similarity: Raw cosine similarity
            pose: Estimated head pose
            quality_score: Sharpness / quality metric
            camera_id: Camera identifier
            best_observation: Dict with best-quality observation details
            is_confirmed: Boolean whether the track's identity has reached confirmation threshold
            direction: Direction of crossing ("Entry", "Exit", or "Stationary")

        Returns:
            event: Dict if an attendance event was newly registered, None if suppressed.
        """
        # Reject unconfirmed, unknown, or uncertain identities
        if not is_confirmed or not name or name in ("Unknown", "Uncertain") or confidence < self.min_confidence:
            return None

        track_key = (camera_id, track_id)

        with self._lock:
            # 1. In-Track Suppression: already logged for this (camera_id, track_id) crossing
            if track_key in self.logged_tracks:
                return None

            # 2. Direction-Aware Cooldown Suppression: check if this person logged recently in this direction
            now = time.time()
            dir_key = (name, direction)
            last_time = self.last_attendance_time.get(dir_key, 0.0)
            if now - last_time < self.cooldown_seconds:
                # Still within cooldown window
                logger.debug(
                    f"Attendance suppressed for '{name}' [{direction}]: in cooldown "
                    f"({int(now - last_time)}s / {self.cooldown_seconds}s)"
                )
                self.logged_tracks.add(track_key)
                return None

            # 3. Stability check: count consecutive observations on this isolated camera track
            if track_key not in self.track_identities_history:
                self.track_identities_history[track_key] = {}
            history = self.track_identities_history[track_key]
            history[name] = history.get(name, 0) + 1

            if history[name] < self.min_stable_frames:
                # Not enough stable frames yet
                return None

            # ─── All gates passed: Register Attendance Event ──────────────────
            event_id = f"ATT_{int(now)}_{random.randint(1000, 9999)}"
            time_str = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

            best_qual = quality_score
            if best_observation and isinstance(best_observation, dict):
                best_qual = best_observation.get("quality", quality_score)

            event = {
                "event_id": event_id,
                "name": name,
                "confidence": round(float(confidence), 1),
                "similarity": round(float(similarity), 4),
                "pose": pose,
                "quality_score": round(float(best_qual), 1),
                "camera_id": camera_id,
                "track_id": track_id,
                "direction": direction,
                "timestamp": now,
                "time_str": time_str,
                "status": "CONFIRMED"
            }

            self.events.append(event)
            self.logged_tracks.add(track_key)
            self.last_attendance_time[dir_key] = now

            logger.info(
                f"✓ Attendance EVENT LOGGED: '{name}' ({confidence:.1f}%) [{direction}] | "
                f"Camera {camera_id} | Track #{track_id} | Time: {time_str}"
            )

            if self.log_path:
                self._save_to_disk()

            return event

    def is_track_logged(self, track_id, camera_id=0):
        """Check if a (camera_id, track_id) has already been recorded for attendance."""
        with self._lock:
            return (camera_id, track_id) in self.logged_tracks or track_id in self.logged_tracks

    def get_recent_events(self, limit=50):
        """Return the most recent attendance events."""
        with self._lock:
            return list(reversed(self.events[-limit:]))

    def get_summary(self):
        """Return attendance summary statistics."""
        with self._lock:
            unique_names = set(e["name"] for e in self.events)
            return {
                "total_events": len(self.events),
                "unique_individuals": len(unique_names),
                "attendees": list(unique_names),
                "last_event": self.events[-1] if self.events else None,
                "cooldown_seconds": self.cooldown_seconds
            }

    def clear(self):
        """Reset today's attendance logs."""
        with self._lock:
            self.events.clear()
            self.logged_tracks.clear()
            self.last_attendance_time.clear()
            self.track_identities_history.clear()
            if self.log_path and os.path.exists(self.log_path):
                try:
                    os.remove(self.log_path)
                except Exception:
                    pass
        logger.info("Attendance records reset.")
        return True

    def _save_to_disk(self):
        """Persist events to JSON."""
        try:
            os.makedirs(os.path.dirname(self.log_path), exist_ok=True)
            with open(self.log_path, 'w') as f:
                json.dump(self.events, f, indent=2)
        except Exception as e:
            logger.warning(f"Failed to persist attendance log: {e}")

    def _load_from_disk(self):
        """Load past events from JSON."""
        if not os.path.exists(self.log_path):
            return
        try:
            with open(self.log_path, 'r') as f:
                loaded = json.load(f)
            if isinstance(loaded, list):
                self.events = loaded
                now = time.time()
                for e in self.events:
                    name = e.get("name")
                    ts = e.get("timestamp", 0.0)
                    if name and ts > self.last_attendance_time.get(name, 0.0):
                        self.last_attendance_time[name] = ts
            logger.info(f"Loaded {len(self.events)} past attendance events from {self.log_path}")
        except Exception as e:
            logger.warning(f"Could not load attendance log from {self.log_path}: {e}")
