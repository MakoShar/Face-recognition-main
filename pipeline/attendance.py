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
import cv2

import config

logger = logging.getLogger(__name__)


class AttendanceManager:
    """
    Manages confirmed attendance events and enforces duplicate suppression.
    """

    def __init__(self,
                 log_path=None,
                 min_confidence=85.0,
                 min_stable_frames=3,
                 cooldown_seconds=300,
                 logs_dir=None):
        """
        Args:
            log_path: Path to persist attendance events JSON file
            min_confidence: Minimum calibrated confidence percentage
            min_stable_frames: Consecutive observations needed before logging
            cooldown_seconds: Cooldown period before the same person can log again
            logs_dir: Directory to save screenshots of logged individuals
        """
        self.log_path = log_path
        self.min_confidence = min_confidence
        self.min_stable_frames = min_stable_frames
        self.cooldown_seconds = cooldown_seconds
        self.logs_dir = logs_dir or getattr(config, 'LOGS_DIR', os.path.join(config.BASE_DIR, "Log"))
        os.makedirs(self.logs_dir, exist_ok=True)

        self._lock = threading.Lock()
        self.events = []
        self.logged_tracks = set()  # set of (camera_id, track_id) tuples logged in current session
        self.last_attendance_time = {}  # {(person_name, direction): timestamp}
        self.track_identities_history = {}  # {(camera_id, track_id): {name: count}}

        if self.log_path:
            self._load_from_disk()

    def _save_screenshot(self, name, track_id, camera_id, confidence, direction, pose, frame=None, bbox=None, best_observation=None):
        """
        Save an annotated full-frame screenshot to the Log/ folder when attendance is recorded.
        """
        try:
            base_frame = None
            box = bbox
            if best_observation and isinstance(best_observation, dict):
                base_frame = best_observation.get('frame')
                if best_observation.get('bbox') is not None:
                    box = best_observation.get('bbox')

            if base_frame is None and frame is not None:
                base_frame = frame

            if base_frame is None:
                return None, None

            img = base_frame.copy()
            h, w = img.shape[:2]

            # Choose bounding box color
            color = config.COLOR_GREEN if confidence >= getattr(config, 'CONFIDENCE_THRESHOLD_GREEN', 90.0) else (
                config.COLOR_YELLOW if confidence >= getattr(config, 'CONFIDENCE_THRESHOLD_YELLOW', 60.0) else config.COLOR_UNKNOWN
            )

            if box is not None and len(box) >= 4:
                x1, y1, x2, y2 = [int(v) for v in box]
                x1, y1 = max(0, x1), max(0, y1)
                x2, y2 = min(w, x2), min(h, y2)
                cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)

                # Modern corner accents
                clen = min(20, max(5, (x2 - x1) // 5), max(5, (y2 - y1) // 5))
                cv2.line(img, (x1, y1), (x1 + clen, y1), color, 3)
                cv2.line(img, (x1, y1), (x1, y1 + clen), color, 3)
                cv2.line(img, (x2, y1), (x2 - clen, y1), color, 3)
                cv2.line(img, (x2, y1), (x2, y1 + clen), color, 3)
                cv2.line(img, (x1, y2), (x1 + clen, y2), color, 3)
                cv2.line(img, (x1, y2), (x1, y2 - clen), color, 3)
                cv2.line(img, (x2, y2), (x2 - clen, y2), color, 3)
                cv2.line(img, (x2, y2), (x2, y2 - clen), color, 3)

                # Label tag
                tag = f"[#{track_id}] {name} ({confidence:.0f}%)" if confidence >= 60.0 else f"[#{track_id}] {name}"
                font = cv2.FONT_HERSHEY_SIMPLEX
                (tw, th), _ = cv2.getTextSize(tag, font, 0.6, 2)
                lbl_y = max(y1 - 6, th + 8)
                cv2.rectangle(img, (x1, lbl_y - th - 6), (x1 + tw + 10, lbl_y), color, -1)
                cv2.putText(img, tag, (x1 + 5, lbl_y - 4), font, 0.6, (255, 255, 255), 2)

            # Metadata banner at top
            banner_h = 32
            overlay = img.copy()
            cv2.rectangle(overlay, (0, 0), (w, banner_h), (18, 18, 24), -1)
            cv2.addWeighted(overlay, 0.75, img, 0.25, 0, img)

            now_str = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            status_text = "CONFIRMED" if confidence >= 60.0 else "VISITOR"
            header_text = f"CAM {camera_id}  |  TRACK #{track_id}  |  {name} ({confidence:.1f}%)  |  {status_text}  |  {direction}  |  {now_str}"
            cv2.putText(img, header_text, (12, 21), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (220, 230, 242), 1, cv2.LINE_AA)

            # Generate filename and save
            slug_time = datetime.now().strftime("%Y%m%d_%H%M%S")
            safe_name = "".join(c if c.isalnum() or c in "-_" else "_" for c in name)
            filename = f"{slug_time}_{safe_name}_Track{track_id}.jpg"
            filepath = os.path.join(self.logs_dir, filename)

            cv2.imwrite(filepath, img, [cv2.IMWRITE_JPEG_QUALITY, 92])
            logger.info(f"📸 Screenshot saved to Log: {filename}")
            return filename, f"/Log/{filename}"
        except Exception as e:
            logger.error(f"Failed to save attendance screenshot: {e}")
            return None, None

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
                            direction="Entry",
                            frame=None,
                            bbox=None):
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
            frame: Raw camera frame for screenshot capture
            bbox: Bounding box coordinates [x1, y1, x2, y2]

        Returns:
            event: Dict if an attendance event was newly registered, None if suppressed.
        """
        # Check if this observation is an unregistered visitor
        is_unknown = (not name) or name == "Unknown" or name.startswith("Unknown-") or name == "Uncertain"

        if is_unknown:
            if not getattr(config, 'LOG_UNKNOWN_ATTENDANCE', True):
                return None
            status = "UNREGISTERED_VISITOR"
            # Format name to Unknown-001, Unknown-002, etc.
            if not name or name in ("Unknown", "Uncertain"):
                prefix = getattr(config, 'UNKNOWN_LABEL_PREFIX', 'Unknown-')
                name = f"{prefix}{track_id + 1:03d}"
            # Require minimum stable frames for unknown visitor to filter transient noise
            min_stable = getattr(config, 'UNKNOWN_MIN_STABLE_FRAMES', self.min_stable_frames)
        else:
            # Enrolled employee: requires confirmed identity and min_confidence threshold
            if not is_confirmed or confidence < self.min_confidence:
                return None
            status = "CONFIRMED"
            min_stable = self.min_stable_frames

        track_key = (camera_id, track_id)

        with self._lock:
            # 1. In-Track Suppression & Upgrade Logic:
            if track_key in self.logged_tracks:
                # If this track was previously recorded as an UNREGISTERED_VISITOR,
                # but has now achieved CONFIRMED employee status (e.g. Anoop Verma),
                # UPGRADE the event in-place so the real employee is correctly logged!
                if status == "CONFIRMED":
                    for ev in self.events:
                        if (ev.get("camera_id"), ev.get("track_id")) == track_key and ev.get("status") == "UNREGISTERED_VISITOR":
                            logger.info(
                                f"✓ UPGRADING attendance event for Track #{track_id}: "
                                f"'{ev.get('name')}' (Visitor) -> '{name}' ({confidence:.1f}% Confirmed Employee)"
                            )
                            # Save upgraded screenshot for the confirmed identity
                            upgraded_file, upgraded_url = self._save_screenshot(
                                name=name,
                                track_id=track_id,
                                camera_id=camera_id,
                                confidence=confidence,
                                direction=direction,
                                pose=pose,
                                frame=frame,
                                bbox=bbox,
                                best_observation=best_observation
                            )
                            if upgraded_file:
                                ev["screenshot"] = upgraded_file
                                ev["screenshot_url"] = upgraded_url

                            ev["name"] = name
                            ev["confidence"] = round(float(confidence), 1)
                            ev["similarity"] = round(float(similarity), 4)
                            ev["pose"] = pose
                            ev["status"] = "CONFIRMED"
                            ev["time_str"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                            ev["timestamp"] = time.time()
                            self.last_attendance_time[(name, direction)] = ev["timestamp"]
                            if self.log_path:
                                self._save_to_disk()
                            return ev
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

            if history[name] < min_stable:
                # Not enough stable frames yet
                return None

            # ─── All gates passed: Register Attendance Event ──────────────────
            event_id = f"ATT_{int(now)}_{random.randint(1000, 9999)}"
            time_str = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

            best_qual = quality_score
            if best_observation and isinstance(best_observation, dict):
                best_qual = best_observation.get("quality", quality_score)

            # Save screenshot into the Log folder
            screenshot_file, screenshot_url = self._save_screenshot(
                name=name,
                track_id=track_id,
                camera_id=camera_id,
                confidence=confidence,
                direction=direction,
                pose=pose,
                frame=frame,
                bbox=bbox,
                best_observation=best_observation
            )

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
                "status": status,
                "screenshot": screenshot_file,
                "screenshot_url": screenshot_url
            }

            self.events.append(event)
            self.logged_tracks.add(track_key)
            self.last_attendance_time[dir_key] = now

            logger.info(
                f"✓ Attendance EVENT LOGGED: '{name}' [{status}] ({confidence:.1f}%) [{direction}] | "
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
