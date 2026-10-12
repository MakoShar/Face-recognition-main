"""
Camera Stream Module.

Manages camera capture with frame sampling in a background thread.
Supports multiple cameras with independent capture threads and
bounded queues to prevent stale frame accumulation.
"""

import cv2
import threading
import queue
import time
import logging

logger = logging.getLogger(__name__)


class CameraStream:
    """
    Threaded camera capture with frame sampling.

    Each camera runs in its own thread, capturing frames and placing
    every Nth frame into a bounded queue for processing.

    Features:
      - Background thread capture (doesn't block main thread)
      - Frame sampling (every Nth frame)
      - Bounded queue (drops stale frames)
      - Automatic reconnection on camera failure
      - Clean start/stop lifecycle
    """

    def __init__(self, camera_id=0, sample_interval=5, queue_maxsize=2,
                 reconnect_delay=3.0, max_reconnect_attempts=10):
        """
        Args:
            camera_id: Camera device ID (int) or video file path (str)
            sample_interval: Process every Nth frame
            queue_maxsize: Maximum frames in the processing queue
            reconnect_delay: Seconds between reconnection attempts
            max_reconnect_attempts: Maximum reconnection attempts before giving up
        """
        self.camera_id = camera_id
        self.sample_interval = sample_interval
        self.reconnect_delay = reconnect_delay
        self.max_reconnect_attempts = max_reconnect_attempts

        self.frame_queue = queue.Queue(maxsize=queue_maxsize)
        self.latest_frame = None
        self.latest_frame_lock = threading.Lock()

        self._capture = None
        self._thread = None
        self._running = False
        self._connected = False
        self._frame_count = 0
        self._fps = 0.0
        self._error_message = None

    @property
    def is_running(self):
        return self._running

    @property
    def is_connected(self):
        return self._connected

    @property
    def fps(self):
        return self._fps

    @property
    def error_message(self):
        return self._error_message

    def start(self):
        """Start the camera capture thread."""
        if self._running:
            logger.warning(f"Camera {self.camera_id} already running")
            return

        self._running = True
        self._error_message = None
        self._thread = threading.Thread(
            target=self._capture_loop,
            name=f"Camera-{self.camera_id}",
            daemon=True
        )
        self._thread.start()
        logger.info(f"Camera {self.camera_id} capture thread started")

    def stop(self):
        """Stop the camera capture thread."""
        self._running = False
        if self._thread is not None:
            self._thread.join(timeout=5.0)
            self._thread = None

        if self._capture is not None:
            self._capture.release()
            self._capture = None

        self._connected = False

        # Drain the queue
        while not self.frame_queue.empty():
            try:
                self.frame_queue.get_nowait()
            except queue.Empty:
                break

        logger.info(f"Camera {self.camera_id} stopped")

    def get_frame(self, timeout=0.1):
        """
        Get the next sampled frame from the queue.

        Args:
            timeout: Max seconds to wait for a frame

        Returns:
            frame: BGR image or None if no frame available
        """
        try:
            return self.frame_queue.get(timeout=timeout)
        except queue.Empty:
            return None

    def get_latest_frame(self):
        """
        Get the most recent frame (for display purposes).
        Non-blocking, returns None if no frame captured yet.
        """
        with self.latest_frame_lock:
            return self.latest_frame.copy() if self.latest_frame is not None else None

    def _capture_loop(self):
        """Main capture loop running in background thread."""
        reconnect_attempts = 0

        while self._running:
            # Try to connect
            if not self._connected:
                if reconnect_attempts >= self.max_reconnect_attempts:
                    self._error_message = (
                        f"Camera {self.camera_id}: max reconnection attempts "
                        f"({self.max_reconnect_attempts}) exceeded"
                    )
                    logger.error(self._error_message)
                    self._running = False
                    break

                success = self._connect()
                if not success:
                    reconnect_attempts += 1
                    self._error_message = (
                        f"Camera {self.camera_id}: connection failed "
                        f"(attempt {reconnect_attempts}/{self.max_reconnect_attempts})"
                    )
                    logger.warning(self._error_message)
                    time.sleep(self.reconnect_delay)
                    continue

                reconnect_attempts = 0
                self._error_message = None

            # Read frame
            ret, frame = self._capture.read()

            if not ret or frame is None:
                logger.warning(f"Camera {self.camera_id}: frame read failed")
                self._connected = False
                if self._capture is not None:
                    self._capture.release()
                    self._capture = None
                continue

            self._frame_count += 1

            # Always update latest frame (for display)
            with self.latest_frame_lock:
                self.latest_frame = frame

            # Only enqueue sampled frames (for processing)
            if self._frame_count % self.sample_interval == 1:
                # Non-blocking put: drop frame if queue is full
                try:
                    self.frame_queue.put_nowait(frame.copy())
                except queue.Full:
                    # Drop the oldest frame and put the new one
                    try:
                        self.frame_queue.get_nowait()
                    except queue.Empty:
                        pass
                    try:
                        self.frame_queue.put_nowait(frame.copy())
                    except queue.Full:
                        pass

        # Cleanup
        if self._capture is not None:
            self._capture.release()
            self._capture = None
        self._connected = False

    def _connect(self):
        """Attempt to open the camera."""
        try:
            self._capture = cv2.VideoCapture(self.camera_id)

            if not self._capture.isOpened():
                return False

            # Try to set camera properties
            self._capture.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
            self._capture.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
            self._capture.set(cv2.CAP_PROP_BUFFERSIZE, 1)

            # Read actual properties
            actual_w = int(self._capture.get(cv2.CAP_PROP_FRAME_WIDTH))
            actual_h = int(self._capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
            actual_fps = self._capture.get(cv2.CAP_PROP_FPS)

            self._connected = True
            self._frame_count = 0

            logger.info(
                f"Camera {self.camera_id} connected: "
                f"{actual_w}x{actual_h} @ {actual_fps:.1f} FPS"
            )
            return True

        except Exception as e:
            logger.error(f"Camera {self.camera_id} connection error: {e}")
            return False

    def get_status(self):
        """Get camera status info dict."""
        return {
            'camera_id': self.camera_id,
            'running': self._running,
            'connected': self._connected,
            'frame_count': self._frame_count,
            'queue_size': self.frame_queue.qsize(),
            'error': self._error_message,
        }
