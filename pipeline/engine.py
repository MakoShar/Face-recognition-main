"""
Recognition Engine — Multi-Camera Orchestrator with Pose Awareness & Section 16 Rules.

Orchestrates the complete face recognition pipeline:
  Camera → Frame Sampler → BlazeFace Detection → Pose Estimation →
  Embedder (GlintR100 or AdaFace IR-101) → Multi-Pose Gallery Matching →
  Section 16 Calibrated Display Annotation → Flask Web Stream

Handles multiple independent cameras concurrently with bounded queues.
"""

import cv2
import time
import threading
import numpy as np
import logging

import config
from pipeline.detector import BlazeFaceDetector
from pipeline.preprocessor import FacePreprocessor
from pipeline.embedder import FaceEmbedder
from pipeline.gallery import FaceGallery
from pipeline.pose_estimator import HeadPoseEstimator, FaceQualityAssessor
from pipeline.tracker import FaceTracker
from pipeline.camera import CameraStream

logger = logging.getLogger(__name__)


class RecognitionResult:
    """Result for a single detected face in a frame."""
    __slots__ = ['bbox', 'name', 'confidence', 'similarity', 'pose', 'track_id']

    def __init__(self, bbox, name="Unknown", confidence=0.0, similarity=0.0, pose="None", track_id=-1):
        self.bbox = bbox
        self.name = name
        self.confidence = float(confidence)
        self.similarity = float(similarity)
        self.pose = pose
        self.track_id = track_id


class CameraWorker:
    """
    Manages the recognition pipeline for a single camera.
    Runs detection + recognition in a dedicated background worker thread.
    """

    def __init__(self, camera_id, detector, preprocessor, embedder, gallery, pose_estimator,
                 tracker_enabled=True, recognition_interval=10):
        self.camera_id = camera_id
        self.detector = detector
        self.preprocessor = preprocessor
        self.embedder = embedder
        self.gallery = gallery
        self.pose_estimator = pose_estimator

        # Tracker (per-camera)
        self.tracker = FaceTracker(
            iou_threshold=config.TRACKER_IOU_THRESHOLD,
            max_disappeared=config.TRACKER_MAX_DISAPPEARED,
            recognition_interval=recognition_interval,
        ) if tracker_enabled else None

        # Camera stream (bounded queue, 1-in-5 frame rule)
        self.camera = CameraStream(
            camera_id=camera_id,
            sample_interval=config.FRAME_SAMPLE_INTERVAL,
            queue_maxsize=config.FRAME_QUEUE_MAXSIZE,
            reconnect_delay=config.CAMERA_RECONNECT_DELAY,
            max_reconnect_attempts=config.CAMERA_MAX_RECONNECT_ATTEMPTS,
        )

        # Output state
        self._annotated_frame = None
        self._annotated_lock = threading.Lock()
        self._results = []
        self._results_lock = threading.Lock()

        # Processing thread
        self._processing_thread = None
        self._running = False

        # Metrics
        self._metrics = {
            'frames_processed': 0,
            'faces_detected': 0,
            'identities_matched': 0,
            'avg_detection_ms': 0.0,
            'avg_embedding_ms': 0.0,
            'avg_total_ms': 0.0,
        }
        self._metrics_lock = threading.Lock()

    def start(self):
        """Start camera capture and background processing."""
        if self._running:
            return

        self._running = True
        self.camera.start()

        self._processing_thread = threading.Thread(
            target=self._processing_loop,
            name=f"Processor-{self.camera_id}",
            daemon=True,
        )
        self._processing_thread.start()
        logger.info(f"CameraWorker {self.camera_id} started")

    def stop(self):
        """Stop camera capture and processing cleanly."""
        self._running = False
        self.camera.stop()

        if self._processing_thread is not None:
            self._processing_thread.join(timeout=5.0)
            self._processing_thread = None

        logger.info(f"CameraWorker {self.camera_id} stopped")

    def get_annotated_frame(self):
        """Get the latest annotated frame (thread-safe)."""
        with self._annotated_lock:
            if self._annotated_frame is not None:
                return self._annotated_frame.copy()
        return self.camera.get_latest_frame()

    def get_results(self):
        """Get the latest recognition results."""
        with self._results_lock:
            return list(self._results)

    def get_metrics(self):
        """Get performance metrics."""
        with self._metrics_lock:
            return dict(self._metrics)

    def get_status(self):
        """Get camera + worker status."""
        cam_status = self.camera.get_status()
        cam_status['processing'] = self._running
        cam_status['metrics'] = self.get_metrics()
        return cam_status

    def _processing_loop(self):
        """Main processing loop: detect → estimate pose → embed → match → annotate."""
        detection_times = []
        embedding_times = []
        total_times = []

        while self._running:
            frame = self.camera.get_frame(timeout=0.1)
            if frame is None:
                latest = self.camera.get_latest_frame()
                if latest is not None:
                    with self._results_lock:
                        results = list(self._results)
                    annotated = self._draw_annotations(latest, results)
                    with self._annotated_lock:
                        self._annotated_frame = annotated
                continue

            t_start = time.perf_counter()

            # ── 1. Face Detection ──
            t_det = time.perf_counter()
            detections = self.detector.detect(frame)
            det_ms = (time.perf_counter() - t_det) * 1000

            results = []

            if detections:
                bboxes = [d['bbox'] for d in detections]

                # ── 2. Tracking ──
                if self.tracker is not None:
                    assignments = self.tracker.update(detections)
                else:
                    assignments = [(i, i, True) for i in range(len(detections))]

                # ── 3. Recognition ──
                t_emb = time.perf_counter()
                faces_to_recognize = []  # (det_idx, track_id, crop, det)

                h, w = frame.shape[:2]
                for det_idx, track_id, needs_recognition in assignments:
                    det = detections[det_idx]
                    x1, y1, x2, y2 = [int(v) for v in det['bbox']]
                    crop = frame[max(0, y1):min(h, y2), max(0, x1):min(w, x2)]

                    if needs_recognition:
                        if crop.size > 0:
                            faces_to_recognize.append((det_idx, track_id, crop, det))
                        else:
                            results.append(RecognitionResult(
                                bbox=bboxes[det_idx],
                                name="Unknown",
                                confidence=0.0,
                                similarity=0.0,
                                track_id=track_id,
                            ))
                    else:
                        # Cached from tracker
                        if self.tracker is not None:
                            track = self.tracker.get_track(track_id)
                            if track is not None:
                                results.append(RecognitionResult(
                                    bbox=bboxes[det_idx],
                                    name=track.name,
                                    confidence=getattr(track, 'confidence', 0.0),
                                    similarity=track.similarity,
                                    track_id=track_id,
                                ))

                # Run embeddings and gallery matching
                if faces_to_recognize:
                    crops = [f[2] for f in faces_to_recognize]
                    embeddings = self.embedder.get_embeddings_batch(crops)

                    for i, embedding in enumerate(embeddings):
                        det_idx, track_id, _, det = faces_to_recognize[i]
                        name, confidence, similarity, match_pose = self.gallery.identify(embedding)

                        # Update tracker with recognized identity
                        if self.tracker is not None:
                            self.tracker.update_identity(track_id, name, similarity, confidence, match_pose)

                        results.append(RecognitionResult(
                            bbox=bboxes[det_idx],
                            name=name,
                            confidence=confidence,
                            similarity=similarity,
                            pose=match_pose,
                            track_id=track_id,
                        ))

                emb_ms = (time.perf_counter() - t_emb) * 1000
            else:
                emb_ms = 0.0
                if self.tracker is not None:
                    self.tracker.update([])

            t_total = (time.perf_counter() - t_start) * 1000

            # Update results & draw annotated frame
            with self._results_lock:
                self._results = results

            annotated = self._draw_annotations(frame, results)
            with self._annotated_lock:
                self._annotated_frame = annotated

            # Metrics
            detection_times.append(det_ms)
            embedding_times.append(emb_ms)
            total_times.append(t_total)
            if len(detection_times) > 30:
                detection_times.pop(0)
                embedding_times.pop(0)
                total_times.pop(0)

            with self._metrics_lock:
                self._metrics['frames_processed'] += 1
                self._metrics['faces_detected'] += len(detections)
                self._metrics['identities_matched'] += sum(
                    1 for r in results if r.name != "Unknown"
                )
                self._metrics['avg_detection_ms'] = float(np.mean(detection_times))
                self._metrics['avg_embedding_ms'] = float(np.mean(embedding_times))
                self._metrics['avg_total_ms'] = float(np.mean(total_times))

    def _draw_annotations(self, frame, results):
        """
        Draw bounding boxes and labels following Section 16 specification:
          - >= 90%: GREEN box + name + confidence percentage
          - 60% to < 90%: YELLOW box + name + confidence percentage
          - < 60%: NO IDENTITY LABEL (treated as unknown/uncertain)
        """
        annotated = frame.copy()

        for result in results:
            x1, y1, x2, y2 = [int(v) for v in result.bbox]
            conf = result.confidence

            # Section 16 Color & Label Rules
            if conf >= config.CONFIDENCE_THRESHOLD_GREEN:
                color = config.COLOR_GREEN
                label = f"{result.name} ({conf:.0f}%)"
            elif conf >= config.CONFIDENCE_THRESHOLD_YELLOW:
                color = config.COLOR_YELLOW
                label = f"{result.name} ({conf:.0f}%)"
            else:
                # Below 60%: NO IDENTITY LABEL
                color = config.COLOR_UNKNOWN
                label = "Unknown"

            # Draw bounding box
            cv2.rectangle(annotated, (x1, y1), (x2, y2), color, config.BBOX_THICKNESS)

            # Modern corner accents
            corner_len = min(20, max(5, (x2 - x1) // 5), max(5, (y2 - y1) // 5))
            thick = config.BBOX_THICKNESS + 1
            cv2.line(annotated, (x1, y1), (x1 + corner_len, y1), color, thick)
            cv2.line(annotated, (x1, y1), (x1, y1 + corner_len), color, thick)
            cv2.line(annotated, (x2, y1), (x2 - corner_len, y1), color, thick)
            cv2.line(annotated, (x2, y1), (x2, y1 + corner_len), color, thick)
            cv2.line(annotated, (x1, y2), (x1 + corner_len, y2), color, thick)
            cv2.line(annotated, (x1, y2), (x1, y2 - corner_len), color, thick)
            cv2.line(annotated, (x2, y2), (x2 - corner_len, y2), color, thick)
            cv2.line(annotated, (x2, y2), (x2, y2 - corner_len), color, thick)

            # Label banner
            font = cv2.FONT_HERSHEY_SIMPLEX
            font_scale = config.LABEL_FONT_SCALE
            font_thick = config.LABEL_FONT_THICKNESS
            pad = config.LABEL_BG_PADDING

            (text_w, text_h), baseline = cv2.getTextSize(label, font, font_scale, font_thick)
            label_y = max(y1 - 8, text_h + pad * 2)

            cv2.rectangle(
                annotated,
                (x1, label_y - text_h - pad * 2),
                (x1 + text_w + pad * 2, label_y),
                color, -1
            )
            cv2.putText(
                annotated, label,
                (x1 + pad, label_y - pad),
                font, font_scale, (255, 255, 255), font_thick
            )

        return annotated


class RecognitionEngine:
    """
    Top-level orchestrator for multi-camera face recognition.
    """

    def __init__(self):
        self.detector = None
        self.preprocessor = None
        self.embedder = None
        self.gallery = None
        self.pose_estimator = None
        self.quality_assessor = None
        self.workers = {}  # camera_id → CameraWorker
        self.active_model_name = config.ACTIVE_MODEL
        self._initialized = False

    def initialize(self, model_name=None):
        """Initialize detector, embedder, and gallery."""
        logger.info("Initializing Recognition Engine...")
        if model_name:
            self.active_model_name = model_name

        # BlazeFace detector
        self.detector = BlazeFaceDetector(
            model_path=config.BLAZEFACE_MODEL_PATH,
            confidence_threshold=config.BLAZEFACE_CONFIDENCE_THRESHOLD,
            nms_iou_threshold=config.BLAZEFACE_NMS_IOU_THRESHOLD,
        )

        self.preprocessor = FacePreprocessor(
            target_size=config.EMBEDDING_INPUT_SIZE,
            min_face_size=config.MIN_FACE_SIZE,
        )

        # Pose & quality modules
        self.pose_estimator = HeadPoseEstimator()
        self.quality_assessor = FaceQualityAssessor(
            min_face_size=config.MIN_FACE_SIZE,
            min_blur_score=config.MIN_BLUR_SCORE,
            min_detection_score=config.BLAZEFACE_CONFIDENCE_THRESHOLD
        )

        # Embedder model
        emb_path = config.GLINTR100_MODEL_PATH if self.active_model_name == "glintr100" else config.ADAFACE_MODEL_PATH
        self.embedder = FaceEmbedder(
            model_path=emb_path,
            model_type=self.active_model_name,
            num_threads=config.ONNX_NUM_THREADS,
            providers=config.ONNX_EXECUTION_PROVIDERS,
        )

        # Gallery
        self.gallery = FaceGallery(
            similarity_threshold=config.SIMILARITY_MATCH_BOUNDARY,
            max_refs_per_pose=config.MAX_REFS_PER_POSE,
            max_total_refs_per_identity=config.MAX_TOTAL_REFS_PER_IDENTITY,
        )

        # Enroll gallery
        self.re_enroll()

        self._initialized = True
        logger.info(f"Engine ready with model '{self.active_model_name}' and {len(self.gallery.get_enrolled_names())} identities.")

    def switch_model(self, model_name):
        """Switch embedding model (glintr100 or adaface) and reload gallery."""
        if model_name not in ["glintr100", "adaface"]:
            raise ValueError(f"Unknown model: {model_name}")

        logger.info(f"Switching recognition model to '{model_name}'...")
        self.active_model_name = model_name
        emb_path = config.GLINTR100_MODEL_PATH if model_name == "glintr100" else config.ADAFACE_MODEL_PATH

        self.embedder = FaceEmbedder(
            model_path=emb_path,
            model_type=model_name,
            num_threads=config.ONNX_NUM_THREADS,
            providers=config.ONNX_EXECUTION_PROVIDERS,
        )

        # Update workers
        for worker in self.workers.values():
            worker.embedder = self.embedder

        # Re-enroll gallery with the new model
        self.re_enroll()
        logger.info(f"Successfully switched to model '{model_name}'.")

    def re_enroll(self):
        """Scan directory and update multi-pose gallery embeddings."""
        self.gallery.gallery.clear()
        enrolled = self.gallery.enroll_from_directory(
            images_dir=config.GALLERY_IMAGES_DIR,
            detector=self.detector,
            embedder=self.embedder,
            pose_estimator=self.pose_estimator
        )
        self.gallery.save(config.GALLERY_EMBEDDINGS_PATH)
        return enrolled

    def add_camera(self, camera_id):
        """Add a camera stream worker."""
        if not self._initialized:
            raise RuntimeError("Engine not initialized.")
        if camera_id in self.workers:
            return

        worker = CameraWorker(
            camera_id=camera_id,
            detector=self.detector,
            preprocessor=self.preprocessor,
            embedder=self.embedder,
            gallery=self.gallery,
            pose_estimator=self.pose_estimator,
            tracker_enabled=config.TRACKER_ENABLED,
            recognition_interval=config.TRACKER_RECOGNITION_INTERVAL,
        )
        self.workers[camera_id] = worker
        logger.info(f"Added camera {camera_id} to engine")

    def remove_camera(self, camera_id):
        if camera_id in self.workers:
            self.workers[camera_id].stop()
            del self.workers[camera_id]

    def start_camera(self, camera_id):
        if camera_id in self.workers:
            self.workers[camera_id].start()

    def stop_camera(self, camera_id):
        if camera_id in self.workers:
            self.workers[camera_id].stop()

    def start_all(self):
        for worker in self.workers.values():
            worker.start()

    def stop_all(self):
        for worker in self.workers.values():
            worker.stop()

    def get_annotated_frame(self, camera_id):
        if camera_id in self.workers:
            return self.workers[camera_id].get_annotated_frame()
        return None

    def get_camera_status(self, camera_id):
        if camera_id in self.workers:
            return self.workers[camera_id].get_status()
        return None

    def get_all_status(self):
        return {cam_id: w.get_status() for cam_id, w in self.workers.items()}

    def get_gallery_info(self):
        if not self.gallery:
            return {}
        return {
            'enrolled_names': self.gallery.get_enrolled_names(),
            'total_embeddings': self.gallery.get_gallery_size(),
            'active_model': self.active_model_name,
            'summary': self.gallery.get_gallery_summary(),
        }

    def process_test_image(self, image_path):
        """Process a test image for evaluation."""
        img = cv2.imread(image_path)
        if img is None:
            return [], None

        detections = self.detector.detect(img)
        results = []
        h, w = img.shape[:2]

        for det in detections:
            x1, y1, x2, y2 = [int(v) for v in det['bbox']]
            crop = img[max(0, y1):min(h, y2), max(0, x1):min(w, x2)]
            if crop.size > 0:
                emb = self.embedder.get_embedding(crop)
                name, conf, sim, pose = self.gallery.identify(emb)
                results.append(RecognitionResult(
                    bbox=det['bbox'], name=name, confidence=conf, similarity=sim, pose=pose
                ))
            else:
                results.append(RecognitionResult(bbox=det['bbox']))

        dummy_worker = CameraWorker.__new__(CameraWorker)
        annotated = CameraWorker._draw_annotations(dummy_worker, img, results)
        return results, annotated
