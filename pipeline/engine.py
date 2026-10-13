"""
Recognition Engine — Multi-Camera Orchestrator with Pose Awareness & Section 16 Rules.

Orchestrates the complete face recognition pipeline:
  Camera → Frame Sampler → BlazeFace Detection → Pose Estimation →
  Embedder (GlintR100 or AdaFace IR-101) → Multi-Pose Gallery Matching →
  Section 16 Calibrated Display Annotation → Flask Web Stream

Handles multiple independent cameras concurrently with bounded queues.
"""

import os
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
                 attendance_manager=None,
                 tracker_enabled=True, recognition_interval=10):
        self.camera_id = camera_id
        self.detector = detector
        self.preprocessor = preprocessor
        self.embedder = embedder
        self.gallery = gallery
        self.pose_estimator = pose_estimator
        self.attendance_manager = attendance_manager

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
            'faces_in_frame': 0,
            'identities_matched': 0,
            'latest_detection_ms': 0.0,
            'latest_embedding_ms': 0.0,
            'latest_total_ms': 0.0,
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

                # ── 3. Face Quality / Pose Check & Realignment ──
                t_emb = time.perf_counter()
                faces_to_recognize = []  # (det_idx, track_id, crop, det, blur_score)

                h, w = frame.shape[:2]
                min_face_size = getattr(config, 'MIN_FACE_SIZE', 50)
                min_blur_score = getattr(config, 'MIN_BLUR_SCORE', 45.0)

                for det_idx, track_id, needs_recognition in assignments:
                    det = detections[det_idx]
                    bbox = det['bbox']
                    face_w = bbox[2] - bbox[0]
                    face_h = bbox[3] - bbox[1]

                    # Face quality verification: reject tiny, invalid, or truncated bounding boxes
                    is_valid_bbox = (
                        face_w >= min_face_size and face_h >= min_face_size and
                        np.isfinite(bbox[0]) and np.isfinite(bbox[1]) and
                        np.isfinite(bbox[2]) and np.isfinite(bbox[3])
                    )

                    if not is_valid_bbox:
                        crop = None
                        blur_score = 0.0
                    else:
                        if self.preprocessor is not None:
                            crop = self.preprocessor.align_face(frame, det.get('keypoints'), bbox)
                        else:
                            x1, y1, x2, y2 = [int(v) for v in bbox]
                            crop = frame[max(0, y1):min(h, y2), max(0, x1):min(w, x2)]

                        if crop is not None and crop.size > 0:
                            # Face Quality Check: Laplacian variance for sharpness
                            gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
                            blur_score = float(cv2.Laplacian(gray, cv2.CV_64F).var())
                        else:
                            blur_score = 0.0

                    # Reject low-quality / blurry faces before embedding (Section 4)
                    is_quality_pass = (crop is not None and crop.size > 0 and blur_score >= min_blur_score)

                    if needs_recognition:
                        if is_quality_pass:
                            faces_to_recognize.append((det_idx, track_id, crop, det, blur_score))
                        else:
                            # Reject crop as Unknown / Uncertain without embedding
                            if self.tracker is not None:
                                self.tracker.update_identity(track_id, "Unknown", 0.0, 0.0, "None", quality=blur_score)
                            results.append(RecognitionResult(
                                bbox=bboxes[det_idx],
                                name="Unknown",
                                confidence=0.0,
                                similarity=0.0,
                                track_id=track_id,
                            ))
                    else:
                        # Cached & smoothed observation from tracker
                        if self.tracker is not None:
                            track = self.tracker.get_track(track_id)
                            if track is not None:
                                results.append(RecognitionResult(
                                    bbox=bboxes[det_idx],
                                    name=track.name,
                                    confidence=getattr(track, 'confidence', 0.0),
                                    similarity=track.similarity,
                                    pose=track.pose,
                                    track_id=track_id,
                                ))

                # ── 4. GlintR100 Embeddings & 5. Gallery Matching ──
                if faces_to_recognize:
                    crops = [f[2] for f in faces_to_recognize]
                    embeddings = self.embedder.get_embeddings_batch(crops)

                    for i, embedding in enumerate(embeddings):
                        det_idx, track_id, crop, det, blur_score = faces_to_recognize[i]
                        
                        # Pose check (3D head rotation matrix)
                        det_pose = "Front"
                        if self.pose_estimator is not None:
                            det_pose, _ = self.pose_estimator.estimate_pose(
                                keypoints=det.get('keypoints'),
                                bbox=det['bbox'],
                                face_crop=crop
                            )

                        # Employee gallery matching
                        raw_name, raw_conf, raw_sim, match_pose = self.gallery.identify(embedding, query_pose=det_pose)

                        # ── 6. Combine Observations Across Frames ──
                        if self.tracker is not None:
                            name, confidence, similarity, match_pose = self.tracker.update_identity(
                                track_id, raw_name, raw_sim, raw_conf, match_pose, quality=blur_score,
                                crop=crop, frame=frame, bbox=bboxes[det_idx]
                            )
                            # Evaluate attendance event (isolated per camera & duplicate suppressed)
                            if self.attendance_manager is not None:
                                track = self.tracker.get_track(track_id)
                                best_obs = track.best_observation if track else None
                                is_confirmed = getattr(track, 'is_confirmed', False) if track else False
                                direction = getattr(track, 'direction', "Entry") if track else "Entry"

                                # GUARD: If the track is currently accumulating observations for an enrolled candidate
                                # (e.g. candidate_name is 'Anoop Verma'), do NOT dispatch an unconfirmed visitor event!
                                is_candidate_pending = (
                                    not is_confirmed and track and
                                    track.candidate_name is not None and
                                    track.candidate_name != "Unknown"
                                )
                                if not is_candidate_pending:
                                    self.attendance_manager.process_observation(
                                        track_id=track_id,
                                        name=name,
                                        confidence=confidence,
                                        similarity=similarity,
                                        pose=match_pose,
                                        quality_score=blur_score,
                                        camera_id=self.camera_id,
                                        best_observation=best_obs,
                                        is_confirmed=is_confirmed,
                                        direction=direction,
                                        frame=frame,
                                        bbox=bboxes[det_idx]
                                    )
                        else:
                            name, confidence, similarity = raw_name, raw_conf, raw_sim

                        # ── 7. Identity or Unknown -> Calibrated Confidence ──
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
                self._metrics['faces_in_frame'] = len(results)
                self._metrics['identities_matched'] += sum(
                    1 for r in results if r.name != "Unknown"
                )
                self._metrics['latest_detection_ms'] = round(float(det_ms), 1)
                self._metrics['latest_embedding_ms'] = round(float(emb_ms), 1)
                self._metrics['latest_total_ms'] = round(float(t_total), 1)
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

            # Track ID prefix and attendance status badge
            track_prefix = f"[#{result.track_id}] " if getattr(result, 'track_id', -1) >= 0 else ""
            is_logged = False
            if self.attendance_manager and getattr(result, 'track_id', -1) >= 0:
                is_logged = self.attendance_manager.is_track_logged(result.track_id, camera_id=self.camera_id)
            att_suffix = " (Logged)" if is_logged else ""

            # Section 16 Color & Label Rules
            if conf >= config.CONFIDENCE_THRESHOLD_GREEN:
                color = config.COLOR_GREEN
                label = f"{track_prefix}{result.name} ({conf:.0f}%){att_suffix}"
            elif conf >= config.CONFIDENCE_THRESHOLD_YELLOW:
                color = config.COLOR_YELLOW
                label = f"{track_prefix}{result.name} ({conf:.0f}%){att_suffix}"
            else:
                # Below 60%: Unknown Visitor label (e.g. Unknown-001)
                color = config.COLOR_UNKNOWN
                unk_name = result.name if (result.name and result.name.startswith("Unknown-")) else (
                    f"Unknown-{result.track_id + 1:03d}" if getattr(result, 'track_id', -1) >= 0 else "Unknown"
                )
                label = f"{track_prefix}{unk_name}{att_suffix}"

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
        self.attendance_manager = None
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
        self.pose_estimator = HeadPoseEstimator(model_path=getattr(config, 'HEAD_POSE_MODEL_PATH', None))
        self.quality_assessor = FaceQualityAssessor(
            min_face_size=config.MIN_FACE_SIZE,
            min_blur_score=config.MIN_BLUR_SCORE,
            min_detection_score=config.BLAZEFACE_CONFIDENCE_THRESHOLD
        )

        # Attendance Event Manager with Duplicate Suppression
        from pipeline.attendance import AttendanceManager
        self.attendance_manager = AttendanceManager(
            log_path=getattr(config, 'ATTENDANCE_LOG_PATH', None),
            min_confidence=getattr(config, 'ATTENDANCE_CONFIDENCE_THRESHOLD', 85.0),
            min_stable_frames=getattr(config, 'ATTENDANCE_MIN_FRAMES', 3),
            cooldown_seconds=getattr(config, 'ATTENDANCE_COOLDOWN_SECONDS', 300)
        )

        # Embedder model
        model_cfg = config.get_model_config(self.active_model_name)
        self.embedder = FaceEmbedder(
            model_path=model_cfg["path"],
            model_type=model_cfg["type"],
            num_threads=config.ONNX_NUM_THREADS,
            providers=config.ONNX_EXECUTION_PROVIDERS,
        )

        # Gallery
        self.gallery = FaceGallery(
            similarity_threshold=config.SIMILARITY_MATCH_BOUNDARY,
            max_refs_per_pose=config.MAX_REFS_PER_POSE,
            max_total_refs_per_identity=config.MAX_TOTAL_REFS_PER_IDENTITY,
        )

        # 1. Fast startup: Load cached gallery embeddings if available (<10ms)
        gallery_cache = config.get_gallery_cache_path(self.active_model_name)
        loaded = self.gallery.load(gallery_cache)
        if loaded:
            logger.info(f"Loaded cached gallery for '{self.active_model_name}': {len(self.gallery.get_enrolled_names())} identities ready instantly.")
        else:
            logger.info(f"No cached gallery found for '{self.active_model_name}'. Enrolling from raw sources...")

        # 2. Incremental check: only process new or modified videos
        new_enrolled = self.sync_gallery(skip_existing=True)
        if new_enrolled > 0 or not loaded:
            self.gallery.save(gallery_cache)

        self._initialized = True
        logger.info(f"Engine ready with model '{self.active_model_name}' and {len(self.gallery.get_enrolled_names())} identities.")

    def switch_model(self, model_name):
        """
        Switch recognition embedding model cleanly:
        1. FIRST stop running camera workers and freeze current inference.
        2. Release the old model's ONNX session and free GPU memory.
        3. START the newly selected model and initialize its weights on GPU.
        4. Load / synchronize the model-specific reference gallery.
        5. Update worker references and restart camera workers.
        """
        model_name = model_name.lower()
        if model_name not in config.AVAILABLE_MODELS:
            raise ValueError(f"Unknown model: '{model_name}'. Available: {list(config.AVAILABLE_MODELS.keys())}")

        if model_name == self.active_model_name and self.embedder is not None:
            logger.info(f"Model '{model_name}' is already active and running.")
            return {
                "previous_model": self.active_model_name,
                "active_model": model_name,
                "name": config.AVAILABLE_MODELS[model_name]["name"],
                "message": f"Model '{config.AVAILABLE_MODELS[model_name]['name']}' is already active."
            }

        prev_model = self.active_model_name
        prev_display = config.AVAILABLE_MODELS.get(prev_model, {}).get("name", prev_model)
        new_display = config.AVAILABLE_MODELS[model_name]["name"]

        logger.info(f"[Model Switch] === Starting model transition: '{prev_display}' -> '{new_display}' ===")

        # ─── Step 1: FIRST STOP the running model and camera workers ─────────
        running_cams = [cid for cid, worker in self.workers.items() if worker._running]
        if running_cams:
            logger.info(f"[Model Switch] Stopping {len(running_cams)} camera worker(s) ({running_cams}) running under '{prev_display}'...")
            for cid in running_cams:
                self.workers[cid].stop()

        # Free previous model session and GPU memory
        if self.embedder is not None:
            logger.info(f"[Model Switch] Releasing old model '{prev_display}' from GPU memory...")
            try:
                del self.embedder.session
            except Exception:
                pass
            self.embedder = None
            import gc
            gc.collect()

        logger.info(f"[Model Switch] Model '{prev_display}' stopped completely.")

        # ─── Step 2: START the new selected model ────────────────────────────
        logger.info(f"[Model Switch] Initializing and starting new model '{new_display}' on GPU...")
        self.active_model_name = model_name
        model_cfg = config.AVAILABLE_MODELS[model_name]

        self.embedder = FaceEmbedder(
            model_path=model_cfg["path"],
            model_type=model_cfg["type"],
            num_threads=config.ONNX_NUM_THREADS,
            providers=config.ONNX_EXECUTION_PROVIDERS,
        )

        # ─── Step 3: Switch gallery to model-specific feature cache ──────────
        gallery_cache = config.get_gallery_cache_path(model_name)
        self.gallery.gallery.clear()
        loaded = self.gallery.load(gallery_cache)
        if loaded:
            logger.info(f"[Model Switch] Loaded {len(self.gallery.get_enrolled_names())} enrolled identities from cache '{gallery_cache}'.")
        else:
            logger.info(f"[Model Switch] No cache found at '{gallery_cache}'. Enrolling from raw sources for '{model_name}'...")
            self.sync_gallery(skip_existing=False)
            self.gallery.save(gallery_cache)

        # Re-point worker embedders & galleries
        for worker in self.workers.values():
            worker.embedder = self.embedder
            worker.gallery = self.gallery

        # ─── Step 4: RESTART camera workers with the new model ───────────────
        if running_cams:
            logger.info(f"[Model Switch] Restarting {len(running_cams)} camera worker(s) with new model '{new_display}'...")
            for cid in running_cams:
                self.workers[cid].start()

        msg = f"Stopped '{prev_display}' and started '{new_display}' successfully on GPU."
        logger.info(f"[Model Switch] {msg}")
        return {
            "previous_model": prev_model,
            "active_model": model_name,
            "name": new_display,
            "message": msg
        }

    def sync_gallery(self, skip_existing=True):
        """
        Incrementally cross-check sources.
        Only processes newly added videos/images, avoiding re-processing existing ones.
        """
        enrolled = 0

        # Check raw videos directory
        if os.path.isdir(config.GALLERY_VIDEOS_DIR):
            enrolled += self.gallery.enroll_from_videos_directory(
                videos_dir=config.GALLERY_VIDEOS_DIR,
                detector=self.detector,
                embedder=self.embedder,
                pose_estimator=self.pose_estimator,
                quality_assessor=self.quality_assessor,
                preprocessor=self.preprocessor,
                skip_existing=skip_existing
            )

        # Check images directory if exists
        if os.path.isdir(config.GALLERY_IMAGES_DIR):
            enrolled += self.gallery.enroll_from_directory(
                images_dir=config.GALLERY_IMAGES_DIR,
                detector=self.detector,
                embedder=self.embedder,
                pose_estimator=self.pose_estimator
            )

        if enrolled > 0:
            self.gallery.save(config.get_gallery_cache_path(self.active_model_name))
        return enrolled

    def re_enroll(self):
        """Force full reload of gallery."""
        self.gallery.gallery.clear()
        enrolled = self.sync_gallery(skip_existing=False)
        return enrolled

    def train_model(self, model_name=None, progress_callback=None):
        """
        Train / enroll a specific model on all videos in Ppl/Raw-Videos.
        Extracts pose-aware, canonically aligned embeddings and saves them
        separately to gallery/embeddings_<model_name>.json.

        If the model being trained is the currently active model,
        the active gallery is updated in-place immediately.
        """
        target_model = (model_name or self.active_model_name).lower()
        if target_model not in config.AVAILABLE_MODELS:
            raise ValueError(f"Unknown model: '{target_model}'. Available: {list(config.AVAILABLE_MODELS.keys())}")

        model_cfg = config.AVAILABLE_MODELS[target_model]
        model_display = model_cfg["name"]
        cache_path = config.get_gallery_cache_path(target_model)

        logger.info(f"[Training] Starting training for '{model_display}' from '{config.GALLERY_VIDEOS_DIR}'...")

        # Obtain embedder for this model
        dedicated_embedder = None
        if target_model == self.active_model_name and self.embedder is not None:
            embedder_to_use = self.embedder
        else:
            embedder_to_use = FaceEmbedder(
                model_path=model_cfg["path"],
                model_type=model_cfg["type"],
                num_threads=config.ONNX_NUM_THREADS,
                providers=config.ONNX_EXECUTION_PROVIDERS,
            )
            dedicated_embedder = embedder_to_use

        # Create target gallery
        target_gallery = FaceGallery(
            similarity_threshold=config.SIMILARITY_MATCH_BOUNDARY,
            max_refs_per_pose=config.MAX_REFS_PER_POSE,
            max_total_refs_per_identity=config.MAX_TOTAL_REFS_PER_IDENTITY,
        )

        try:
            # Process all videos with alignment
            enrolled_count = target_gallery.enroll_from_videos_directory(
                videos_dir=config.GALLERY_VIDEOS_DIR,
                detector=self.detector,
                embedder=embedder_to_use,
                pose_estimator=self.pose_estimator,
                quality_assessor=self.quality_assessor,
                preprocessor=self.preprocessor,
                skip_existing=False,
                progress_callback=progress_callback
            )

            # Save embeddings separately for this model
            target_gallery.save(cache_path)
            logger.info(f"[Training] Saved {target_gallery.get_gallery_size()} embeddings for '{model_display}' to '{cache_path}'")

            # If trained model is active, update engine gallery
            if target_model == self.active_model_name:
                self.gallery.gallery = target_gallery.gallery.copy()
                for worker in self.workers.values():
                    worker.gallery = self.gallery

            return {
                "success": True,
                "model": target_model,
                "model_display": model_display,
                "identities_enrolled": len(target_gallery.get_enrolled_names()),
                "total_embeddings": target_gallery.get_gallery_size(),
                "enrolled_names": target_gallery.get_enrolled_names(),
                "cache_file": cache_path,
                "summary": target_gallery.get_gallery_summary()
            }
        finally:
            if dedicated_embedder is not None:
                try:
                    del dedicated_embedder.session
                except Exception:
                    pass
                del dedicated_embedder

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
            attendance_manager=self.attendance_manager,
            tracker_enabled=config.TRACKER_ENABLED,
            recognition_interval=config.TRACKER_RECOGNITION_INTERVAL,
        )
        self.workers[camera_id] = worker
        logger.info(f"Added camera {camera_id} to engine")

    def get_attendance_events(self, limit=100):
        if self.attendance_manager:
            return self.attendance_manager.get_recent_events(limit=limit)
        return []

    def get_attendance_summary(self):
        if self.attendance_manager:
            return self.attendance_manager.get_summary()
        return {}

    def clear_attendance(self):
        if self.attendance_manager:
            return self.attendance_manager.clear()
        return False

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
