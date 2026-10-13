"""
Face Gallery — Pose-Aware Identity Enrollment and Matching.

Stores multi-pose reference embeddings per identity:
  - Front, Left, Right, Up, Down
Implements calibrated confidence percentage scoring following Section 16:
  - >= 90%: High confidence (Green display)
  - 60% - 89.9%: Medium confidence (Yellow display)
  - < 60%: Unknown / Uncertain (No identity label)
"""

import os
import json
import time
import numpy as np
import cv2
import logging

import config

logger = logging.getLogger(__name__)


def calibrate_confidence_score(cosine_sim):
    """
    Calibrate raw cosine similarity into a 0.0 - 100.0% confidence score.

    Calibrated against genuine cross-pose and impostor distributions:
      - Impostors / background noise: sim <= 0.20 -> Conf < 40%
      - Decision boundary: sim = 0.35 -> Conf = 60.0% (Yellow threshold)
      - High confidence boundary: sim = 0.50 -> Conf = 90.0% (Green threshold)
      - Perfect / near identical: sim >= 0.70 -> Conf >= 98.0%

    Args:
        cosine_sim: float in [-1.0, 1.0]

    Returns:
        confidence: float in [0.0, 100.0]
    """
    sim = float(cosine_sim)

    if sim <= 0.15:
        # Deep unknown / impostor territory
        conf = max(0.0, (sim / 0.15) * 30.0) if sim > 0 else 0.0
    elif sim < 0.35:
        # 0.15 -> 30%, 0.35 -> 60%
        conf = 30.0 + ((sim - 0.15) / 0.20) * 30.0
    elif sim < 0.50:
        # 0.35 -> 60%, 0.50 -> 90%
        conf = 60.0 + ((sim - 0.35) / 0.15) * 30.0
    else:
        # 0.50 -> 90%, 0.70+ -> 99.5%
        conf = min(99.9, 90.0 + ((sim - 0.50) / 0.20) * 9.9)

    return float(conf)


class FaceGallery:
    """
    Pose-aware identity gallery.

    Structure:
      self.gallery = {
        "person_name": {
          "id": "person_name_id",
          "name": "Person Name",
          "created_at": timestamp,
          "references": [
            {
              "pose": "Front" | "Left" | "Right" | "Up" | "Down" | "General",
              "embedding": np.array(512,),
              "quality_score": float,
              "timestamp": float
            },
            ...
          ]
        }
      }
    """

    def __init__(self,
                 similarity_threshold=0.35,
                 max_refs_per_pose=3,
                 max_total_refs_per_identity=15):
        """
        Args:
            similarity_threshold: Min cosine similarity boundary for unknown rejection
            max_refs_per_pose: Max reference samples per pose category
            max_total_refs_per_identity: Max total reference samples per person
        """
        self.similarity_threshold = similarity_threshold
        self.max_refs_per_pose = max_refs_per_pose
        self.max_total_refs_per_identity = max_total_refs_per_identity

        # {identity_name: {"name": ..., "references": [...]}}
        self.gallery = {}

    def enroll(self, name, embedding, pose="General", quality_score=100.0):
        """
        Add a reference embedding with pose annotation.

        Args:
            name: Person display name
            embedding: (512,) float32 L2-normalized vector
            pose: One of "Front", "Left", "Right", "Up", "Down", "General"
            quality_score: Sharpness or quality metric
        """
        if embedding is None or len(embedding) == 0:
            return False

        emb_norm = embedding / np.linalg.norm(embedding)

        if name not in self.gallery:
            self.gallery[name] = {
                "id": name.lower().replace(" ", "_"),
                "name": name,
                "created_at": time.time(),
                "references": []
            }

        person = self.gallery[name]
        # Count existing references for this pose
        pose_refs = [r for r in person["references"] if r["pose"] == pose]

        if len(pose_refs) >= self.max_refs_per_pose:
            # Replace lowest quality reference if new one is sharper
            lowest = min(pose_refs, key=lambda r: r.get("quality_score", 0.0))
            if quality_score > lowest.get("quality_score", 0.0):
                person["references"].remove(lowest)
                person["references"].append({
                    "pose": pose,
                    "embedding": emb_norm.copy(),
                    "quality_score": float(quality_score),
                    "timestamp": time.time()
                })
                logger.info(f"Updated '{name}' ({pose}) reference with higher quality ({quality_score:.1f})")
                return True
            return False

        if len(person["references"]) >= self.max_total_refs_per_identity:
            logger.debug(f"'{name}' reached max total references limit ({self.max_total_refs_per_identity})")
            return False

        person["references"].append({
            "pose": pose,
            "embedding": emb_norm.copy(),
            "quality_score": float(quality_score),
            "timestamp": time.time()
        })
        logger.info(f"Enrolled '{name}' [{pose}]: {len(person['references'])} total reference(s)")
        return True

    def identify(self, query_embedding, query_pose=None):
        """
        Match a query embedding against the gallery.
        If query_pose is specified, applies pose-filtered matching:
        compares against templates matching that specific pose
        (e.g. Front vs Front, Up vs Up) to prevent cross-angle impostor contamination.

        Args:
            query_embedding: (512,) float32 vector
            query_pose: Optional pose string ("Front", "Left", "Right", "Up", "Down")

        Returns:
            name: Person name or "Unknown"
            confidence: Calibrated confidence percentage (0.0 - 100.0)
            max_similarity: Raw cosine similarity (-1.0 to 1.0)
            best_pose: Matching reference pose (e.g. "Front", "Left")
        """
        if not self.gallery or query_embedding is None:
            return "Unknown", 0.0, -1.0, "None"

        q_norm = query_embedding / np.linalg.norm(query_embedding)

        # 1. Pose-Filtered Matching if query_pose is provided
        target_pose = query_pose if (query_pose in ["Front", "Left", "Right", "Up", "Down"]) else None

        best_name = "Unknown"
        best_sim = -1.0
        best_pose = "None"

        # Compare query against enrolled identities
        for name, record in self.gallery.items():
            refs = record.get("references", [])
            if not refs:
                continue

            person_best_sim = -1.0
            person_best_pose = "None"

            # 1. Check matching pose references if available
            if target_pose:
                for ref in refs:
                    if ref.get("pose") == target_pose:
                        sim = float(np.dot(q_norm, ref["embedding"]))
                        if sim > person_best_sim:
                            person_best_sim = sim
                            person_best_pose = target_pose

            # 2. Check canonical Front and general references (anchor representation)
            # Aligned face crops preserve canonical features even when head has minor roll/tilt
            for ref in refs:
                sim = float(np.dot(q_norm, ref["embedding"]))
                if sim > person_best_sim:
                    person_best_sim = sim
                    person_best_pose = ref.get("pose", "Front")

            if person_best_sim > best_sim:
                best_sim = person_best_sim
                best_name = name
                best_pose = person_best_pose

        conf_percent = calibrate_confidence_score(best_sim)

        # Section 16: Below 60% — NO IDENTITY LABEL (treated as unknown/uncertain)
        if conf_percent >= config.CONFIDENCE_THRESHOLD_YELLOW:
            return best_name, conf_percent, best_sim, best_pose
        else:
            return "Unknown", conf_percent, best_sim, best_pose

    def identify_batch(self, query_embeddings):
        """Match multiple embeddings."""
        return [self.identify(emb) for emb in query_embeddings]

    def enroll_from_directory(self, images_dir, detector, embedder, pose_estimator=None):
        """
        Auto-enroll identities from directory structure:
          Images/
            Anoop_Verma/
              Front.jpeg, Left.jpeg, Right.jpeg, Up.jpeg, Down.jpeg
            Nirvighna/
              Nirvighna.jpeg

        Returns:
            enrolled_count: Number of successfully enrolled identities
        """
        if not os.path.isdir(images_dir):
            logger.error(f"Gallery directory not found: {images_dir}")
            return 0

        image_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.webp'}
        enrolled_identities = 0

        for item in sorted(os.listdir(images_dir)):
            item_path = os.path.join(images_dir, item)

            if os.path.isdir(item_path):
                # Subdirectory per person (e.g. Anoop_Verma)
                name = item.replace('_', ' ')
                has_enrolled = False

                for img_name in sorted(os.listdir(item_path)):
                    ext = os.path.splitext(img_name)[1].lower()
                    if ext not in image_extensions:
                        continue

                    img_file = os.path.join(item_path, img_name)
                    img = cv2.imread(img_file)
                    if img is None:
                        continue

                    dets = detector.detect(img)
                    if not dets:
                        continue

                    best_det = max(dets, key=lambda d: d['score'])
                    x1, y1, x2, y2 = [int(v) for v in best_det['bbox']]
                    h, w = img.shape[:2]
                    crop = img[max(0, y1):min(h, y2), max(0, x1):min(w, x2)]
                    if crop.size == 0:
                        continue

                    # Pose estimation
                    base_name = os.path.splitext(img_name)[0]
                    detected_pose = "General"
                    if base_name in ["Front", "Left", "Right", "Up", "Down"]:
                        detected_pose = base_name
                    elif pose_estimator is not None:
                        detected_pose, _ = pose_estimator.estimate_pose(best_det['keypoints'], best_det['bbox'])

                    emb = embedder.get_embedding(crop)
                    if emb is not None:
                        self.enroll(name, emb, pose=detected_pose, quality_score=float(best_det['score'] * 100))
                        has_enrolled = True

                if has_enrolled:
                    enrolled_identities += 1

            elif os.path.isfile(item_path):
                # Single image file in directory
                ext = os.path.splitext(item)[1].lower()
                if ext in image_extensions:
                    name = os.path.splitext(item)[0].replace('_', ' ')
                    img = cv2.imread(item_path)
                    if img is None:
                        continue
                    dets = detector.detect(img)
                    if not dets:
                        continue
                    best_det = max(dets, key=lambda d: d['score'])
                    x1, y1, x2, y2 = [int(v) for v in best_det['bbox']]
                    h, w = img.shape[:2]
                    crop = img[max(0, y1):min(h, y2), max(0, x1):min(w, x2)]
                    emb = embedder.get_embedding(crop)
                    if emb is not None:
                        self.enroll(name, emb, pose="Front", quality_score=float(best_det['score'] * 100))
                        enrolled_identities += 1

        logger.info(f"Directory enrollment complete: {enrolled_identities} identities enrolled")
        return enrolled_identities

    def enroll_from_videos_directory(self, videos_dir, detector, embedder, pose_estimator, quality_assessor, preprocessor=None, skip_existing=True, progress_callback=None):
        """
        Incrementally enroll identities from video files in data/raw_videos.
        If skip_existing=True, skips videos whose identity is already enrolled,
        unless the video file was modified after enrollment.
        """
        from pipeline.enrollment import VideoEnrollmentManager
        from pipeline.preprocessor import FacePreprocessor

        if not os.path.isdir(videos_dir):
            logger.warning(f"Gallery videos directory not found: {videos_dir}")
            return 0

        if preprocessor is None:
            preprocessor = FacePreprocessor()

        manager = VideoEnrollmentManager(
            target_samples_per_pose=self.max_refs_per_pose,
            frame_sample_step=config.FRAME_SAMPLE_INTERVAL
        )

        enrolled = 0
        video_exts = {'.mp4', '.avi', '.mov', '.webm', '.mkv'}
        all_files = [f for f in sorted(os.listdir(videos_dir)) if os.path.splitext(f)[1].lower() in video_exts]
        total_files = len(all_files)

        for idx, filename in enumerate(all_files):
            name = os.path.splitext(filename)[0]
            video_path = os.path.join(videos_dir, filename)
            video_mtime = os.path.getmtime(video_path)

            if skip_existing and name in self.gallery:
                rec = self.gallery[name]
                refs = rec.get("references", [])
                recorded_mtime = rec.get("video_mtime", rec.get("created_at", 0.0))
                if len(refs) > 0 and recorded_mtime >= video_mtime:
                    logger.debug(f"Identity '{name}' already enrolled and up to date, skipping.")
                    if progress_callback:
                        progress_callback(idx + 1, total_files, filename, name, "skipped", f"Already enrolled ({len(refs)} refs)")
                    continue
                # If modified or incomplete, remove old references to re-enroll
                self.remove_identity(name)

            if progress_callback:
                progress_callback(idx + 1, total_files, filename, name, "processing", f"Processing {filename}...")

            logger.info(f"Enrolling new/updated video: '{filename}' for '{name}'...")
            try:
                report = manager.enroll_from_video(
                    video_path=video_path,
                    identity_name=name,
                    gallery=self,
                    detector=detector,
                    embedder=embedder,
                    pose_estimator=pose_estimator,
                    quality_assessor=quality_assessor,
                    preprocessor=preprocessor,
                )
                if report.get("success"):
                    enrolled += 1
                    if name in self.gallery:
                        self.gallery[name]["video_mtime"] = video_mtime
                    logger.info(f"✓ Enrolled '{name}': {report.get('total_enrolled_references')} references across poses: {report.get('enrolled_per_pose')}")
                    if progress_callback:
                        progress_callback(idx + 1, total_files, filename, name, "completed", f"Enrolled {report.get('total_enrolled_references')} refs")
                else:
                    if progress_callback:
                        progress_callback(idx + 1, total_files, filename, name, "warning", f"Incomplete: {report.get('status')}")
            except Exception as e:
                logger.error(f"Error enrolling video {filename}: {e}")
                if progress_callback:
                    progress_callback(idx + 1, total_files, filename, name, "error", str(e))

        if enrolled > 0:
            logger.info(f"Video enrollment: {enrolled} new/updated identities processed")
        return enrolled

    def get_enrolled_names(self):
        """Return list of enrolled names."""
        return list(self.gallery.keys())

    def get_gallery_summary(self):
        """Return summary of identities and pose counts."""
        summary = {}
        for name, record in self.gallery.items():
            pose_counts = {}
            for r in record["references"]:
                p = r.get("pose", "General")
                pose_counts[p] = pose_counts.get(p, 0) + 1
            summary[name] = {
                "id": record.get("id", name),
                "total_references": len(record["references"]),
                "poses": pose_counts
            }
        return summary

    get_summary = get_gallery_summary

    def get_gallery_size(self):
        """Total number of reference vectors across all identities."""
        return sum(len(rec["references"]) for rec in self.gallery.values())

    def remove_identity(self, name):
        """Remove a person entirely."""
        if name in self.gallery:
            del self.gallery[name]
            logger.info(f"Removed '{name}' from gallery")
            return True
        return False

    def remove_pose_reference(self, name, pose):
        """Remove references of a specific pose for a person."""
        if name in self.gallery:
            person = self.gallery[name]
            orig_len = len(person["references"])
            person["references"] = [r for r in person["references"] if r.get("pose") != pose]
            removed = orig_len - len(person["references"])
            logger.info(f"Removed {removed} reference(s) for '{name}' with pose '{pose}'")
            return removed > 0
        return False

    def save(self, filepath):
        """Save gallery to JSON."""
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        data = {}
        for name, record in self.gallery.items():
            data[name] = {
                "id": record["id"],
                "name": record["name"],
                "created_at": record["created_at"],
                "references": [
                    {
                        "pose": r["pose"],
                        "embedding": r["embedding"].tolist(),
                        "quality_score": r.get("quality_score", 100.0),
                        "timestamp": r.get("timestamp", time.time())
                    }
                    for r in record["references"]
                ]
            }
        with open(filepath, 'w') as f:
            json.dump(data, f, indent=2)
        logger.info(f"Saved gallery to {filepath} ({len(data)} identities)")

    def load(self, filepath):
        """Load gallery from JSON."""
        if not os.path.isfile(filepath):
            logger.warning(f"Gallery file not found: {filepath}")
            return False

        try:
            with open(filepath, 'r') as f:
                data = json.load(f)

            self.gallery = {}
            for name, record in data.items():
                if isinstance(record, dict) and "references" in record:
                    # New format
                    self.gallery[name] = {
                        "id": record.get("id", name.lower().replace(" ", "_")),
                        "name": record.get("name", name),
                        "created_at": record.get("created_at", time.time()),
                        "references": [
                            {
                                "pose": r.get("pose", "General"),
                                "embedding": np.array(r["embedding"], dtype=np.float32),
                                "quality_score": r.get("quality_score", 100.0),
                                "timestamp": r.get("timestamp", time.time())
                            }
                            for r in record["references"]
                        ]
                    }
                elif isinstance(record, list):
                    # Legacy format {name: [emb1, emb2]}
                    self.gallery[name] = {
                        "id": name.lower().replace(" ", "_"),
                        "name": name,
                        "created_at": time.time(),
                        "references": [
                            {
                                "pose": "General",
                                "embedding": np.array(emb, dtype=np.float32),
                                "quality_score": 100.0,
                                "timestamp": time.time()
                            }
                            for emb in record
                        ]
                    }

            logger.info(
                f"Gallery loaded from {filepath}: {len(self.gallery)} identities, "
                f"{self.get_gallery_size()} total references"
            )
            return True
        except Exception as e:
            logger.error(f"Failed to load gallery from {filepath}: {e}")
            return False
