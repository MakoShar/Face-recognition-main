"""
Automated Test Suite for Critical Face Recognition Bug Fixes.

Covers the 10 core requirements:
  1. Genuine enrolled-person match.
  2. Unknown person resembling an enrolled person.
  3. Two candidate identities with nearly identical scores (Ambiguity margin rejection).
  4. Low-quality or tiny face crop rejection.
  5. All five enrollment poses and unseen intermediate poses.
  6. Multiple faces in one frame.
  7. Contradictory identity observations across a track (Revocation).
  8. Two cameras producing overlapping track IDs (Multi-camera isolation).
  9. Concurrent detector inference thread-safety.
  10. Duplicate attendance-event prevention and direction tracking.
"""

import os
import sys
import time
import threading
import numpy as np
import cv2

# Ensure project root is in sys.path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import config
from pipeline.gallery import FaceGallery, calibrate_confidence_score
from pipeline.detector import BlazeFaceDetector
from pipeline.preprocessor import FacePreprocessor
from pipeline.tracker import FaceTracker, TrackedFace
from pipeline.attendance import AttendanceManager


def generate_orthogonal_vector(base_vec, similarity):
    """Generate a unit vector with exact cosine similarity to base_vec."""
    base_u = base_vec / np.linalg.norm(base_vec)
    rand = np.random.randn(len(base_vec))
    ortho = rand - np.dot(rand, base_u) * base_u
    ortho_u = ortho / np.linalg.norm(ortho)
    target = similarity * base_u + np.sqrt(max(0.0, 1.0 - similarity ** 2)) * ortho_u
    return (target / np.linalg.norm(target)).astype(np.float32)


def test_1_genuine_enrolled_match():
    """1. A genuine enrolled-person match exceeds thresholds and produces high confidence."""
    print("Running Test 1: Genuine enrolled-person match...")
    gallery = FaceGallery(similarity_threshold=config.MIN_RECOGNITION_SIMILARITY)
    np.random.seed(42)
    alice_front = np.random.randn(512).astype(np.float32)
    alice_front /= np.linalg.norm(alice_front)
    gallery.enroll("Alice", alice_front, pose="Front")

    # Query with high similarity (0.85) to Alice
    query_emb = generate_orthogonal_vector(alice_front, 0.85)
    name, conf, sim, pose = gallery.identify(query_emb)

    assert name == "Alice", f"Expected 'Alice', got '{name}'"
    assert sim >= 0.80, f"Expected similarity >= 0.80, got {sim:.4f}"
    assert conf >= config.CONFIDENCE_THRESHOLD_GREEN, f"Expected Green confidence (>=90%), got {conf:.1f}%"
    print(f"  ✓ Passed: Matched '{name}' with {conf:.1f}% confidence (similarity={sim:.4f})")


def test_2_unknown_person_resembling_enrolled():
    """2. An unknown person resembling an enrolled person is rejected."""
    print("Running Test 2: Unknown person resembling enrolled person...")
    gallery = FaceGallery(similarity_threshold=config.MIN_RECOGNITION_SIMILARITY)
    np.random.seed(101)
    alice_front = np.random.randn(512).astype(np.float32)
    alice_front /= np.linalg.norm(alice_front)
    gallery.enroll("Alice", alice_front, pose="Front")

    # Query with borderline impostor similarity (0.42 < 0.50 MIN_RECOGNITION_SIMILARITY)
    query_emb = generate_orthogonal_vector(alice_front, 0.42)
    name, conf, sim, pose = gallery.identify(query_emb)

    assert name == "Unknown", f"Expected 'Unknown', got '{name}'"
    assert conf < config.CONFIDENCE_THRESHOLD_YELLOW, f"Confidence {conf:.1f}% should be below 60%"
    print(f"  ✓ Passed: Correctly returned 'Unknown' for similarity {sim:.4f} (confidence={conf:.1f}%)")


def test_3_ambiguity_margin_rejection():
    """3. Two candidate identities with nearly identical scores are rejected for ambiguity."""
    print("Running Test 3: Best-vs-second-best ambiguity rejection...")
    gallery = FaceGallery(similarity_threshold=config.MIN_RECOGNITION_SIMILARITY)
    np.random.seed(202)

    # Base vector for query
    query_emb = np.random.randn(512).astype(np.float32)
    query_emb /= np.linalg.norm(query_emb)

    # Enroll Candidate A with 0.68 similarity to query
    alice_emb = generate_orthogonal_vector(query_emb, 0.68)
    gallery.enroll("Alice", alice_emb, pose="Front")

    # Enroll Candidate B with 0.67 similarity to query (margin = 0.01 < 0.08)
    bob_emb = generate_orthogonal_vector(query_emb, 0.67)
    gallery.enroll("Bob", bob_emb, pose="Front")

    detailed = gallery.identify_detailed(query_emb, min_margin=0.08)

    assert detailed["name"] == "Unknown", f"Expected 'Unknown' on ambiguous match, got '{detailed['name']}'"
    assert detailed["status"] == "REJECTED_AMBIGUOUS", f"Expected status 'REJECTED_AMBIGUOUS', got '{detailed['status']}'"
    assert detailed["margin"] < 0.08, f"Expected margin < 0.08, got {detailed['margin']:.4f}"
    print(f"  ✓ Passed: Ambiguity rejection triggered (Margin={detailed['margin']:.4f} < 0.0800)")


def test_4_low_quality_and_tiny_face():
    """4. Low-quality and tiny face crops are rejected."""
    print("Running Test 4: Low-quality or tiny face rejection...")
    preprocessor = FacePreprocessor(min_face_size=50)

    # Tiny frame with tiny face
    frame = np.full((200, 200, 3), 120, dtype=np.uint8)

    # Tiny bounding box: 20x20 pixels (< min_face_size 50)
    tiny_bbox = [50, 50, 70, 70]
    tensor, crop = preprocessor.extract_face(frame, tiny_bbox)
    assert tensor is None, "Expected tiny face crop to be rejected (tensor=None)"
    assert crop is None, "Expected tiny face crop to be rejected (crop=None)"

    aligned = preprocessor.align_face(frame, bbox=tiny_bbox)
    assert aligned is None, "Expected align_face to reject tiny face (None)"

    # Blurry crop test
    blurry_crop = np.full((112, 112, 3), 128, dtype=np.uint8)
    gray = cv2.cvtColor(blurry_crop, cv2.COLOR_BGR2GRAY)
    blur_score = float(cv2.Laplacian(gray, cv2.CV_64F).var())
    assert blur_score < config.MIN_BLUR_SCORE, f"Blur score {blur_score} should be < {config.MIN_BLUR_SCORE}"
    print(f"  ✓ Passed: Rejected tiny face (20x20) and detected blur score ({blur_score:.2f} < 45.0)")


def test_5_five_angle_and_intermediate_poses():
    """5. All five enrollment poses and unseen intermediate poses match correctly without internal competition."""
    print("Running Test 5: Five-angle enrollment and intermediate pose matching...")
    gallery = FaceGallery(similarity_threshold=config.MIN_RECOGNITION_SIMILARITY)
    np.random.seed(303)

    # Generate 5 angle templates for Alice
    alice_front = np.random.randn(512).astype(np.float32); alice_front /= np.linalg.norm(alice_front)
    alice_left = generate_orthogonal_vector(alice_front, 0.75)
    alice_right = generate_orthogonal_vector(alice_front, 0.74)
    alice_up = generate_orthogonal_vector(alice_front, 0.73)
    alice_down = generate_orthogonal_vector(alice_front, 0.72)

    gallery.enroll("Alice", alice_front, pose="Front")
    gallery.enroll("Alice", alice_left, pose="Left")
    gallery.enroll("Alice", alice_right, pose="Right")
    gallery.enroll("Alice", alice_up, pose="Up")
    gallery.enroll("Alice", alice_down, pose="Down")

    # Unseen intermediate pose (e.g., halfway between Front and Left)
    intermediate = (alice_front + alice_left) / 2.0
    intermediate /= np.linalg.norm(intermediate)

    detailed = gallery.identify_detailed(intermediate)
    assert detailed["name"] == "Alice", f"Expected 'Alice', got '{detailed['name']}'"
    assert detailed["is_accepted"] is True, "Expected accepted match for intermediate pose"
    # Verify that Alice's multiple templates do NOT compete with herself as second_best_name
    assert detailed["second_best_name"] is None, f"Alice's templates should not compete as distinct second identity: {detailed['second_best_name']}"
    print(f"  ✓ Passed: Intermediate pose matched '{detailed['name']}' ({detailed['confidence']:.1f}%), no self-competition")


def test_6_multiple_faces_in_one_frame():
    """6. Multiple faces in one frame receive distinct tracks."""
    print("Running Test 6: Multiple faces in one frame tracking...")
    tracker = FaceTracker(iou_threshold=0.3, max_disappeared=5)

    # Frame with 2 distinct people side by side
    dets_frame_1 = [
        {'bbox': [50, 50, 150, 150], 'score': 0.95},
        {'bbox': [300, 50, 400, 150], 'score': 0.93}
    ]
    assignments_1 = tracker.update(dets_frame_1)
    assert len(assignments_1) == 2, f"Expected 2 assignments, got {len(assignments_1)}"
    track_id_0 = assignments_1[0][1]
    track_id_1 = assignments_1[1][1]
    assert track_id_0 != track_id_1, f"Expected distinct track IDs, got {track_id_0} and {track_id_1}"

    # Frame 2: minor movement
    dets_frame_2 = [
        {'bbox': [52, 51, 152, 151], 'score': 0.94},
        {'bbox': [302, 51, 402, 151], 'score': 0.92}
    ]
    assignments_2 = tracker.update(dets_frame_2)
    assert assignments_2[0][1] == track_id_0, "Track 0 should maintain continuity"
    assert assignments_2[1][1] == track_id_1, "Track 1 should maintain continuity"
    print(f"  ✓ Passed: Distinct tracks #{track_id_0} and #{track_id_1} maintained across frames")


def test_7_contradictory_identity_revocation():
    """7. Contradictory identity observations across a track cause identity revocation."""
    print("Running Test 7: Contradictory identity observations and revocation...")
    tracked = TrackedFace(track_id=1, bbox=[100, 100, 200, 200])

    # 3 consistent observations of Alice confirm identity
    tracked.add_observation("Alice", similarity=0.85, confidence=95.0)
    time.sleep(0.06)
    tracked.update_bbox([100, 106, 200, 206])
    tracked.add_observation("Alice", similarity=0.86, confidence=96.0)
    time.sleep(0.06)
    tracked.update_bbox([100, 112, 200, 212])
    tracked.add_observation("Alice", similarity=0.84, confidence=94.0)

    assert tracked.is_confirmed is True, "Track should be confirmed as Alice after 3 observations"
    assert tracked.confirmed_name == "Alice"

    # Conflicting observations of Bob occur
    time.sleep(0.06)
    tracked.update_bbox([100, 118, 200, 218])
    tracked.add_observation("Bob", similarity=0.88, confidence=95.0)

    time.sleep(0.06)
    tracked.update_bbox([100, 124, 200, 224])
    tracked.add_observation("Bob", similarity=0.89, confidence=96.0)

    # Persistent contradictions must revoke Alice confirmation!
    assert tracked.confirmed_name != "Alice", f"Confirmed name should no longer be Alice: got {tracked.confirmed_name}"
    print(f"  ✓ Passed: Alice identity revoked upon persistent contradictions (current: '{tracked.name}')")


def test_8_multi_camera_track_isolation():
    """8. Two cameras producing overlapping track IDs are strictly isolated."""
    print("Running Test 8: Multi-camera track isolation...")
    att_mgr = AttendanceManager(min_confidence=80.0, min_stable_frames=1, cooldown_seconds=300)

    # Camera 0, Track #1 represents Alice
    event_cam0 = att_mgr.process_observation(
        track_id=1,
        name="Alice",
        confidence=92.0,
        similarity=0.85,
        camera_id=0,
        is_confirmed=True,
        direction="Entry"
    )
    assert event_cam0 is not None, "Camera 0 Track #1 should log attendance for Alice"

    # Camera 1, Track #1 represents Bob (identical track_id, different camera_id)
    event_cam1 = att_mgr.process_observation(
        track_id=1,
        name="Bob",
        confidence=91.0,
        similarity=0.84,
        camera_id=1,
        is_confirmed=True,
        direction="Entry"
    )
    assert event_cam1 is not None, "Camera 1 Track #1 should NOT be blocked by Camera 0 Track #1"
    assert event_cam1["name"] == "Bob"
    assert event_cam1["camera_id"] == 1
    print(f"  ✓ Passed: Camera 0 (Alice) and Camera 1 (Bob) isolated on track_id=1")


def test_9_concurrent_detector_inference():
    """9. Concurrent detector inference across multiple worker threads is thread-safe."""
    print("Running Test 9: Concurrent detector inference thread-safety...")
    detector = BlazeFaceDetector(config.BLAZEFACE_MODEL_PATH, confidence_threshold=0.45)
    test_frame = np.full((480, 640, 3), 120, dtype=np.uint8)

    errors = []

    def worker_run(thread_id):
        try:
            for _ in range(5):
                dets = detector.detect(test_frame)
                assert isinstance(dets, list)
        except Exception as e:
            errors.append((thread_id, e))

    threads = [threading.Thread(target=worker_run, args=(i,)) for i in range(4)]
    for t in threads:
        t.start()
    for t in threads:
        t.join(timeout=10.0)

    assert len(errors) == 0, f"Thread-safety errors encountered: {errors}"
    print(f"  ✓ Passed: 4 concurrent threads completed 20 inferences without error")


def test_10_duplicate_attendance_prevention():
    """10. Duplicate attendance-event prevention and direction tracking."""
    print("Running Test 10: Duplicate attendance prevention and direction tracking...")
    att_mgr = AttendanceManager(min_confidence=80.0, min_stable_frames=1, cooldown_seconds=10)

    # 1. First crossing: Event logged
    ev1 = att_mgr.process_observation(
        track_id=5,
        name="Alice",
        confidence=93.0,
        similarity=0.86,
        camera_id=0,
        is_confirmed=True,
        direction="Entry"
    )
    assert ev1 is not None, "First observation should log attendance"

    # 2. In-Track suppression: repeated frame for same track_id
    ev2 = att_mgr.process_observation(
        track_id=5,
        name="Alice",
        confidence=94.0,
        similarity=0.87,
        camera_id=0,
        is_confirmed=True,
        direction="Entry"
    )
    assert ev2 is None, "In-track duplicate should be suppressed"

    # 3. Cooldown suppression: new track_id for same person within cooldown
    ev3 = att_mgr.process_observation(
        track_id=6,
        name="Alice",
        confidence=93.5,
        similarity=0.86,
        camera_id=0,
        is_confirmed=True,
        direction="Entry"
    )
    assert ev3 is None, "Cooldown duplicate should be suppressed"

    # 4. Opposite direction crossing (Exit) is permitted
    ev4 = att_mgr.process_observation(
        track_id=7,
        name="Alice",
        confidence=93.5,
        similarity=0.86,
        camera_id=0,
        is_confirmed=True,
        direction="Exit"
    )
    assert ev4 is not None, "Opposite direction crossing (Exit) should be tracked"
    assert ev4["direction"] == "Exit"
    print(f"  ✓ Passed: In-track duplicate suppressed, cooldown enforced, direction Exit tracked")


def test_11_unknown_visitor_logging():
    """11. Unknown visitors are correctly labeled Unknown-001... and logged to attendance."""
    print("Running Test 11: Unknown visitor Unknown-001 logging and employee upgrade...")
    att_mgr = AttendanceManager(min_confidence=80.0, min_stable_frames=3, cooldown_seconds=60)
    req_frames = getattr(config, 'UNKNOWN_MIN_STABLE_FRAMES', 8)

    # Accumulate frames up to req_frames - 1
    for _ in range(req_frames - 1):
        ev = att_mgr.process_observation(track_id=0, name="Unknown-001", confidence=0.0, camera_id=0, is_confirmed=False)
        assert ev is None, "Under-threshold observations should not trigger immediate visitor event"

    # Frame req_frames: reaches stability threshold
    ev_logged = att_mgr.process_observation(track_id=0, name="Unknown-001", confidence=0.0, camera_id=0, is_confirmed=False)
    assert ev_logged is not None, f"Frame {req_frames} should log Unknown-001 visitor attendance"
    assert ev_logged["name"] == "Unknown-001"
    assert ev_logged["status"] == "UNREGISTERED_VISITOR"

    # Subsequent employee confirmation upgrades the event
    ev_upgraded = att_mgr.process_observation(track_id=0, name="Alice", confidence=92.0, similarity=0.85, camera_id=0, is_confirmed=True)
    assert ev_upgraded is not None, "Confirmed employee observation should upgrade visitor event"
    assert ev_upgraded["name"] == "Alice"
    assert ev_upgraded["status"] == "CONFIRMED"
    print(f"  ✓ Passed: Correctly logged '{ev_logged['name']}' (Visitor) and upgraded to '{ev_upgraded['name']}' (Confirmed)")


def test_12_attendance_screenshot_logging():
    """12. Screenshot capture and persistence to Log/ folder upon attendance logging and upgrade."""
    print("Running Test 12: Attendance screenshot storage in Log/ folder...")
    test_logs_dir = os.path.join(config.BASE_DIR, "Log")
    os.makedirs(test_logs_dir, exist_ok=True)
    att_mgr = AttendanceManager(min_confidence=85.0, min_stable_frames=2, logs_dir=test_logs_dir)

    dummy_frame = np.zeros((480, 640, 3), dtype=np.uint8)
    dummy_frame[:] = (40, 40, 50)
    dummy_bbox = [100, 100, 250, 250]

    # Frame 1: under stability threshold
    att_mgr.process_observation(track_id=42, name="Anoop Verma", confidence=91.0, similarity=0.82, frame=dummy_frame, bbox=dummy_bbox)
    # Frame 2: triggers attendance event
    ev = att_mgr.process_observation(track_id=42, name="Anoop Verma", confidence=91.0, similarity=0.82, frame=dummy_frame, bbox=dummy_bbox)

    assert ev is not None, "Expected attendance event to be logged"
    assert "screenshot" in ev and ev["screenshot"] is not None, "Event must include 'screenshot' filename"
    assert "screenshot_url" in ev and ev["screenshot_url"].startswith("/Log/"), "Event must include '/Log/' URL"

    saved_path = os.path.join(test_logs_dir, ev["screenshot"])
    assert os.path.exists(saved_path), f"Screenshot file '{saved_path}' must exist on disk"
    assert os.path.getsize(saved_path) > 1000, "Screenshot file size must be non-trivial"

    read_img = cv2.imread(saved_path)
    assert read_img is not None and read_img.shape == (480, 640, 3), "Saved screenshot must be readable as an image"

    print(f"  ✓ Passed: Screenshot '{ev['screenshot']}' successfully stored in Log folder ({os.path.getsize(saved_path)} bytes)")


def run_all_tests():
    print("=" * 70)
    print("  RUNNING ALL AUTOMATED CRITICAL BUG FIX & VISITOR TESTS")
    print("=" * 70)
    t0 = time.time()

    test_1_genuine_enrolled_match()
    test_2_unknown_person_resembling_enrolled()
    test_3_ambiguity_margin_rejection()
    test_4_low_quality_and_tiny_face()
    test_5_five_angle_and_intermediate_poses()
    test_6_multiple_faces_in_one_frame()
    test_7_contradictory_identity_revocation()
    test_8_multi_camera_track_isolation()
    test_9_concurrent_detector_inference()
    test_10_duplicate_attendance_prevention()
    test_11_unknown_visitor_logging()
    test_12_attendance_screenshot_logging()

    duration = time.time() - t0
    print("=" * 70)
    print(f"  ALL 12 TESTS PASSED SUCCESSFULLY in {duration:.2f}s!")
    print("=" * 70)


if __name__ == "__main__":
    run_all_tests()
