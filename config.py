"""
Configuration settings for the Real-Time Multi-Camera Face Recognition Pipeline.
Centralizes paths, thresholds, Section 16 display rules, and performance settings.
"""

import os

# ─── Paths ────────────────────────────────────────────────────────────────────
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Models directory
MODELS_DIR = os.path.join(BASE_DIR, "models")

# Detector path
BLAZEFACE_MODEL_PATH = os.path.join(MODELS_DIR, "detection", "blaze_face_full_range_sparse.tflite")

# Head Pose Angle model (ResNet-50 6D rotation matrix)
HEAD_POSE_MODEL_PATH = os.path.join(MODELS_DIR, "pose", "resnet50.onnx")

# Recognition candidate models
GLINTR100_MODEL_PATH = os.path.join(MODELS_DIR, "embeddings", "glintr100.onnx")
ADAFACE_MODEL_PATH = os.path.join(MODELS_DIR, "embeddings", "adaface_ir101.onnx")
ADAFACE_CKPT_PATH = os.path.join(MODELS_DIR, "embeddings", "adaface_ir101_webface12m.ckpt")

# Active recognition model ("glintr100" or "adaface")
ACTIVE_MODEL = "glintr100"
EMBEDDING_MODEL_PATH = GLINTR100_MODEL_PATH if ACTIVE_MODEL == "glintr100" else ADAFACE_MODEL_PATH

# ─── Extensible Model Registry ───────────────────────────────────────────────
# Easily add or compare additional models by registering them here
AVAILABLE_MODELS = {
    "glintr100": {
        "id": "glintr100",
        "name": "GlintR100 (ResNet100)",
        "path": GLINTR100_MODEL_PATH,
        "type": "glintr100",
        "description": "ArcFace ResNet100 trained on Glint360K",
    },
    "adaface": {
        "id": "adaface",
        "name": "AdaFace IR-101 (WebFace12M)",
        "path": ADAFACE_MODEL_PATH,
        "type": "adaface",
        "description": "Adaptive Margin Loss IR-101 on WebFace12M",
    },
}


def get_model_config(model_id):
    """Retrieve model configuration dictionary by ID."""
    m_id = model_id.lower()
    if m_id in AVAILABLE_MODELS:
        return AVAILABLE_MODELS[m_id]
    raise ValueError(f"Model '{m_id}' not found in AVAILABLE_MODELS: {list(AVAILABLE_MODELS.keys())}")


def get_gallery_cache_path(model_id):
    """Path to the cached embeddings file for a specific model."""
    specific = os.path.join(BASE_DIR, "gallery", f"embeddings_{model_id.lower()}.json")
    if os.path.exists(specific):
        return specific
    return GALLERY_EMBEDDINGS_PATH


# Data & Gallery paths
DATA_DIR = os.path.join(BASE_DIR, "data")
GALLERY_VIDEOS_DIR = os.path.join(DATA_DIR, "raw_videos")
GALLERY_IMAGES_DIR = os.path.join(DATA_DIR, "images")
GALLERY_EMBEDDINGS_PATH = os.path.join(BASE_DIR, "gallery", "embeddings.json")
TESTS_DIR = os.path.join(BASE_DIR, "tests")
TEST_IMAGES_DIR = os.path.join(TESTS_DIR, "test_data")
BENCHMARK_RESULTS_PATH = os.path.join(TESTS_DIR, "benchmark_results.json")

# ─── BlazeFace Detection Settings ────────────────────────────────────────────
BLAZEFACE_INPUT_SIZE = 192                # Model expects 192x192 input
BLAZEFACE_CONFIDENCE_THRESHOLD = 0.45     # Calibrated to detect non-frontal poses (Right, Up, Down)
BLAZEFACE_NMS_IOU_THRESHOLD = 0.3         # Non-max suppression IoU threshold
BLAZEFACE_NUM_ANCHORS = 2304              # Number of anchor boxes
BLAZEFACE_NUM_KEYPOINTS = 6               # 6 facial keypoints per detection

# ─── Face Embedding Settings ─────────────────────────────────────────────────
EMBEDDING_INPUT_SIZE = 112                # Both GlintR100 and AdaFace expect 112x112
EMBEDDING_DIMENSION = 512                 # 512-dimensional L2-normalized embedding
EMBEDDING_MODEL_NAME = ACTIVE_MODEL

# ─── Section 16: Recognition Confidence and Dashboard Color Rules ─────────────
# Thresholds:
#   >= 90%         -> GREEN:  Display name and confidence percentage
#   60% to < 90%   -> YELLOW: Display name and confidence percentage
#   < 60%          -> NO IDENTITY LABEL: Treat as Unknown / Uncertain
CONFIDENCE_THRESHOLD_GREEN = 90.0         # 90% and higher
CONFIDENCE_THRESHOLD_YELLOW = 60.0        # 60% to below 90%

# Color palette (BGR format for OpenCV)
COLOR_GREEN = (72, 199, 142)              # Vibrant modern green
COLOR_YELLOW = (40, 190, 245)             # Amber / warm yellow
COLOR_UNKNOWN = (60, 76, 231)             # Muted red / neutral

# ─── Gallery / Matching Settings ─────────────────────────────────────────────
SIMILARITY_METRIC = "cosine"              # "cosine"
SIMILARITY_MATCH_BOUNDARY = 0.35          # Raw cosine similarity corresponding to 60% confidence
MAX_REFS_PER_POSE = 3                     # Reference samples per pose category
MAX_TOTAL_REFS_PER_IDENTITY = 15          # Max reference embeddings per person

# Pose categories
POSE_CATEGORIES = ["Front", "Left", "Right", "Up", "Down"]

# ─── Camera & Frame Sampling ─────────────────────────────────────────────────
# Single primary video source (/dev/video0)
DEFAULT_CAMERA_IDS = [0]
CAMERA_FPS = 30                           # Expected camera FPS
FRAME_SAMPLE_INTERVAL = 5                 # Nominal rule: Process every 5th frame (~6 processed FPS)
FRAME_QUEUE_MAXSIZE = 2                   # Bounded queue to prevent stale frame accumulation
CAMERA_RECONNECT_DELAY = 3.0             # Seconds to wait before reconnecting
CAMERA_MAX_RECONNECT_ATTEMPTS = 10        # Max reconnection attempts

# ─── Tracker Settings ────────────────────────────────────────────────────────
TRACKER_ENABLED = True                    # Frame-to-frame tracking
TRACKER_IOU_THRESHOLD = 0.3              # Min IoU to associate tracked faces
TRACKER_MAX_DISAPPEARED = 15              # Frames before dropping a tracked face
TRACKER_RECOGNITION_INTERVAL = 10         # Re-run recognition every N frames

# ─── Attendance & Event Suppression Settings ─────────────────────────────────
ATTENDANCE_ENABLED = True
ATTENDANCE_CONFIDENCE_THRESHOLD = 85.0     # Minimum confidence % to record attendance
ATTENDANCE_MIN_FRAMES = 3                  # Minimum stable observations in track before logging
ATTENDANCE_COOLDOWN_SECONDS = 300          # 5 minutes suppression window for duplicate check-ins
ATTENDANCE_LOG_PATH = os.path.join(DATA_DIR, "attendance.json")

# ─── Performance Settings ────────────────────────────────────────────────────
ONNX_NUM_THREADS = 4                      # ONNX Runtime intra-op threads
ONNX_EXECUTION_PROVIDERS = ["CUDAExecutionProvider", "CPUExecutionProvider"]

# ─── Flask / Web Settings ────────────────────────────────────────────────────
FLASK_HOST = "0.0.0.0"
FLASK_PORT = 5000
FLASK_DEBUG = False
MJPEG_QUALITY = 80                        # JPEG quality for MJPEG stream (1-100)
MJPEG_MAX_FPS = 15                        # Max FPS for browser stream

# ─── Bounding Box & Annotation Styling ───────────────────────────────────────
BBOX_THICKNESS = 2
LABEL_FONT_SCALE = 0.65
LABEL_FONT_THICKNESS = 2
LABEL_BG_PADDING = 5

# ─── Face Crop Validation ────────────────────────────────────────────────────
MIN_FACE_SIZE = 30                        # Minimum face crop dimension (pixels)
MIN_BLUR_SCORE = 45.0                     # Minimum Laplacian sharpness score
