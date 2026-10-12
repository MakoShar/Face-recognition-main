"""
Face Recognition Pipeline Package.

Modules:
    detector    — BlazeFace TFLite face detection
    preprocessor — Face crop extraction and normalization
    embedder    — MobileFaceNet ONNX face embedding
    gallery     — Identity enrollment and matching
    tracker     — Frame-to-frame face tracking
    camera      — Camera capture with frame sampling
    engine      — Full pipeline orchestrator
"""

from pipeline.detector import BlazeFaceDetector
from pipeline.preprocessor import FacePreprocessor
from pipeline.embedder import FaceEmbedder
from pipeline.gallery import FaceGallery
from pipeline.tracker import FaceTracker
from pipeline.camera import CameraStream
from pipeline.engine import RecognitionEngine

__all__ = [
    "BlazeFaceDetector",
    "FacePreprocessor",
    "FaceEmbedder",
    "FaceGallery",
    "FaceTracker",
    "CameraStream",
    "RecognitionEngine",
]
