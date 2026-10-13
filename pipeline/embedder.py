"""
Face Embedding Module — Multi-Model ONNX Inference.

Supports both:
  1. GlintR100 (ArcFace ResNet100 from AntelopeV2)
  2. AdaFace IR-101 (Trained on WebFace12M)

Uses ONNX Runtime with model-specific channel ordering, normalization,
and L2-normalized 512-dimensional feature embedding vectors.
"""

import os
import cv2
import numpy as np
import logging

logger = logging.getLogger(__name__)


def setup_cuda_environment():
    """
    Ensure NVIDIA CUDA and cuDNN shared libraries from pip packages
    are registered and preloaded so ONNX Runtime CUDAExecutionProvider
    loads cleanly without missing shared library errors on Linux.
    """
    import site
    import glob
    import ctypes

    nvidia_paths = []
    for p in site.getsitepackages() + [site.getusersitepackages()]:
        pattern = os.path.join(p, "nvidia", "*", "lib")
        nvidia_paths.extend(glob.glob(pattern))

    for p in nvidia_paths:
        if not os.path.isdir(p):
            continue
        cur_ld = os.environ.get("LD_LIBRARY_PATH", "")
        if p not in cur_ld:
            os.environ["LD_LIBRARY_PATH"] = f"{p}:{cur_ld}" if cur_ld else p
        for fname in sorted(os.listdir(p)):
            if fname.endswith(".so") or ".so." in fname:
                try:
                    ctypes.CDLL(os.path.join(p, fname), mode=ctypes.RTLD_GLOBAL)
                except Exception:
                    pass


# Ensure CUDA/cuDNN libraries are available before initializing ONNX Runtime
setup_cuda_environment()
import onnxruntime as ort  # noqa: E402


class FaceEmbedder:
    """
    Face embedding generator supporting both GlintR100 and AdaFace IR-101.
    """

    def __init__(self, model_path, model_type="auto", num_threads=4, providers=None):
        """
        Args:
            model_path: Path to ONNX model file
            model_type: "glintr100", "adaface", or "auto" (inferred from path)
            num_threads: Number of ONNX Runtime intra-op threads
            providers: Execution providers (default: from config.ONNX_EXECUTION_PROVIDERS)
        """
        self.model_path = model_path
        if providers is None:
            try:
                import config
                providers = config.ONNX_EXECUTION_PROVIDERS
            except Exception:
                providers = ["CUDAExecutionProvider", "CPUExecutionProvider"]

        # Infer model type
        if model_type == "auto":
            fname = os.path.basename(model_path).lower()
            if "adaface" in fname or "webface" in fname:
                self.model_type = "adaface"
            else:
                self.model_type = "glintr100"
        else:
            self.model_type = model_type.lower()

        # Configure session options
        sess_options = ort.SessionOptions()
        sess_options.intra_op_num_threads = num_threads
        sess_options.inter_op_num_threads = num_threads
        sess_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL

        # Load ONNX session
        self.session = ort.InferenceSession(
            model_path,
            sess_options=sess_options,
            providers=providers
        )

        self.input_name = self.session.get_inputs()[0].name
        self.output_name = self.session.get_outputs()[0].name
        self.input_shape = self.session.get_inputs()[0].shape
        self.embedding_dim = self.session.get_outputs()[0].shape[1]

        logger.info(
            f"FaceEmbedder initialized: type={self.model_type}, "
            f"input={self.input_name} {self.input_shape}, "
            f"output={self.output_name} dim={self.embedding_dim}, "
            f"active_providers={self.session.get_providers()}"
        )

    def preprocess_face_crop(self, face_bgr, target_size=112):
        """
        Convert raw BGR face crop into model-specific input tensor.

        Model requirements:
          - GlintR100: RGB, normalized to [-1.0, 1.0], shape (1, 3, 112, 112)
          - AdaFace:   BGR, normalized to [-1.0, 1.0], shape (1, 3, 112, 112)

        Args:
            face_bgr: Raw cropped face BGR image (H, W, 3)
            target_size: Target dimension (default: 112)

        Returns:
            tensor: (1, 3, 112, 112) float32 array
        """
        if face_bgr is None or face_bgr.size == 0:
            return None

        # Resize to 112x112
        resized = cv2.resize(face_bgr, (target_size, target_size), interpolation=cv2.INTER_LINEAR)

        # Channel order
        if self.model_type == "glintr100":
            # RGB
            img = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
        else:
            # AdaFace expects BGR channel order
            img = resized

        # Normalize to [-1.0, 1.0]
        norm = (img.astype(np.float32) / 127.5) - 1.0

        # Transpose HWC -> CHW and add batch dimension NCHW
        tensor = np.transpose(norm, (2, 0, 1))
        return np.expand_dims(tensor, axis=0).astype(np.float32)

    def get_embedding(self, face_input):
        """
        Generate embedding for a single face.

        Args:
            face_input: Either (1, 3, 112, 112) tensor OR raw BGR crop

        Returns:
            embedding: (512,) float32 L2-normalized vector
        """
        if face_input is None:
            return None

        # If passed raw BGR crop (3-dimensional HWC)
        if len(face_input.shape) == 3:
            tensor = self.preprocess_face_crop(face_input)
        else:
            tensor = face_input

        if tensor is None:
            return None

        # Run inference
        result = self.session.run(
            [self.output_name],
            {self.input_name: tensor}
        )
        embedding = result[0][0]  # Shape: (512,)

        # L2 normalize
        norm = np.linalg.norm(embedding)
        if norm > 0:
            embedding = embedding / norm

        return embedding.astype(np.float32)

    def get_embeddings_batch(self, face_inputs):
        """
        Generate embeddings for a batch of faces.

        Args:
            face_inputs: List of tensors or list of BGR crops

        Returns:
            embeddings: List of (512,) float32 L2-normalized vectors
        """
        if not face_inputs:
            return []

        tensors = []
        for item in face_inputs:
            if len(item.shape) == 3:
                t = self.preprocess_face_crop(item)
            else:
                t = item
            if t is not None:
                tensors.append(t)

        if not tensors:
            return []

        try:
            batch = np.concatenate(tensors, axis=0)  # (N, 3, 112, 112)
            results = self.session.run(
                [self.output_name],
                {self.input_name: batch}
            )
            raw_embeddings = results[0]  # (N, 512)

            norms = np.linalg.norm(raw_embeddings, axis=1, keepdims=True)
            norms = np.maximum(norms, 1e-10)
            normalized = raw_embeddings / norms
            return [normalized[i].astype(np.float32) for i in range(len(tensors))]

        except Exception as e:
            logger.debug(f"Batch inference fallback ({e})")
            return [self.get_embedding(t) for t in tensors]
