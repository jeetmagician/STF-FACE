"""Backend registry and lazy singleton loader."""

from __future__ import annotations

import logging
import threading

from app.config import Settings, get_settings
from app.models.base import FaceEmbedder

logger = logging.getLogger(__name__)

_embedder: FaceEmbedder | None = None
_load_error: str | None = None
_lock = threading.Lock()

_fallback_embedder: FaceEmbedder | None = None
_fallback_lock = threading.Lock()


def _build(settings: Settings) -> FaceEmbedder:
    backend = settings.model_backend

    if backend == "insightface":
        from app.models.insightface_backend import InsightFaceEmbedder

        return InsightFaceEmbedder(
            model_dir=settings.model_dir,
            pack=settings.insightface_pack,
            det_size=settings.insightface_det_size,
            detection_threshold=settings.detection_confidence_threshold,
            use_gpu=settings.use_gpu,
        )

    if backend == "opencv":
        from app.models.opencv_backend import OpenCVEmbedder

        return OpenCVEmbedder(
            model_dir=settings.model_dir,
            detection_threshold=settings.detection_confidence_threshold,
            use_gpu=settings.use_gpu,
        )

    # auto: prefer InsightFace if it is genuinely available, else OpenCV.
    try:
        from app.models.insightface_backend import InsightFaceEmbedder

        embedder = InsightFaceEmbedder(
            model_dir=settings.model_dir,
            pack=settings.insightface_pack,
            det_size=settings.insightface_det_size,
            detection_threshold=settings.detection_confidence_threshold,
            use_gpu=settings.use_gpu,
        )
        logger.warning(
            "MODEL_BACKEND=auto selected InsightFace. Its pretrained weights are "
            "licensed for non-commercial research use only. Set "
            "MODEL_BACKEND=opencv for a commercially licensable stack."
        )
        return embedder
    except Exception as exc:
        logger.info("InsightFace unavailable (%s); falling back to OpenCV backend.", exc)

    from app.models.opencv_backend import OpenCVEmbedder

    return OpenCVEmbedder(
        model_dir=settings.model_dir,
        detection_threshold=settings.detection_confidence_threshold,
        use_gpu=settings.use_gpu,
    )


def get_embedder(settings: Settings | None = None) -> FaceEmbedder:
    """Return the process-wide embedder, constructing it on first use."""
    global _embedder, _load_error
    if _embedder is not None:
        return _embedder

    with _lock:
        if _embedder is not None:
            return _embedder
        settings = settings or get_settings()
        try:
            _embedder = _build(settings)
            _load_error = None
            logger.info("Loaded face model backend: %s", _embedder.info.backend)
        except Exception as exc:
            _load_error = str(exc)
            raise
        return _embedder


def get_fallback_embedder(settings: Settings) -> FaceEmbedder:
    """A permissively-licensed, always-available embedder used only as an
    automatic retry when the primary backend fails to detect a face at all.

    Always OpenCV/YuNet, regardless of the primary MODEL_BACKEND setting -
    it has no licence restriction and has proven reliable at detection across
    every test image tried so far.
    """
    global _fallback_embedder
    if _fallback_embedder is not None:
        return _fallback_embedder
    with _fallback_lock:
        if _fallback_embedder is not None:
            return _fallback_embedder
        from app.models.opencv_backend import OpenCVEmbedder

        _fallback_embedder = OpenCVEmbedder(
            model_dir=settings.model_dir,
            detection_threshold=settings.detection_confidence_threshold,
            use_gpu=settings.use_gpu,
        )
        logger.info("Fallback face model backend ready: opencv")
        return _fallback_embedder


def get_load_error() -> str | None:
    return _load_error


def is_loaded() -> bool:
    return _embedder is not None


def reset() -> None:
    """Drop the cached embedder(s). Used by tests."""
    global _embedder, _load_error, _fallback_embedder
    with _lock:
        _embedder = None
        _load_error = None
    with _fallback_lock:
        _fallback_embedder = None
