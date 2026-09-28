"""Shared "detect, retry with fallback on total failure" policy.

`analyse_pair_with_fallback` (orchestrator.py) applies this idea at the
pair-comparison level, restarting the whole comparison on `NoFaceError`. The
live per-photo check (`/api/face-detect`) and the database-search index
builder both need the same policy at the single-image level: try the primary
backend, and only if it finds literally zero faces, retry with the OpenCV
backend before giving up. Factored out so the call sites cannot drift apart.
"""

from __future__ import annotations

import logging

import numpy as np

from app.config import Settings
from app.models.base import DetectedFace, FaceEmbedder

logger = logging.getLogger(__name__)


def detect_with_fallback(
    embedder: FaceEmbedder, image_bgr: np.ndarray, settings: Settings
) -> tuple[list[DetectedFace], FaceEmbedder, bool]:
    """Detect faces, retrying once with the OpenCV backend on zero detections.

    Returns `(faces, backend_used, used_fallback)`. `backend_used` is
    `embedder` itself unless the fallback found faces the primary could not -
    callers that go on to call `.embed()` must use it, not `embedder`, or the
    resulting vector will silently belong to the wrong backend's space.
    """
    faces = embedder.detect(image_bgr, max_faces=settings.max_faces_returned)
    if faces or not settings.enable_detection_fallback or embedder.info.backend == "opencv":
        return faces, embedder, False

    from app.models.registry import get_fallback_embedder

    fallback_embedder = get_fallback_embedder(settings)
    faces = fallback_embedder.detect(image_bgr, max_faces=settings.max_faces_returned)
    if faces:
        logger.info(
            "Primary backend (%s) detected no face; fallback %s backend succeeded.",
            embedder.info.backend,
            fallback_embedder.info.backend,
        )
        return faces, fallback_embedder, True
    return [], embedder, False
