"""Abstract model interface.

Swapping the face-recognition model means implementing `FaceEmbedder` and
registering it. Nothing outside `app/models/` needs to change.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field

import numpy as np

# Canonical 5-point template for a 112x112 aligned face crop (ArcFace
# convention: left eye, right eye, nose tip, left mouth corner, right mouth
# corner). Both backends align to this same template so that region crops and
# downstream geometry are directly comparable across backends.
ARCFACE_112_TEMPLATE = np.array(
    [
        [38.2946, 51.6963],
        [73.5318, 51.5014],
        [56.0252, 71.7366],
        [41.5493, 92.3655],
        [70.7299, 92.2041],
    ],
    dtype=np.float32,
)


@dataclass
class DetectedFace:
    """One detected face, with everything downstream stages need."""

    index: int
    bbox: tuple[float, float, float, float]  # x1, y1, x2, y2 in source pixels
    detection_score: float
    keypoints_5: np.ndarray  # (5, 2) in source pixels
    aligned: np.ndarray  # (112, 112, 3) BGR, similarity-transformed
    landmarks: np.ndarray | None = None  # (N, 2) in source pixels, N=106 or 68
    landmarks_aligned: np.ndarray | None = None  # same points in aligned space
    pose: tuple[float, float, float] | None = None  # yaw, pitch, roll degrees
    embedding: np.ndarray | None = None  # L2-normalised

    @property
    def width(self) -> float:
        return self.bbox[2] - self.bbox[0]

    @property
    def height(self) -> float:
        return self.bbox[3] - self.bbox[1]


@dataclass
class ModelInfo:
    backend: str
    detector: str
    recognizer: str
    embedding_dim: int
    license: str
    commercial_use: bool
    notes: str = ""
    extras: dict[str, str] = field(default_factory=dict)


class FaceEmbedder(ABC):
    """Contract every recognition backend must satisfy."""

    @property
    @abstractmethod
    def info(self) -> ModelInfo: ...

    @abstractmethod
    def detect(self, image_bgr: np.ndarray, max_faces: int) -> list[DetectedFace]:
        """Detect faces, align them, and attach landmarks + pose.

        Returns faces sorted by bounding-box area, largest first. Does not
        compute embeddings - call `embed` for that, so callers can detect
        cheaply without paying for recognition.
        """

    @abstractmethod
    def embed(self, aligned_bgr: np.ndarray) -> np.ndarray:
        """Embed a single aligned 112x112 BGR crop. Returns an L2-normalised vector."""

    def embed_batch(self, aligned_batch: list[np.ndarray]) -> np.ndarray:
        """Embed several aligned crops. Override where a backend supports batching."""
        if not aligned_batch:
            return np.zeros((0, self.info.embedding_dim), dtype=np.float32)
        return np.stack([self.embed(a) for a in aligned_batch])

    def warmup(self) -> None:
        """Run one dummy inference so the first real request isn't slow."""
        dummy = np.zeros((112, 112, 3), dtype=np.uint8)
        try:
            self.embed(dummy)
        except Exception:  # pragma: no cover - warmup is best-effort
            pass


def l2_normalise(vector: np.ndarray) -> np.ndarray:
    norm = float(np.linalg.norm(vector))
    if norm < 1e-10:
        return vector.astype(np.float32)
    return (vector / norm).astype(np.float32)
