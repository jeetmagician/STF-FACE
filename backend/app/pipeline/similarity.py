"""Similarity computation and multi-photo template aggregation.

The multi-photo design is live, not merely "architected for later": the
analyse endpoint accepts lists on both sides from the outset, and a single
photograph is simply a template of size one.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from app.models.base import DetectedFace
from app.pipeline.quality import QualityReport


def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    """Cosine similarity between two embeddings.

    Both backends return L2-normalised vectors, so this is a dot product; the
    explicit normalisation keeps the function correct for any future backend
    that does not normalise.
    """
    a = np.asarray(a, dtype=np.float64).ravel()
    b = np.asarray(b, dtype=np.float64).ravel()
    norm_a = float(np.linalg.norm(a))
    norm_b = float(np.linalg.norm(b))
    if norm_a < 1e-10 or norm_b < 1e-10:
        return 0.0
    return float(np.clip(np.dot(a, b) / (norm_a * norm_b), -1.0, 1.0))


@dataclass
class Template:
    """A quality-weighted aggregate of one or more embeddings of one subject."""

    embedding: np.ndarray
    member_count: int
    weights: list[float]
    mean_quality: float

    @property
    def is_multi(self) -> bool:
        return self.member_count > 1


def build_template(
    embeddings: list[np.ndarray], qualities: list[QualityReport]
) -> Template:
    """Quality-weighted average of L2-normalised embeddings, renormalised.

    This is the standard template-pooling approach used in set-based face
    benchmarks: averaging in embedding space suppresses per-image noise, and
    weighting by quality stops a poor frame from dragging the template around.
    """
    if not embeddings:
        raise ValueError("Cannot build a template from zero embeddings.")

    if len(embeddings) == 1:
        return Template(
            embedding=np.asarray(embeddings[0], dtype=np.float64).ravel(),
            member_count=1,
            weights=[1.0],
            mean_quality=qualities[0].composite_score if qualities else 0.0,
        )

    raw_weights = np.array(
        [max(q.composite_score, 1.0) for q in qualities], dtype=np.float64
    )
    weights = raw_weights / raw_weights.sum()

    stacked = np.stack(
        [np.asarray(e, dtype=np.float64).ravel() for e in embeddings], axis=0
    )
    pooled = (stacked * weights[:, None]).sum(axis=0)
    norm = float(np.linalg.norm(pooled))
    if norm > 1e-10:
        pooled = pooled / norm

    return Template(
        embedding=pooled,
        member_count=len(embeddings),
        weights=[float(w) for w in weights],
        mean_quality=float(np.mean([q.composite_score for q in qualities])),
    )


@dataclass
class PairwiseMatrix:
    values: list[list[float]]
    minimum: float
    maximum: float
    mean: float
    spread: float

    def to_dict(self) -> dict:
        return {
            "values": [[round(v, 4) for v in row] for row in self.values],
            "min": round(self.minimum, 4),
            "max": round(self.maximum, 4),
            "mean": round(self.mean, 4),
            "spread": round(self.spread, 4),
        }


def pairwise_similarities(
    old_embeddings: list[np.ndarray], new_embeddings: list[np.ndarray]
) -> PairwiseMatrix:
    """Every old-vs-new similarity.

    Reported alongside the template score because a wide spread is itself
    diagnostic: it means the images disagree with one another, which the single
    pooled number would hide.
    """
    values = [
        [cosine_similarity(old, new) for new in new_embeddings]
        for old in old_embeddings
    ]
    flat = np.array([v for row in values for v in row], dtype=np.float64)
    return PairwiseMatrix(
        values=values,
        minimum=float(flat.min()),
        maximum=float(flat.max()),
        mean=float(flat.mean()),
        spread=float(flat.max() - flat.min()),
    )


def pose_difference(old_face: DetectedFace, new_face: DetectedFace) -> float:
    """Angular difference in head pose between two faces, in degrees.

    Euclidean norm over (yaw, pitch, roll) deltas - a coarse but adequate
    summary for gating and reporting.
    """
    old_pose = np.array(old_face.pose or (0.0, 0.0, 0.0), dtype=np.float64)
    new_pose = np.array(new_face.pose or (0.0, 0.0, 0.0), dtype=np.float64)
    return float(np.linalg.norm(old_pose - new_pose))


def representative_faces(
    faces: list[DetectedFace], qualities: list[QualityReport]
) -> int:
    """Index of the highest-quality face in a set, used for visualisation."""
    if not qualities:
        return 0
    return int(np.argmax([q.composite_score for q in qualities]))
