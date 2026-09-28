"""Synthetic faces and a deterministic mock backend.

The suite runs without downloaded model weights by substituting a deterministic
mock backend. That covers every stage of the pipeline except the two ONNX
forward passes: decoding, validation, detection plumbing, alignment, quality
gating, region analysis, templating, calibration, banding, rendering, the API
surface, security middleware and the ephemeral store.

What the mock does NOT establish is recognition *accuracy*. Its embeddings come
from downsampled pixels, not a trained network. Tests that assert on score
values therefore assert on the plumbing (that a score is produced, bounded,
ordered sensibly) and never on biometric performance. Measuring that requires
real weights and real labelled pairs - see scripts/evaluate.py.
"""

from __future__ import annotations

import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.config import Settings  # noqa: E402
from app.models.alignment import align_face, apply_affine, estimate_pose  # noqa: E402
from app.models.base import DetectedFace, FaceEmbedder, ModelInfo, l2_normalise  # noqa: E402


# --------------------------------------------------------------------------
# Synthetic faces
# --------------------------------------------------------------------------

def make_face(
    identity: int = 0,
    size: int = 480,
    brightness: float = 1.0,
    blur: float = 0.0,
    face_scale: float = 1.0,
    offset: tuple[int, int] = (0, 0),
    beard: bool = False,
    glasses: bool = False,
    noise: float = 0.0,
    second_face: bool = False,
    seed: int | None = None,
) -> np.ndarray:
    """Draw a schematic face whose geometry is a deterministic function of `identity`.

    Not photorealistic - it does not need to be. It needs consistent, gradient-
    rich structure in the right places so that detection, alignment, quality and
    region code all receive something with the statistics they expect.
    """
    rng = np.random.default_rng(seed if seed is not None else identity)
    canvas = np.full((size, size, 3), 28, dtype=np.uint8)

    # Identity-specific geometry.
    eye_spacing = 0.26 + (identity % 5) * 0.018
    eye_height = 0.40 + ((identity // 5) % 4) * 0.015
    nose_length = 0.14 + ((identity // 20) % 4) * 0.012
    jaw_width = 0.34 + ((identity // 3) % 5) * 0.012
    # Keep B well below R so the colour stays saturated enough for the mock
    # detector's HSV skin gate at every identity value.
    skin_tone = (
        120 + (identity % 7) * 6,
        150 + (identity % 5) * 6,
        200 + (identity % 6) * 5,
    )

    cx = size // 2 + offset[0]
    cy = size // 2 + offset[1]
    face_w = int(size * jaw_width * face_scale)
    face_h = int(size * 0.44 * face_scale)

    cv2.ellipse(canvas, (cx, cy), (face_w, face_h), 0, 0, 360, skin_tone, -1)
    # Shading gives the quality metrics real contrast to measure.
    cv2.ellipse(
        canvas,
        (cx - face_w // 4, cy - face_h // 6),
        (face_w // 2, face_h // 2),
        0,
        0,
        360,
        tuple(min(255, c + 22) for c in skin_tone),
        -1,
    )

    eye_dx = int(size * eye_spacing * face_scale / 2)
    eye_y = int(cy - face_h * (eye_height - 0.22))
    eye_r = max(5, int(size * 0.035 * face_scale))

    for sign in (-1, 1):
        ex = cx + sign * eye_dx
        cv2.ellipse(canvas, (ex, eye_y), (eye_r + 4, eye_r), 0, 0, 360, (242, 242, 246), -1)
        cv2.circle(canvas, (ex, eye_y), max(2, eye_r // 2), (62, 48, 40), -1)
        cv2.circle(canvas, (ex, eye_y), max(1, eye_r // 5), (12, 12, 14), -1)
        # Brow
        cv2.ellipse(
            canvas,
            (ex, eye_y - eye_r - 7),
            (eye_r + 6, max(2, eye_r // 3)),
            0,
            180,
            360,
            (72, 62, 58),
            -1,
        )

    nose_y = int(eye_y + size * nose_length * face_scale)
    cv2.line(
        canvas,
        (cx, eye_y + eye_r),
        (cx, nose_y),
        tuple(max(0, c - 28) for c in skin_tone),
        max(2, int(3 * face_scale)),
    )
    cv2.ellipse(
        canvas,
        (cx, nose_y),
        (max(4, int(size * 0.026 * face_scale)), max(3, int(size * 0.014 * face_scale))),
        0,
        0,
        360,
        tuple(max(0, c - 18) for c in skin_tone),
        -1,
    )

    mouth_y = int(nose_y + size * 0.085 * face_scale)
    mouth_w = int(size * 0.075 * face_scale)
    cv2.ellipse(canvas, (cx, mouth_y), (mouth_w, max(4, mouth_w // 3)), 0, 0, 180, (108, 76, 92), -1)

    if beard:
        cv2.ellipse(
            canvas,
            (cx, mouth_y + int(face_h * 0.22)),
            (int(face_w * 0.72), int(face_h * 0.34)),
            0,
            0,
            180,
            (58, 52, 50),
            -1,
        )
    if glasses:
        for sign in (-1, 1):
            ex = cx + sign * eye_dx
            cv2.rectangle(
                canvas,
                (ex - eye_r - 8, eye_y - eye_r - 5),
                (ex + eye_r + 8, eye_y + eye_r + 5),
                (18, 18, 20),
                -1,
            )

    if second_face:
        # Place the extra face on its own canvas beside the first, so the two
        # skin regions stay separate connected components. Overlaying it inside
        # the same frame merges the blobs and defeats the purpose.
        companion = make_face(identity + 11, size=size, seed=identity + 11)
        if blur > 0:
            canvas = cv2.GaussianBlur(canvas, (0, 0), sigmaX=blur, sigmaY=blur)
        if brightness != 1.0:
            canvas = np.clip(canvas.astype(np.float32) * brightness, 0, 255).astype(np.uint8)
        if noise > 0:
            canvas = np.clip(
                canvas.astype(np.float32) + rng.normal(0, noise, canvas.shape), 0, 255
            ).astype(np.uint8)
        return np.hstack([canvas, companion])

    if blur > 0:
        canvas = cv2.GaussianBlur(canvas, (0, 0), sigmaX=blur, sigmaY=blur)
    if brightness != 1.0:
        canvas = np.clip(canvas.astype(np.float32) * brightness, 0, 255).astype(np.uint8)
    if noise > 0:
        canvas = np.clip(
            canvas.astype(np.float32) + rng.normal(0, noise, canvas.shape), 0, 255
        ).astype(np.uint8)

    return canvas


def encode_jpeg(image: np.ndarray, quality: int = 92) -> bytes:
    ok, buffer = cv2.imencode(".jpg", image, [int(cv2.IMWRITE_JPEG_QUALITY), quality])
    assert ok
    return buffer.tobytes()


def encode_png(image: np.ndarray) -> bytes:
    ok, buffer = cv2.imencode(".png", image)
    assert ok
    return buffer.tobytes()


# --------------------------------------------------------------------------
# Mock backend
# --------------------------------------------------------------------------

class MockEmbedder(FaceEmbedder):
    """Deterministic stand-in for a real recognition backend.

    Detection: finds skin-toned blobs and places canonical keypoints inside
    each bounding box, so alignment, pose and quality run on real geometry.

    Embedding: a normalised, mean-centred 10x10 grayscale reduction of the
    aligned crop. A real function of image content - same face under different
    lighting scores high, different geometry scores lower - but emphatically
    not a trained face descriptor.
    """

    @property
    def info(self) -> ModelInfo:
        return ModelInfo(
            backend="mock",
            detector="MockDetector (colour blob)",
            recognizer="MockRecognizer (pixel reduction)",
            embedding_dim=100,
            license="N/A - test fixture",
            commercial_use=False,
            notes="Test fixture only. Carries no recognition accuracy whatsoever.",
        )

    def detect(self, image_bgr: np.ndarray, max_faces: int = 8) -> list[DetectedFace]:
        hsv = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2HSV)
        # Value floor kept low so deliberately under-exposed test images are
        # still detected - the quality gate, not the detector, should be what
        # rejects them.
        mask = cv2.inRange(hsv, np.array([0, 25, 32]), np.array([30, 200, 255]))
        mask = cv2.morphologyEx(
            mask, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (13, 13))
        )

        count, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
        candidates = []
        for label in range(1, count):
            x, y, w, h, area = stats[label]
            if area < 900 or w < 24 or h < 24:
                continue
            candidates.append((area, float(x), float(y), float(w), float(h)))

        candidates.sort(reverse=True)
        candidates = candidates[:max_faces]

        faces: list[DetectedFace] = []
        for index, (_, x, y, w, h) in enumerate(candidates):
            # Canonical keypoint placement inside the box, matching where the
            # synthetic generator draws the features.
            keypoints = np.array(
                [
                    [x + w * 0.33, y + h * 0.40],
                    [x + w * 0.67, y + h * 0.40],
                    [x + w * 0.50, y + h * 0.58],
                    [x + w * 0.38, y + h * 0.75],
                    [x + w * 0.62, y + h * 0.75],
                ],
                dtype=np.float32,
            )
            aligned, matrix = align_face(image_bgr, keypoints)
            faces.append(
                DetectedFace(
                    index=index,
                    bbox=(x, y, x + w, y + h),
                    detection_score=0.97,
                    keypoints_5=keypoints,
                    aligned=aligned,
                    landmarks=keypoints,
                    landmarks_aligned=apply_affine(keypoints, matrix),
                    pose=estimate_pose(keypoints, image_bgr.shape[:2]),
                )
            )
        return faces

    def embed(self, aligned_bgr: np.ndarray) -> np.ndarray:
        gray = cv2.cvtColor(aligned_bgr, cv2.COLOR_BGR2GRAY)
        reduced = cv2.resize(gray, (10, 10), interpolation=cv2.INTER_AREA).astype(np.float32)
        centred = reduced.ravel() - reduced.mean()
        return l2_normalise(centred)
