"""Visualisation renderings returned with the analysis.

All images are returned as base64 data URIs in the JSON response. Nothing is
written to disk and no image is ever served from a public URL, so there is no
path by which an uploaded photograph becomes fetchable by a third party.

Design constraint: the heatmap shows *appearance difference between regions*,
using the same descriptors reported numerically in the region table. It is not
a saliency map over the embedding, because no such map is available for these
models, and rendering one would imply an explanatory power the system does not
have.
"""

from __future__ import annotations

import base64

import cv2
import numpy as np

from app.models.base import DetectedFace
from app.pipeline.quality import CANONICAL_REGIONS
from app.pipeline.regions import RegionComparison

# OpenCV uses BGR channel order, so these are written B, G, R - not RGB.
# The accent matches the interface's brass-500 (#c39440 -> RGB 195,148,64).
_ACCENT = (64, 148, 195)      # BGR - amber
_LANDMARK = (150, 230, 160)   # BGR - soft green


def to_data_uri(image_bgr: np.ndarray, max_side: int = 420) -> str:
    height, width = image_bgr.shape[:2]
    scale = min(1.0, max_side / max(height, width))
    if scale < 1.0:
        image_bgr = cv2.resize(
            image_bgr,
            (max(1, int(round(width * scale))), max(1, int(round(height * scale)))),
            interpolation=cv2.INTER_AREA,
        )
    ok, buffer = cv2.imencode(".jpg", image_bgr, [int(cv2.IMWRITE_JPEG_QUALITY), 88])
    if not ok:  # pragma: no cover - defensive
        return ""
    return "data:image/jpeg;base64," + base64.b64encode(buffer.tobytes()).decode("ascii")


def annotate_detection(image_bgr: np.ndarray, face: DetectedFace) -> np.ndarray:
    """Draw the detected bounding box and landmarks on a copy of the source."""
    canvas = image_bgr.copy()
    x1, y1, x2, y2 = (int(round(v)) for v in face.bbox)

    thickness = max(2, int(round(min(canvas.shape[:2]) / 260)))
    cv2.rectangle(canvas, (x1, y1), (x2, y2), _ACCENT, thickness)

    # Corner ticks make the box read as a deliberate UI element rather than a
    # raw debug rectangle.
    tick = max(8, int(round((x2 - x1) * 0.16)))
    for (cx, cy, dx, dy) in (
        (x1, y1, 1, 1),
        (x2, y1, -1, 1),
        (x1, y2, 1, -1),
        (x2, y2, -1, -1),
    ):
        cv2.line(canvas, (cx, cy), (cx + dx * tick, cy), _ACCENT, thickness + 1)
        cv2.line(canvas, (cx, cy), (cx, cy + dy * tick), _ACCENT, thickness + 1)

    if face.landmarks is not None:
        radius = max(1, int(round(min(canvas.shape[:2]) / 400)))
        dense = face.landmarks.shape[0] > 10
        for point in face.landmarks:
            cv2.circle(
                canvas,
                (int(round(point[0])), int(round(point[1]))),
                radius if dense else radius + 1,
                _LANDMARK,
                -1,
                lineType=cv2.LINE_AA,
            )

    return canvas


def render_aligned(face: DetectedFace, size: int = 224) -> np.ndarray:
    return cv2.resize(face.aligned, (size, size), interpolation=cv2.INTER_CUBIC)


def render_region_heatmap(
    face: DetectedFace, regions: list[RegionComparison], size: int = 224
) -> np.ndarray:
    """Overlay per-region appearance change onto the aligned face.

    Warmer = larger measured difference between the two photographs in that
    region. Values come directly from the region table, so the picture and the
    numbers can never disagree.
    """
    heat = np.zeros((112, 112), dtype=np.float32)
    weight = np.zeros((112, 112), dtype=np.float32)

    for region in regions:
        box = CANONICAL_REGIONS.get(region.key)
        if box is None:
            continue
        x1, y1, x2, y2 = box
        heat[y1:y2, x1:x2] += float(region.change_magnitude)
        weight[y1:y2, x1:x2] += 1.0

    valid = weight > 0
    heat[valid] /= weight[valid]

    # Blur so region boundaries do not read as anatomical edges.
    heat = cv2.GaussianBlur(heat, (0, 0), sigmaX=9, sigmaY=9)
    heat = np.clip(heat, 0.0, 1.0)

    heat_u8 = (heat * 255).astype(np.uint8)
    coloured = cv2.applyColorMap(heat_u8, cv2.COLORMAP_INFERNO)

    base = face.aligned.copy()
    alpha = np.clip(heat * 0.72, 0.0, 0.72)[..., None]
    blended = (base.astype(np.float32) * (1 - alpha) + coloured.astype(np.float32) * alpha)
    blended = np.clip(blended, 0, 255).astype(np.uint8)

    return cv2.resize(blended, (size, size), interpolation=cv2.INTER_CUBIC)


def render_face_thumbnail(image_bgr: np.ndarray, face: DetectedFace, size: int = 128) -> str:
    """Cropped thumbnail for the multiple-face picker."""
    x1, y1, x2, y2 = face.bbox
    pad_x = (x2 - x1) * 0.18
    pad_y = (y2 - y1) * 0.18
    height, width = image_bgr.shape[:2]

    cx1 = max(0, int(round(x1 - pad_x)))
    cy1 = max(0, int(round(y1 - pad_y)))
    cx2 = min(width, int(round(x2 + pad_x)))
    cy2 = min(height, int(round(y2 + pad_y)))

    if cx2 <= cx1 or cy2 <= cy1:
        crop = face.aligned
    else:
        crop = image_bgr[cy1:cy2, cx1:cx2]

    crop = cv2.resize(crop, (size, size), interpolation=cv2.INTER_AREA)
    return to_data_uri(crop, max_side=size)
