"""Image and face quality assessment.

Everything here is measured on the aligned 112x112 crop wherever possible, so
values are directly comparable between images of different sizes. The composite
score gates the pipeline: a pair failing a hard gate produces a refusal rather
than a misleading number.

The occlusion checks are explicitly *heuristics*. They are reported as
"suspected" and never silently alter the similarity score.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import cv2
import numpy as np

from app.config import Settings
from app.models.base import DetectedFace

# Canonical regions in the 112x112 aligned frame, derived from the ArcFace
# template landmark positions (eyes y~51.6, nose y~71.7, mouth y~92.3).
CANONICAL_REGIONS: dict[str, tuple[int, int, int, int]] = {
    # name: (x1, y1, x2, y2)
    "forehead": (28, 8, 84, 38),
    "periocular": (14, 36, 98, 68),
    "eye_left": (24, 40, 54, 64),
    "eye_right": (58, 40, 88, 64),
    "nose": (42, 52, 70, 84),
    "mouth": (34, 80, 78, 104),
    "cheek_left": (10, 64, 40, 92),
    "cheek_right": (72, 64, 102, 92),
    "jaw_chin": (24, 92, 88, 112),
}


@dataclass
class QualityReport:
    face_width_px: float
    blur_score: float
    exposure_score: float
    brightness: float
    contrast: float
    clipped_highlight_fraction: float
    clipped_shadow_fraction: float
    yaw_deg: float
    pitch_deg: float
    roll_deg: float
    detection_score: float
    suspected_sunglasses: bool
    suspected_mask: bool
    occlusion_score: float
    composite_score: float
    usable: bool
    hard_failures: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "composite_score": round(self.composite_score, 1),
            "usable": self.usable,
            "face_width_px": round(self.face_width_px, 1),
            "blur_score": round(self.blur_score, 1),
            "exposure_score": round(self.exposure_score, 3),
            "brightness": round(self.brightness, 1),
            "contrast": round(self.contrast, 1),
            "clipped_highlight_fraction": round(self.clipped_highlight_fraction, 4),
            "clipped_shadow_fraction": round(self.clipped_shadow_fraction, 4),
            "pose": {
                "yaw_deg": round(self.yaw_deg, 1),
                "pitch_deg": round(self.pitch_deg, 1),
                "roll_deg": round(self.roll_deg, 1),
            },
            "detection_score": round(self.detection_score, 3),
            "occlusion": {
                "score": round(self.occlusion_score, 3),
                "suspected_sunglasses": self.suspected_sunglasses,
                "suspected_mask": self.suspected_mask,
            },
            "hard_failures": self.hard_failures,
            "warnings": self.warnings,
        }


def _crop(aligned_gray: np.ndarray, region: str) -> np.ndarray:
    x1, y1, x2, y2 = CANONICAL_REGIONS[region]
    return aligned_gray[y1:y2, x1:x2]


def laplacian_variance(gray: np.ndarray) -> float:
    if gray.size == 0:
        return 0.0
    return float(cv2.Laplacian(gray, cv2.CV_64F).var())


def _exposure_metrics(gray: np.ndarray) -> tuple[float, float, float, float, float]:
    """Return (score, brightness, contrast, clipped_high, clipped_low)."""
    pixels = gray.astype(np.float32)
    brightness = float(pixels.mean())
    contrast = float(pixels.std())
    clipped_high = float((pixels >= 250).mean())
    clipped_low = float((pixels <= 5).mean())

    # Penalise departures from mid-grey, low contrast, and clipping. Each term
    # is in 0..1 and the score is their product, so any single severe problem
    # drags the result down rather than being averaged away.
    brightness_term = 1.0 - min(1.0, abs(brightness - 128.0) / 110.0)
    contrast_term = min(1.0, contrast / 52.0)
    clipping_term = 1.0 - min(1.0, (clipped_high + clipped_low) * 4.0)

    score = float(
        np.clip(brightness_term * (0.35 + 0.65 * contrast_term) * clipping_term, 0.0, 1.0)
    )
    return score, brightness, contrast, clipped_high, clipped_low


def _region_texture(aligned_gray: np.ndarray, names: list[str]) -> float:
    """Mean Laplacian variance across several regions.

    Regions have different widths, so they are measured individually and
    averaged rather than stacked into one array.
    """
    values = [laplacian_variance(_crop(aligned_gray, name)) for name in names]
    values = [v for v in values if v is not None]
    return float(np.mean(values)) if values else 0.0


def _occlusion_metrics(
    aligned_gray: np.ndarray, settings: Settings
) -> tuple[bool, bool, float]:
    """Heuristic occlusion detection. Returns (sunglasses, mask, score 0..1)."""
    face_mean = float(aligned_gray.mean()) + 1e-6

    periocular = _crop(aligned_gray, "periocular")
    periocular_mean = float(periocular.mean())
    darkness_ratio = periocular_mean / face_mean
    periocular_texture = laplacian_variance(periocular)

    # Sunglasses: the eye band is both notably darker than the rest of the face
    # and unusually flat (lenses remove iris/sclera detail).
    suspected_sunglasses = bool(
        darkness_ratio < settings.sunglasses_darkness_ratio and periocular_texture < 120.0
    )

    upper_texture = _region_texture(aligned_gray, ["forehead", "periocular"]) + 1e-6
    lower_texture = _region_texture(aligned_gray, ["mouth", "jaw_chin"])
    texture_ratio = lower_texture / upper_texture

    # Mask: the lower face is far smoother than the upper face.
    suspected_mask = bool(texture_ratio < settings.mask_texture_ratio and upper_texture > 20.0)

    score = 1.0
    if suspected_sunglasses:
        score -= 0.5
    if suspected_mask:
        score -= 0.4
    return suspected_sunglasses, suspected_mask, float(np.clip(score, 0.0, 1.0))


def assess_quality(
    face: DetectedFace, settings: Settings, label: str = "image"
) -> QualityReport:
    """Measure quality for one detected face and apply the configured gates."""
    aligned_gray = cv2.cvtColor(face.aligned, cv2.COLOR_BGR2GRAY)

    face_width = float(face.width)
    blur = laplacian_variance(aligned_gray)
    exposure, brightness, contrast, clip_hi, clip_lo = _exposure_metrics(aligned_gray)
    yaw, pitch, roll = face.pose or (0.0, 0.0, 0.0)
    sunglasses, mask, occlusion = _occlusion_metrics(aligned_gray, settings)

    hard_failures: list[str] = []
    warnings: list[str] = []

    if face_width < settings.min_face_width_px:
        hard_failures.append(
            f"The face in the {label} is only {face_width:.0f}px wide; "
            f"at least {settings.min_face_width_px}px is required."
        )
    elif face_width < settings.advisory_face_width_px:
        warnings.append(
            f"The face in the {label} is small ({face_width:.0f}px wide), "
            "which reduces the reliability of the comparison."
        )

    if blur < settings.min_blur_score:
        hard_failures.append(f"The {label} is too blurred to compare reliably.")
    elif blur < settings.advisory_blur_score:
        warnings.append(f"The {label} is somewhat soft or out of focus.")

    if abs(yaw) > settings.max_abs_yaw_deg:
        hard_failures.append(
            f"The head in the {label} is turned {abs(yaw):.0f}° away from the camera, "
            f"beyond the {settings.max_abs_yaw_deg:.0f}° limit."
        )
    elif abs(yaw) > settings.advisory_abs_yaw_deg:
        warnings.append(f"The head in the {label} is turned {abs(yaw):.0f}° to one side.")

    if abs(pitch) > settings.max_abs_pitch_deg:
        hard_failures.append(
            f"The head in the {label} is tilted {abs(pitch):.0f}° up or down, "
            f"beyond the {settings.max_abs_pitch_deg:.0f}° limit."
        )

    if exposure < settings.min_exposure_score:
        hard_failures.append(
            f"Lighting in the {label} is too extreme for a reliable comparison."
        )
    elif exposure < settings.advisory_exposure_score:
        warnings.append(f"Lighting in the {label} is uneven or low-contrast.")

    if sunglasses:
        warnings.append(
            f"The eye region in the {label} may be occluded (possibly glasses or "
            "sunglasses). The periocular region is normally among the most stable "
            "areas, so occluding it weakens the comparison."
        )
    if mask:
        warnings.append(
            f"The lower face in the {label} may be covered (possibly a mask or scarf)."
        )

    # Composite: each term in 0..1, weighted, expressed on 0..100.
    size_term = float(np.clip(face_width / settings.advisory_face_width_px, 0.0, 1.0))
    blur_term = float(np.clip(blur / settings.advisory_blur_score, 0.0, 1.0))
    pose_term = float(
        np.clip(1.0 - (abs(yaw) / 60.0) - (abs(pitch) / 50.0), 0.0, 1.0)
    )
    detection_term = float(np.clip(face.detection_score, 0.0, 1.0))

    composite = 100.0 * (
        0.26 * size_term
        + 0.26 * blur_term
        + 0.18 * exposure
        + 0.16 * pose_term
        + 0.08 * occlusion
        + 0.06 * detection_term
    )

    if composite < settings.min_quality_score and not hard_failures:
        hard_failures.append(
            f"Overall quality of the {label} ({composite:.0f}/100) is below the "
            f"minimum of {settings.min_quality_score:.0f} required for a "
            "defensible comparison."
        )
    elif composite < settings.advisory_quality_score:
        warnings.append(
            f"Overall quality of the {label} is modest ({composite:.0f}/100); "
            "treat the result with extra caution."
        )

    return QualityReport(
        face_width_px=face_width,
        blur_score=blur,
        exposure_score=exposure,
        brightness=brightness,
        contrast=contrast,
        clipped_highlight_fraction=clip_hi,
        clipped_shadow_fraction=clip_lo,
        yaw_deg=yaw,
        pitch_deg=pitch,
        roll_deg=roll,
        detection_score=face.detection_score,
        suspected_sunglasses=sunglasses,
        suspected_mask=mask,
        occlusion_score=occlusion,
        composite_score=composite,
        usable=not hard_failures,
        hard_failures=hard_failures,
        warnings=warnings,
    )
