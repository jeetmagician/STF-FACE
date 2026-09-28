"""Region-level morphological analysis.

WHAT THIS IS AND IS NOT
-----------------------
This module answers "what changed between these two faces?" It does NOT answer
"are these the same person?" - that is the global embedding's job, and nothing
computed here is fused into the similarity score.

The reason is technical, not merely cautious. A global ArcFace/SFace embedding
is not decomposable into per-region contributions: you cannot re-weight the
periocular region "more heavily" inside a single 512-d vector, because the
vector has no region-indexed structure. Anything claiming otherwise is
inventing an interpretation the model does not support.

So the two signals are kept separate and labelled differently:

  * Geometric ratios  - anthropometric distances normalised by inter-ocular
                        distance, measured in original image space. Strongly
                        affected by pose, so they are reported with an explicit
                        pose-reliability caveat.
  * Appearance deltas - gradient-orientation descriptors over canonical regions
                        of the aligned crop. Lighting-robust relative to raw
                        pixels, but still descriptive only.

Stability weights below are heuristics informed by the general biometrics
literature on which facial areas persist across ageing and soft-tissue change.
They order the narrative; they do not weight any score.
"""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np

from app.models.base import DetectedFace
from app.pipeline.quality import CANONICAL_REGIONS

# Heuristic stability ranking. Higher = generally more persistent across time,
# weight change and soft-tissue surgery.
REGION_STABILITY: dict[str, float] = {
    "periocular": 0.95,
    "eye_left": 0.90,
    "eye_right": 0.90,
    "forehead": 0.80,
    "cheek_left": 0.55,
    "cheek_right": 0.55,
    "jaw_chin": 0.50,
    "mouth": 0.60,
    "nose": 0.35,
}

REGION_LABELS: dict[str, str] = {
    "periocular": "Periocular (eye surround)",
    "eye_left": "Left eye",
    "eye_right": "Right eye",
    "forehead": "Forehead",
    "cheek_left": "Left cheek",
    "cheek_right": "Right cheek",
    "jaw_chin": "Jaw and chin",
    "mouth": "Mouth",
    "nose": "Nose",
}

# Which measurements are commonly altered by which interventions. Used only to
# phrase the narrative, never to assert that an intervention occurred.
RHINOPLASTY_SENSITIVE = {"nose_projection_ratio", "nose_horizontal_offset", "nose_to_mouth_ratio"}
WEIGHT_SENSITIVE = {"face_width_ratio", "jaw_width_ratio", "facial_index"}


@dataclass
class GeometricMeasure:
    key: str
    label: str
    old_value: float
    new_value: float
    absolute_delta: float
    relative_delta: float  # fraction of the old value

    def to_dict(self) -> dict:
        return {
            "key": self.key,
            "label": self.label,
            "old_value": round(self.old_value, 4),
            "new_value": round(self.new_value, 4),
            "absolute_delta": round(self.absolute_delta, 4),
            "relative_delta": round(self.relative_delta, 4),
        }


@dataclass
class RegionComparison:
    key: str
    label: str
    appearance_similarity: float  # 0..1, cosine over gradient descriptors
    change_magnitude: float  # 1 - appearance_similarity
    stability_weight: float

    def to_dict(self) -> dict:
        return {
            "key": self.key,
            "label": self.label,
            "appearance_similarity": round(self.appearance_similarity, 4),
            "change_magnitude": round(self.change_magnitude, 4),
            "stability_weight": self.stability_weight,
        }


@dataclass
class RegionAnalysis:
    regions: list[RegionComparison]
    measures: list[GeometricMeasure]
    geometry_reliable: bool
    geometry_caveat: str
    landmark_density: int
    narrative: str

    def to_dict(self) -> dict:
        return {
            "regions": [r.to_dict() for r in self.regions],
            "geometric_measures": [m.to_dict() for m in self.measures],
            "geometry_reliable": self.geometry_reliable,
            "geometry_caveat": self.geometry_caveat,
            "landmark_density": self.landmark_density,
            "narrative": self.narrative,
            "interpretation": (
                "Region analysis is descriptive only. It indicates which areas of "
                "the face differ between the two photographs; it is not evidence "
                "of identity and is not included in the similarity score."
            ),
        }


# --------------------------------------------------------------------------
# Appearance descriptors
# --------------------------------------------------------------------------

def gradient_descriptor(patch_gray: np.ndarray, bins: int = 9, cells: int = 2) -> np.ndarray:
    """A small HOG-style descriptor: orientation histograms over a cell grid.

    Gradient orientation is substantially more robust to lighting change than
    raw intensity, which matters because most before/after pairs differ in
    illumination.
    """
    if patch_gray.size == 0:
        return np.zeros(bins * cells * cells, dtype=np.float32)

    patch = patch_gray.astype(np.float32)
    gx = cv2.Sobel(patch, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(patch, cv2.CV_32F, 0, 1, ksize=3)
    magnitude = np.sqrt(gx * gx + gy * gy)
    # Unsigned orientation in [0, 180)
    orientation = (np.rad2deg(np.arctan2(gy, gx)) % 180.0)

    height, width = patch.shape
    descriptor: list[np.ndarray] = []
    for cy in range(cells):
        for cx in range(cells):
            y1, y2 = cy * height // cells, (cy + 1) * height // cells
            x1, x2 = cx * width // cells, (cx + 1) * width // cells
            cell_mag = magnitude[y1:y2, x1:x2].ravel()
            cell_ori = orientation[y1:y2, x1:x2].ravel()
            if cell_mag.size == 0:
                descriptor.append(np.zeros(bins, dtype=np.float32))
                continue
            hist, _ = np.histogram(
                cell_ori, bins=bins, range=(0.0, 180.0), weights=cell_mag
            )
            descriptor.append(hist.astype(np.float32))

    vector = np.concatenate(descriptor)
    norm = float(np.linalg.norm(vector))
    if norm < 1e-8:
        return np.zeros_like(vector)
    return vector / norm


def compare_regions(old_aligned: np.ndarray, new_aligned: np.ndarray) -> list[RegionComparison]:
    """Compare canonical regions of two aligned 112x112 crops."""
    old_gray = cv2.cvtColor(old_aligned, cv2.COLOR_BGR2GRAY)
    new_gray = cv2.cvtColor(new_aligned, cv2.COLOR_BGR2GRAY)
    # Equalise globally so a uniform exposure difference does not masquerade as
    # a morphological change.
    old_gray = cv2.equalizeHist(old_gray)
    new_gray = cv2.equalizeHist(new_gray)

    comparisons: list[RegionComparison] = []
    for key, (x1, y1, x2, y2) in CANONICAL_REGIONS.items():
        old_patch = old_gray[y1:y2, x1:x2]
        new_patch = new_gray[y1:y2, x1:x2]
        old_desc = gradient_descriptor(old_patch)
        new_desc = gradient_descriptor(new_patch)
        similarity = float(np.clip(np.dot(old_desc, new_desc), 0.0, 1.0))
        comparisons.append(
            RegionComparison(
                key=key,
                label=REGION_LABELS[key],
                appearance_similarity=similarity,
                change_magnitude=1.0 - similarity,
                stability_weight=REGION_STABILITY[key],
            )
        )
    return comparisons


# --------------------------------------------------------------------------
# Geometric measures
# --------------------------------------------------------------------------

def _keypoint_measures(keypoints: np.ndarray) -> dict[str, float]:
    """Anthropometric ratios computable from the 5 canonical keypoints.

    All lengths are divided by inter-ocular distance, which makes them scale
    invariant. They remain pose dependent - see `geometry_reliable`.
    """
    left_eye, right_eye, nose, left_mouth, right_mouth = keypoints.astype(np.float64)

    iod = float(np.linalg.norm(right_eye - left_eye))
    if iod < 1e-6:
        return {}

    eye_mid = (left_eye + right_eye) / 2.0
    mouth_mid = (left_mouth + right_mouth) / 2.0

    # Rotate into an eye-line-aligned frame so roll does not contaminate the
    # vertical measurements.
    axis = (right_eye - left_eye) / iod
    normal = np.array([-axis[1], axis[0]])

    def project(point: np.ndarray) -> tuple[float, float]:
        delta = point - eye_mid
        return float(np.dot(delta, axis)), float(np.dot(delta, normal))

    nose_u, nose_v = project(nose)
    mouth_u, mouth_v = project(mouth_mid)

    measures = {
        "eye_to_mouth_ratio": abs(mouth_v) / iod,
        "eye_to_nose_ratio": abs(nose_v) / iod,
        "nose_projection_ratio": abs(nose_v) / (abs(mouth_v) + 1e-6),
        "nose_to_mouth_ratio": abs(mouth_v - nose_v) / iod,
        "nose_horizontal_offset": abs(nose_u) / iod,
        "mouth_width_ratio": float(np.linalg.norm(right_mouth - left_mouth)) / iod,
        "mouth_horizontal_offset": abs(mouth_u) / iod,
    }

    # Angle at the nose tip subtended by the two eyes - a compact summary of the
    # central facial triangle.
    v1 = left_eye - nose
    v2 = right_eye - nose
    denom = float(np.linalg.norm(v1) * np.linalg.norm(v2))
    if denom > 1e-6:
        cos_angle = float(np.clip(np.dot(v1, v2) / denom, -1.0, 1.0))
        measures["nasal_apex_angle_deg"] = float(np.degrees(np.arccos(cos_angle)))

    return measures


def _contour_measures(landmarks: np.ndarray, keypoints: np.ndarray) -> dict[str, float]:
    """Measures from a dense landmark set.

    Points are selected by geometric extreme rather than by hardcoded index, so
    this stays correct if a backend changes its landmark ordering.
    """
    if landmarks is None or landmarks.shape[0] < 20:
        return {}

    left_eye, right_eye = keypoints[0].astype(np.float64), keypoints[1].astype(np.float64)
    iod = float(np.linalg.norm(right_eye - left_eye))
    if iod < 1e-6:
        return {}

    eye_mid = (left_eye + right_eye) / 2.0
    axis = (right_eye - left_eye) / iod
    normal = np.array([-axis[1], axis[0]])

    points = landmarks.astype(np.float64)
    deltas = points - eye_mid
    u = deltas @ axis
    v = deltas @ normal

    face_width = float(u.max() - u.min())
    chin_v = float(np.abs(v).max())

    measures = {
        "face_width_ratio": face_width / iod,
        "face_height_ratio": chin_v / iod,
        "facial_index": (chin_v / face_width) if face_width > 1e-6 else 0.0,
    }

    # Jaw width: horizontal extent of landmarks in the lower third of the face.
    lower_band = np.abs(v) > (0.6 * chin_v)
    if int(lower_band.sum()) >= 4:
        jaw_width = float(u[lower_band].max() - u[lower_band].min())
        measures["jaw_width_ratio"] = jaw_width / iod

    return measures


_MEASURE_LABELS = {
    "eye_to_mouth_ratio": "Eye-line to mouth distance (÷ inter-ocular)",
    "eye_to_nose_ratio": "Eye-line to nose tip distance (÷ inter-ocular)",
    "nose_projection_ratio": "Nose position between eye-line and mouth",
    "nose_to_mouth_ratio": "Nose tip to mouth distance (÷ inter-ocular)",
    "nose_horizontal_offset": "Nose lateral offset (÷ inter-ocular)",
    "mouth_width_ratio": "Mouth width (÷ inter-ocular)",
    "mouth_horizontal_offset": "Mouth lateral offset (÷ inter-ocular)",
    "nasal_apex_angle_deg": "Eye–nose–eye angle (degrees)",
    "face_width_ratio": "Face width (÷ inter-ocular)",
    "face_height_ratio": "Face height (÷ inter-ocular)",
    "facial_index": "Facial index (height ÷ width)",
    "jaw_width_ratio": "Jaw width (÷ inter-ocular)",
}


def compute_geometry(
    old_face: DetectedFace, new_face: DetectedFace
) -> list[GeometricMeasure]:
    old_measures = _keypoint_measures(old_face.keypoints_5)
    new_measures = _keypoint_measures(new_face.keypoints_5)

    if (
        old_face.landmarks is not None
        and new_face.landmarks is not None
        and old_face.landmarks.shape[0] >= 20
        and new_face.landmarks.shape[0] >= 20
    ):
        old_measures.update(_contour_measures(old_face.landmarks, old_face.keypoints_5))
        new_measures.update(_contour_measures(new_face.landmarks, new_face.keypoints_5))

    results: list[GeometricMeasure] = []
    for key in sorted(set(old_measures) & set(new_measures)):
        old_value = old_measures[key]
        new_value = new_measures[key]
        absolute = new_value - old_value
        relative = absolute / old_value if abs(old_value) > 1e-6 else 0.0
        results.append(
            GeometricMeasure(
                key=key,
                label=_MEASURE_LABELS.get(key, key),
                old_value=old_value,
                new_value=new_value,
                absolute_delta=absolute,
                relative_delta=relative,
            )
        )
    return results


# --------------------------------------------------------------------------
# Narrative
# --------------------------------------------------------------------------

def _describe_change(magnitude: float) -> str:
    if magnitude >= 0.55:
        return "substantial"
    if magnitude >= 0.35:
        return "moderate"
    if magnitude >= 0.18:
        return "modest"
    return "minimal"


def build_narrative(
    regions: list[RegionComparison],
    measures: list[GeometricMeasure],
    global_similarity_band: str,
    geometry_reliable: bool,
) -> str:
    """Compose the human-readable morphological summary.

    Deliberately never asserts a cause. "The nasal region differs" is supportable;
    "the subject has had rhinoplasty" is not.
    """
    if not regions:
        return "Region-level comparison was not available for this pair."

    ranked = sorted(regions, key=lambda r: r.change_magnitude, reverse=True)
    most_changed = ranked[0]
    stable = [r for r in regions if r.stability_weight >= 0.80]
    stable_mean = (
        float(np.mean([r.appearance_similarity for r in stable])) if stable else 0.0
    )

    parts: list[str] = []

    changed = [r for r in ranked if r.change_magnitude >= 0.35]
    if changed:
        names = ", ".join(r.label.lower() for r in changed[:3])
        parts.append(
            f"The {names} {'region shows' if len(changed) == 1 else 'regions show'} "
            f"{_describe_change(most_changed.change_magnitude)} appearance differences "
            "between the two photographs."
        )
    else:
        parts.append(
            "No facial region shows a large appearance difference between the two "
            "photographs."
        )

    if stable:
        if stable_mean >= 0.70:
            parts.append(
                "The periocular and forehead regions - generally among the more "
                "persistent facial areas - remain comparatively consistent."
            )
        elif stable_mean >= 0.45:
            parts.append(
                "The periocular and forehead regions show partial consistency."
            )
        else:
            parts.append(
                "The periocular and forehead regions also differ noticeably, which "
                "weakens the case that the differences are confined to soft tissue "
                "or styling."
            )

    parts.append(
        f"The global embedding comparison places this pair in the "
        f"'{global_similarity_band}' band."
    )

    if geometry_reliable:
        notable = [
            m
            for m in measures
            if abs(m.relative_delta) >= 0.12 and m.key in RHINOPLASTY_SENSITIVE
        ]
        if notable:
            parts.append(
                "Nasal geometric measurements differ by more than 12% between the "
                "photographs. Note that nasal geometry is among the least stable "
                "facial characteristics, being alterable by surgery and by ageing."
            )
        weight_changed = [
            m for m in measures if abs(m.relative_delta) >= 0.12 and m.key in WEIGHT_SENSITIVE
        ]
        if weight_changed:
            parts.append(
                "Facial width and jaw measurements also differ, which is consistent "
                "with soft-tissue change such as weight variation, though other "
                "causes are equally possible."
            )
    else:
        parts.append(
            "Geometric measurements were not used in this summary because the head "
            "poses differ enough to distort two-dimensional distance ratios."
        )

    return " ".join(parts)


def analyse_regions(
    old_face: DetectedFace,
    new_face: DetectedFace,
    global_similarity_band: str,
    pose_difference_deg: float,
    pose_reliability_limit_deg: float = 15.0,
) -> RegionAnalysis:
    regions = compare_regions(old_face.aligned, new_face.aligned)
    measures = compute_geometry(old_face, new_face)

    geometry_reliable = pose_difference_deg <= pose_reliability_limit_deg
    caveat = (
        "Two-dimensional geometric ratios are measured in the original image "
        "plane and are sensitive to head pose. "
        + (
            f"Pose difference between the photographs is {pose_difference_deg:.0f}°, "
            "which is small enough for these ratios to be broadly comparable."
            if geometry_reliable
            else f"Pose difference between the photographs is "
            f"{pose_difference_deg:.0f}°, large enough that apparent differences in "
            "these ratios may be projection artefacts rather than real "
            "morphological change."
        )
    )

    landmark_density = (
        int(old_face.landmarks.shape[0]) if old_face.landmarks is not None else 0
    )

    narrative = build_narrative(
        regions, measures, global_similarity_band, geometry_reliable
    )

    return RegionAnalysis(
        regions=regions,
        measures=measures,
        geometry_reliable=geometry_reliable,
        geometry_caveat=caveat,
        landmark_density=landmark_density,
        narrative=narrative,
    )
