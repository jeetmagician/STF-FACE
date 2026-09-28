"""Pydantic response schemas.

These document the API contract. The pipeline builds plain dicts matching these
shapes; the models are declared as `response_model` so that FastAPI publishes an
accurate OpenAPI schema at /docs.
"""

from __future__ import annotations

from pydantic import BaseModel, Field


class PoseInfo(BaseModel):
    yaw_deg: float
    pitch_deg: float
    roll_deg: float


class OcclusionInfo(BaseModel):
    score: float
    suspected_sunglasses: bool
    suspected_mask: bool


class QualityInfo(BaseModel):
    composite_score: float = Field(description="Composite image quality, 0-100.")
    usable: bool
    face_width_px: float
    blur_score: float = Field(description="Variance of Laplacian on the aligned crop.")
    exposure_score: float
    brightness: float
    contrast: float
    clipped_highlight_fraction: float
    clipped_shadow_fraction: float
    pose: PoseInfo
    detection_score: float
    occlusion: OcclusionInfo
    hard_failures: list[str]
    warnings: list[str]


class ScoringInfo(BaseModel):
    raw_embedding_similarity: float = Field(
        description="Cosine similarity between the pooled embeddings, before calibration."
    )
    metric: str
    calibrated_similarity_score: float
    log_likelihood_ratio: float = Field(
        description=(
            "Natural log of p(score|same)/p(score|different). The prior-free "
            "measure of how much evidence this comparison carries."
        )
    )
    likelihood_ratio: float
    evidence_statement: str
    prior_used: float
    prior_note: str
    decision_threshold_raw: float
    bands: dict[str, float]


class CalibrationInfo(BaseModel):
    profile: str
    method: str
    is_validated: bool = Field(
        description="False means the shipped placeholder parameters are in use."
    )
    n_genuine_pairs: int
    n_impostor_pairs: int
    fitted_at: str | None
    dataset_description: str
    notes: str
    metrics: dict


class ModelInfoSchema(BaseModel):
    backend: str
    detector: str
    recognizer: str
    embedding_dim: int
    license: str
    commercial_use_permitted: bool
    notes: str


class PairwiseInfo(BaseModel):
    values: list[list[float]]
    min: float
    max: float
    mean: float
    spread: float


class TemplateInfo(BaseModel):
    old_photo_count: int
    new_photo_count: int
    old_weights: list[float]
    new_weights: list[float]
    aggregation: str
    pairwise: PairwiseInfo


class RegionInfo(BaseModel):
    key: str
    label: str
    appearance_similarity: float
    change_magnitude: float
    stability_weight: float


class MeasureInfo(BaseModel):
    key: str
    label: str
    old_value: float
    new_value: float
    absolute_delta: float
    relative_delta: float


class RegionAnalysisInfo(BaseModel):
    regions: list[RegionInfo]
    geometric_measures: list[MeasureInfo]
    geometry_reliable: bool
    geometry_caveat: str
    landmark_density: int
    narrative: str
    interpretation: str


class UncertaintySource(BaseModel):
    factor: str
    severity: str
    detail: str


class VisualisationInfo(BaseModel):
    old_annotated: str
    new_annotated: str
    old_aligned: str
    new_aligned: str
    change_heatmap: str
    heatmap_caption: str


class AnalyzeResponse(BaseModel):
    similarity_score: float
    confidence_level: str
    confidence_key: str
    statement: str
    guidance: str
    disclaimer: str
    method_statement: str
    face_detected_old: bool
    face_detected_new: bool
    image_quality_old: QualityInfo
    image_quality_new: QualityInfo
    pose_difference: float
    warnings: list[str]
    analysis: str
    scoring: ScoringInfo
    calibration: CalibrationInfo
    model_config = {"protected_namespaces": ()}
    model: ModelInfoSchema
    templates: TemplateInfo
    region_analysis: RegionAnalysisInfo
    uncertainty_sources: list[UncertaintySource]
    visualisations: VisualisationInfo | None = None


class DetectedFaceSchema(BaseModel):
    index: int
    bbox: list[float]
    detection_score: float
    width_px: float
    quality: QualityInfo
    thumbnail: str


class DetectResponse(BaseModel):
    face_count: int
    faces: list[DetectedFaceSchema]
    session_token: str | None = None
    preview: str
    warnings: list[str]


class HealthResponse(BaseModel):
    status: str
    version: str
    model_loaded: bool
    model_config = {"protected_namespaces": ()}
    model: ModelInfoSchema | None = None
    calibration_validated: bool
    calibration_profile: str
    active_sessions: int
    notes: list[str]
