/**
 * API response types.
 *
 * These mirror app/schemas.py on the backend. If you change one, change both.
 */

export interface PoseInfo {
  yaw_deg: number;
  pitch_deg: number;
  roll_deg: number;
}

export interface OcclusionInfo {
  score: number;
  suspected_sunglasses: boolean;
  suspected_mask: boolean;
}

export interface QualityInfo {
  composite_score: number;
  usable: boolean;
  face_width_px: number;
  blur_score: number;
  exposure_score: number;
  brightness: number;
  contrast: number;
  clipped_highlight_fraction: number;
  clipped_shadow_fraction: number;
  pose: PoseInfo;
  detection_score: number;
  occlusion: OcclusionInfo;
  hard_failures: string[];
  warnings: string[];
}

export interface ScoringInfo {
  raw_embedding_similarity: number;
  metric: string;
  calibrated_similarity_score: number;
  log_likelihood_ratio: number;
  likelihood_ratio: number;
  evidence_statement: string;
  prior_used: number;
  prior_note: string;
  decision_threshold_raw: number;
  bands: Record<string, number>;
}

export interface CalibrationInfo {
  profile: string;
  method: string;
  is_validated: boolean;
  n_genuine_pairs: number;
  n_impostor_pairs: number;
  fitted_at: string | null;
  dataset_description: string;
  notes: string;
  metrics: Record<string, number>;
}

export interface ModelInfo {
  backend: string;
  detector: string;
  recognizer: string;
  embedding_dim: number;
  license: string;
  commercial_use_permitted: boolean;
  notes: string;
}

export interface RegionInfo {
  key: string;
  label: string;
  appearance_similarity: number;
  change_magnitude: number;
  stability_weight: number;
}

export interface MeasureInfo {
  key: string;
  label: string;
  old_value: number;
  new_value: number;
  absolute_delta: number;
  relative_delta: number;
}

export interface RegionAnalysisInfo {
  regions: RegionInfo[];
  geometric_measures: MeasureInfo[];
  geometry_reliable: boolean;
  geometry_caveat: string;
  landmark_density: number;
  narrative: string;
  interpretation: string;
}

export interface UncertaintySource {
  factor: string;
  severity: string;
  detail: string;
}

export interface PairwiseInfo {
  values: number[][];
  min: number;
  max: number;
  mean: number;
  spread: number;
}

export interface TemplateInfo {
  old_photo_count: number;
  new_photo_count: number;
  old_weights: number[];
  new_weights: number[];
  aggregation: string;
  pairwise: PairwiseInfo;
}

export interface VisualisationInfo {
  old_annotated: string;
  new_annotated: string;
  old_aligned: string;
  new_aligned: string;
  change_heatmap: string;
  heatmap_caption: string;
}

export interface AnalyzeResponse {
  similarity_score: number;
  confidence_level: string;
  confidence_key: "very_high" | "high" | "moderate" | "low";
  statement: string;
  guidance: string;
  disclaimer: string;
  method_statement: string;
  face_detected_old: boolean;
  face_detected_new: boolean;
  image_quality_old: QualityInfo;
  image_quality_new: QualityInfo;
  pose_difference: number;
  warnings: string[];
  analysis: string;
  scoring: ScoringInfo;
  calibration: CalibrationInfo;
  model: ModelInfo;
  templates: TemplateInfo;
  region_analysis: RegionAnalysisInfo;
  uncertainty_sources: UncertaintySource[];
  visualisations?: VisualisationInfo;
}

export interface DetectedFaceInfo {
  index: number;
  bbox: number[];
  detection_score: number;
  width_px: number;
  quality: QualityInfo;
  thumbnail: string;
}

export interface DetectResponse {
  face_count: number;
  faces: DetectedFaceInfo[];
  session_token: string | null;
  preview: string;
  warnings: string[];
}

export interface HealthResponse {
  status: string;
  version: string;
  model_loaded: boolean;
  model: ModelInfo | null;
  calibration_validated: boolean;
  calibration_profile: string;
  active_sessions: number;
  notes: string[];
}

export interface ApiErrorPayload {
  error: string;
  detail: string;
  faces?: DetectedFaceInfo[];
  session_token?: string;
  quality?: {
    side?: string;
    reasons?: string[];
    per_image?: QualityInfo[];
    pose_difference_deg?: number;
  };
}

/** A photo held in the browser, before upload. */
export interface LocalPhoto {
  id: string;
  file: File;
  previewUrl: string;
  detection?: DetectResponse;
  detecting: boolean;
  detectError?: string;
  selectedFaceIndex: number;
}
