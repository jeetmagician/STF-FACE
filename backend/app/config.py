"""Application configuration.

Every operational threshold lives here and is overridable by environment
variable, so that tuning the system never requires editing code.
"""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Literal

from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

BACKEND_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_MODEL_DIR = BACKEND_ROOT / "assets" / "models"
DEFAULT_CALIBRATION_DIR = BACKEND_ROOT / "assets" / "calibration"


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
        protected_namespaces=(),
    )

    # ---------------------------------------------------------------- app ---
    app_name: str = "Facet"
    environment: Literal["development", "production"] = "development"
    log_level: str = "INFO"

    # ------------------------------------------------------------ security ---
    api_key: str | None = Field(
        default=None,
        description=(
            "If set, every /api/* request except /api/health must present it as "
            "the X-API-Key header. Unset means open access - only acceptable "
            "for local development."
        ),
    )
    cors_origins: list[str] = Field(default_factory=lambda: ["http://localhost:3000"])

    max_upload_bytes: int = 12 * 1024 * 1024  # 12 MB per file
    max_files_per_side: int = 5
    max_image_pixels: int = 50_000_000  # decompression-bomb guard
    max_image_dimension: int = 8000
    min_image_dimension: int = 64

    rate_limit_requests: int = 30
    rate_limit_window_seconds: int = 60
    rate_limit_analyze_requests: int = 10
    rate_limit_analyze_window_seconds: int = 60

    # --------------------------------------------------------------- model ---
    model_backend: Literal["auto", "opencv", "insightface"] = "auto"
    model_dir: Path = DEFAULT_MODEL_DIR
    use_gpu: bool = False

    # InsightFace specifics
    insightface_pack: str = "buffalo_l"
    insightface_det_size: int = 640

    # Detection
    detection_confidence_threshold: float = 0.60
    max_faces_returned: int = 8

    # ------------------------------------------------------------- quality ---
    # Each gate is expressed in its natural unit. A face failing any HARD gate
    # produces a refusal rather than a score.
    min_face_width_px: int = 60          # hard gate
    advisory_face_width_px: int = 110    # below this -> warning
    min_blur_score: float = 18.0         # hard gate; normalised Laplacian variance
    advisory_blur_score: float = 45.0
    max_abs_yaw_deg: float = 45.0        # hard gate
    advisory_abs_yaw_deg: float = 25.0
    max_abs_pitch_deg: float = 35.0      # hard gate
    max_pose_difference_deg: float = 50.0  # hard gate on the *pair*
    advisory_pose_difference_deg: float = 25.0
    min_exposure_score: float = 0.20     # hard gate; 0..1
    advisory_exposure_score: float = 0.45
    min_quality_score: float = 30.0      # hard gate on composite 0..100
    advisory_quality_score: float = 55.0

    # Occlusion heuristics
    sunglasses_darkness_ratio: float = 0.55
    mask_texture_ratio: float = 0.42

    # ----------------------------------------------------------- calibration ---
    calibration_dir: Path = DEFAULT_CALIBRATION_DIR
    calibration_profile: str = "default"
    # Prior probability that a submitted pair is genuine. 0.5 is a deliberately
    # neutral, assumption-free default. It is NOT an empirical base rate.
    prior_same_person: float = 0.5

    # Score bands. Configurable; expressed on the calibrated 0-100 scale.
    band_very_high: float = 85.0
    band_high: float = 70.0
    band_moderate: float = 45.0

    # ------------------------------------------------------------- storage ---
    # Ephemeral in-memory store used only so the user can pick a face without
    # re-uploading. Never written to disk, never served as a public URL.
    session_ttl_seconds: int = 300
    session_max_entries: int = 200
    retain_for_face_selection: bool = True

    @field_validator("cors_origins", mode="before")
    @classmethod
    def _split_origins(cls, v: object) -> object:
        if isinstance(v, str):
            return [o.strip() for o in v.split(",") if o.strip()]
        return v

    @field_validator("prior_same_person")
    @classmethod
    def _check_prior(cls, v: float) -> float:
        if not 0.0 < v < 1.0:
            raise ValueError("prior_same_person must be strictly between 0 and 1")
        return v

    @property
    def is_production(self) -> bool:
        return self.environment == "production"


@lru_cache
def get_settings() -> Settings:
    return Settings()
