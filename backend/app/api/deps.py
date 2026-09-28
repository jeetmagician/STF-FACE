"""Shared FastAPI dependencies."""

from __future__ import annotations

from fastapi import Request

from app.config import Settings, get_settings
from app.core.errors import ModelUnavailableError
from app.core.store import EphemeralStore
from app.models.base import FaceEmbedder
from app.models.registry import get_embedder
from app.scoring.calibration import CalibrationProfile, load_profile


def settings_dependency() -> Settings:
    return get_settings()


def embedder_dependency() -> FaceEmbedder:
    try:
        return get_embedder()
    except FileNotFoundError as exc:
        raise ModelUnavailableError(
            f"{exc} The face-recognition model is not available, so no comparison "
            "can be performed."
        ) from exc
    except ImportError as exc:
        raise ModelUnavailableError(str(exc)) from exc
    except Exception as exc:  # pragma: no cover - defensive
        raise ModelUnavailableError(
            f"The face-recognition model failed to load: {exc}"
        ) from exc


def calibration_dependency() -> CalibrationProfile:
    settings = get_settings()
    embedder = embedder_dependency()
    return load_profile(
        settings.calibration_dir, settings.calibration_profile, embedder.info.backend
    )


def store_dependency(request: Request) -> EphemeralStore:
    return request.app.state.store
