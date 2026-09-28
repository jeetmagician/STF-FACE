"""Pytest fixtures.

The synthetic-face generator and the mock backend live in `fixtures.py` so they
can be imported without pytest installed - `scripts/selfcheck.py` uses them to
verify the pipeline in environments that only have numpy, OpenCV and Pillow.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from app.config import Settings  # noqa: E402
from fixtures import (  # noqa: E402,F401
    MockEmbedder,
    encode_jpeg,
    encode_png,
    make_face,
)


@pytest.fixture
def settings(tmp_path) -> Settings:
    return Settings(
        environment="development",
        model_dir=tmp_path / "models",
        calibration_dir=tmp_path / "calibration",
        api_key=None,
        cors_origins=["http://localhost:3000"],
        rate_limit_requests=10_000,
        rate_limit_analyze_requests=10_000,
        _env_file=None,
    )


@pytest.fixture
def embedder() -> MockEmbedder:
    return MockEmbedder()


@pytest.fixture
def face_image() -> np.ndarray:
    return make_face(identity=1)


@pytest.fixture
def api_client(monkeypatch, tmp_path):
    """A TestClient wired to the mock backend."""
    from fastapi.testclient import TestClient

    import app.api.deps as deps
    import app.config as config_module
    import app.models.registry as registry

    config_module.get_settings.cache_clear()
    monkeypatch.setenv("MODEL_DIR", str(tmp_path / "models"))
    monkeypatch.setenv("CALIBRATION_DIR", str(tmp_path / "calibration"))
    monkeypatch.setenv("RATE_LIMIT_REQUESTS", "10000")
    monkeypatch.setenv("RATE_LIMIT_ANALYZE_REQUESTS", "10000")
    monkeypatch.delenv("API_KEY", raising=False)

    mock = MockEmbedder()
    registry.reset()
    monkeypatch.setattr(registry, "get_embedder", lambda settings=None: mock)
    monkeypatch.setattr(deps, "embedder_dependency", lambda: mock)

    from app.main import create_app

    application = create_app()
    application.dependency_overrides[deps.embedder_dependency] = lambda: mock

    with TestClient(application) as client:
        yield client

    registry.reset()
    config_module.get_settings.cache_clear()
