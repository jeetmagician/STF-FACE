"""API surface, security and privacy tests."""

from __future__ import annotations

import time

import numpy as np
import pytest

from app.config import Settings
from app.core.security import SlidingWindowLimiter
from app.core.store import EphemeralStore

from fixtures import encode_jpeg, encode_png, make_face


def _files(old_count: int = 1, new_count: int = 1, identity: int = 1):
    files = []
    for i in range(old_count):
        files.append(
            ("old_photos", (f"old{i}.jpg", encode_jpeg(make_face(identity)), "image/jpeg"))
        )
    for i in range(new_count):
        files.append(
            (
                "new_photos",
                (f"new{i}.jpg", encode_jpeg(make_face(identity, brightness=1.1)), "image/jpeg"),
            )
        )
    return files


class TestHealth:
    def test_health_is_reachable(self, api_client):
        response = api_client.get("/api/health")
        assert response.status_code == 200
        body = response.json()
        assert body["status"] == "ok"
        assert "calibration_validated" in body

    def test_health_discloses_uncalibrated_state(self, api_client):
        body = api_client.get("/api/health").json()
        assert body["calibration_validated"] is False
        assert any("alibration" in note for note in body["notes"])

    def test_health_discloses_open_access(self, api_client):
        body = api_client.get("/api/health").json()
        assert any("unauthenticated" in note for note in body["notes"])

    def test_root_carries_the_notice(self, api_client):
        body = api_client.get("/").json()
        assert "does not determine identity" in body["notice"]


class TestDetectEndpoint:
    def test_detects_a_face(self, api_client):
        response = api_client.post(
            "/api/face-detect",
            files={"image": ("face.jpg", encode_jpeg(make_face(1)), "image/jpeg")},
        )
        assert response.status_code == 200
        body = response.json()
        assert body["face_count"] >= 1
        assert body["faces"][0]["thumbnail"].startswith("data:image/")
        assert "quality" in body["faces"][0]

    def test_no_face_returns_422(self, api_client):
        blank = np.full((300, 300, 3), 15, dtype=np.uint8)
        response = api_client.post(
            "/api/face-detect",
            files={"image": ("blank.png", encode_png(blank), "image/png")},
        )
        assert response.status_code == 422
        assert response.json()["error"] == "no_face_detected"

    def test_rejects_non_image(self, api_client):
        response = api_client.post(
            "/api/face-detect",
            files={"image": ("evil.jpg", b"#!/bin/sh\nrm -rf /", "image/jpeg")},
        )
        assert response.status_code == 422
        assert response.json()["error"] == "unsupported_format"

    def test_mislabelled_content_type_still_rejected(self, api_client):
        """Trusting the client's declared MIME type would be the bug here."""
        response = api_client.post(
            "/api/face-detect",
            files={"image": ("x.png", b"%PDF-1.4 fake", "image/png")},
        )
        assert response.status_code == 422


class TestAnalyzeEndpoint:
    def test_returns_full_payload(self, api_client):
        response = api_client.post("/api/analyze", files=_files())
        assert response.status_code == 200, response.text
        body = response.json()
        assert 0 <= body["similarity_score"] <= 100
        assert body["confidence_level"] in {
            "Very High Similarity", "High Similarity",
            "Moderate Similarity", "Low Similarity",
        }
        assert body["face_detected_old"] is True
        assert body["face_detected_new"] is True

    def test_response_never_asserts_identity(self, api_client):
        """The ethical constraint, checked against the whole serialised response."""
        body = api_client.post("/api/analyze", files=_files()).text.lower()
        for phrase in (
            "definitely the same person",
            "is the same person",
            "proves identity",
            "identity confirmed",
            "verified as the same",
        ):
            assert phrase not in body

    def test_disclaimer_present(self, api_client):
        body = api_client.post("/api/analyze", files=_files()).json()
        assert "should not be treated as proof of identity" in body["disclaimer"]

    def test_multi_photo(self, api_client):
        response = api_client.post("/api/analyze", files=_files(3, 2))
        assert response.status_code == 200
        body = response.json()
        assert body["templates"]["old_photo_count"] == 3
        assert body["templates"]["new_photo_count"] == 2

    def test_too_many_files_rejected(self, api_client):
        response = api_client.post("/api/analyze", files=_files(9, 1))
        assert response.status_code == 422
        assert response.json()["error"] == "too_many_files"

    def test_no_face_on_one_side(self, api_client):
        blank = np.full((300, 300, 3), 15, dtype=np.uint8)
        response = api_client.post(
            "/api/analyze",
            files=[
                ("old_photos", ("o.jpg", encode_jpeg(make_face(1)), "image/jpeg")),
                ("new_photos", ("n.png", encode_png(blank), "image/png")),
            ],
        )
        assert response.status_code == 422
        assert response.json()["error"] == "no_face_detected"

    def test_multiple_faces_prompts_for_selection(self, api_client):
        image = make_face(1, second_face=True)
        response = api_client.post(
            "/api/analyze",
            files=[
                ("old_photos", ("o.jpg", encode_jpeg(image), "image/jpeg")),
                ("new_photos", ("n.jpg", encode_jpeg(make_face(1)), "image/jpeg")),
            ],
        )
        if response.status_code == 200:
            pytest.skip("mock detector found only one face in the composite")
        assert response.status_code == 409
        body = response.json()
        assert body["error"] == "multiple_faces_detected"
        assert len(body["faces"]) >= 2

    def test_invalid_selection_json_rejected(self, api_client):
        response = api_client.post(
            "/api/analyze", files=_files(), data={"old_face_indices": "not json"}
        )
        assert response.status_code == 422
        assert response.json()["error"] == "invalid_selection"

    def test_selection_length_must_match(self, api_client):
        response = api_client.post(
            "/api/analyze", files=_files(1, 1), data={"old_face_indices": "[0, 1, 2]"}
        )
        assert response.status_code == 422

    def test_visualisations_toggle(self, api_client):
        response = api_client.post(
            "/api/analyze", files=_files(), data={"include_visualisations": "false"}
        )
        assert response.json().get("visualisations") is None


class TestSecurityHeaders:
    def test_headers_present(self, api_client):
        headers = api_client.get("/api/health").headers
        assert headers["X-Content-Type-Options"] == "nosniff"
        assert headers["X-Frame-Options"] == "DENY"
        assert headers["Referrer-Policy"] == "no-referrer"

    def test_responses_are_not_cacheable(self, api_client):
        """Uploaded imagery must not sit in an intermediary cache."""
        headers = api_client.post("/api/analyze", files=_files()).headers
        assert "no-store" in headers["Cache-Control"]


class TestApiKey:
    def test_key_required_when_configured(self, monkeypatch, tmp_path):
        from fastapi.testclient import TestClient

        import app.api.deps as deps
        import app.config as config_module
        import app.models.registry as registry
        from fixtures import MockEmbedder

        config_module.get_settings.cache_clear()
        monkeypatch.setenv("API_KEY", "secret-key-value")
        monkeypatch.setenv("MODEL_DIR", str(tmp_path / "m"))
        monkeypatch.setenv("CALIBRATION_DIR", str(tmp_path / "c"))

        mock = MockEmbedder()
        registry.reset()
        monkeypatch.setattr(registry, "get_embedder", lambda settings=None: mock)

        from app.main import create_app

        application = create_app()
        application.dependency_overrides[deps.embedder_dependency] = lambda: mock

        with TestClient(application) as client:
            assert client.get("/api/health").status_code == 200  # health stays open
            assert client.post("/api/analyze", files=_files()).status_code == 401
            assert (
                client.post(
                    "/api/analyze", files=_files(), headers={"X-API-Key": "wrong"}
                ).status_code
                == 401
            )
            assert (
                client.post(
                    "/api/analyze",
                    files=_files(),
                    headers={"X-API-Key": "secret-key-value"},
                ).status_code
                == 200
            )

        registry.reset()
        config_module.get_settings.cache_clear()


class TestRateLimiter:
    def test_allows_up_to_limit(self):
        limiter = SlidingWindowLimiter()
        for _ in range(5):
            allowed, _ = limiter.check("client", limit=5, window_seconds=60)
            assert allowed

    def test_blocks_past_limit(self):
        limiter = SlidingWindowLimiter()
        for _ in range(3):
            limiter.check("client", 3, 60)
        allowed, retry_after = limiter.check("client", 3, 60)
        assert not allowed
        assert retry_after > 0

    def test_window_slides(self):
        limiter = SlidingWindowLimiter()
        for _ in range(2):
            limiter.check("client", 2, 1)
        assert not limiter.check("client", 2, 1)[0]
        time.sleep(1.05)
        assert limiter.check("client", 2, 1)[0]

    def test_clients_are_independent(self):
        limiter = SlidingWindowLimiter()
        for _ in range(3):
            limiter.check("a", 3, 60)
        assert not limiter.check("a", 3, 60)[0]
        assert limiter.check("b", 3, 60)[0]


class TestEphemeralStore:
    def _store(self, **overrides):
        base = dict(session_ttl_seconds=300, session_max_entries=5, _env_file=None)
        base.update(overrides)
        return EphemeralStore(Settings(**base))

    def test_roundtrip(self):
        store = self._store()
        images = [make_face(1, size=120)]
        token = store.put(images, images)
        assert token
        assert store.get(token) is not None

    def test_expiry(self):
        store = self._store(session_ttl_seconds=1)
        token = store.put([make_face(1, size=120)], [make_face(1, size=120)])
        assert store.get(token) is not None
        time.sleep(1.1)
        assert store.get(token) is None

    def test_drop_is_immediate(self):
        store = self._store()
        token = store.put([make_face(1, size=120)], [make_face(1, size=120)])
        store.drop(token)
        assert store.get(token) is None

    def test_capacity_evicts_oldest(self):
        store = self._store(session_max_entries=3)
        tokens = [
            store.put([make_face(i, size=120)], [make_face(i, size=120)])
            for i in range(5)
        ]
        assert store.size <= 3
        assert store.get(tokens[0]) is None
        assert store.get(tokens[-1]) is not None

    def test_tokens_are_unguessable(self):
        store = self._store()
        tokens = {
            store.put([make_face(1, size=120)], [make_face(1, size=120)])
            for _ in range(20)
        }
        assert len(tokens) == 20
        assert all(len(t) >= 40 for t in tokens)

    def test_retention_can_be_disabled(self):
        store = self._store(retain_for_face_selection=False)
        assert store.put([make_face(1, size=120)], [make_face(1, size=120)]) == ""
        assert store.size == 0

    def test_clear_empties_everything(self):
        store = self._store()
        for i in range(3):
            store.put([make_face(i, size=120)], [make_face(i, size=120)])
        store.clear()
        assert store.size == 0

    def test_analysis_does_not_retain_images(self, api_client):
        """After a successful comparison nothing should remain held."""
        response = api_client.post("/api/analyze", files=_files())
        assert response.status_code == 200
        assert api_client.get("/api/health").json()["active_sessions"] == 0
