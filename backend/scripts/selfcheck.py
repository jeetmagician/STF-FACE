#!/usr/bin/env python3
"""End-to-end pipeline self-check.

Runs the whole analysis pipeline against synthetic faces using a deterministic
mock backend, and asserts the properties that matter. Needs only numpy, OpenCV,
Pillow and pydantic - no FastAPI, no pytest, no downloaded weights.

Use it to confirm an install is sane before adding real weights, and in CI as a
fast smoke test:

    python scripts/selfcheck.py
    python scripts/selfcheck.py --verbose

What it does NOT do is measure recognition accuracy. The mock embedder reduces
pixels; it is not a trained face descriptor. Accuracy needs real weights and
real labelled pairs - see scripts/evaluate.py.
"""

from __future__ import annotations

import argparse
import sys
import tempfile
import traceback
from pathlib import Path

import numpy as np

BACKEND_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BACKEND_ROOT))
sys.path.insert(0, str(BACKEND_ROOT / "tests"))

from app.config import Settings  # noqa: E402
from app.core.errors import (  # noqa: E402
    DatabaseSearchDisabledError,
    DatabaseSearchPathError,
    MultipleFacesError,
    NoFaceError,
    QualityError,
)
from app.core.store import EphemeralStore  # noqa: E402
from app.models.alignment import apply_affine, umeyama_similarity  # noqa: E402
from app.models.base import (  # noqa: E402
    ARCFACE_112_TEMPLATE,
    DetectedFace,
    FaceEmbedder,
    ModelInfo,
    l2_normalise,
)
from app.pipeline import database_search  # noqa: E402
from app.pipeline.database_search import (  # noqa: E402
    build_index,
    recent_captures,
    run_database_compare,
    run_database_search,
    run_device_capture_search,
)
from app.pipeline.loader import decode_image  # noqa: E402
from app.pipeline.orchestrator import analyse_pair, process_side  # noqa: E402
from app.pipeline.quality import assess_quality  # noqa: E402
from app.pipeline.regions import REGION_STABILITY, analyse_regions, compare_regions  # noqa: E402
from app.pipeline.similarity import build_template, cosine_similarity  # noqa: E402
from app.scoring import bands  # noqa: E402
from app.scoring.calibration import MAX_ABS_LLR, default_profile, fit_gaussian, fit_logistic, CalibrationProfile  # noqa: E402

from fixtures import MockEmbedder, encode_jpeg, encode_png, make_face  # noqa: E402

PASSED: list[str] = []
FAILED: list[tuple[str, str]] = []
VERBOSE = False


def check(name: str):
    def decorator(function):
        def wrapper():
            try:
                function()
                PASSED.append(name)
                print(f"  \033[32mPASS\033[0m  {name}")
            except Exception as exc:
                detail = traceback.format_exc() if VERBOSE else f"{type(exc).__name__}: {exc}"
                FAILED.append((name, detail))
                print(f"  \033[31mFAIL\033[0m  {name}")
                print(f"        {detail.splitlines()[-1] if not VERBOSE else ''}")
                if VERBOSE:
                    print(detail)
        wrapper.__name__ = function.__name__
        return wrapper
    return decorator


def lenient_settings(**overrides) -> Settings:
    base = dict(
        min_quality_score=1.0, min_blur_score=1.0, min_face_width_px=20,
        model_dir=BACKEND_ROOT / "assets" / "models",
        calibration_dir=BACKEND_ROOT / "assets" / "calibration",
        _env_file=None,
    )
    base.update(overrides)
    return Settings(**base)


EMBEDDER = MockEmbedder()


# ------------------------------------------------------------- alignment ---

@check("Umeyama recovers a known similarity transform")
def t_umeyama():
    source = np.array([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0], [1.0, 1.0]])
    angle, scale = np.deg2rad(37), 3.1
    rotation = np.array([[np.cos(angle), -np.sin(angle)], [np.sin(angle), np.cos(angle)]])
    target = (source @ rotation.T) * scale + np.array([5.0, -2.0])
    mapped = apply_affine(source, umeyama_similarity(source, target))
    assert np.allclose(mapped, target, atol=1e-6), "transform not recovered"


@check("Alignment lands keypoints on the ArcFace template")
def t_alignment():
    face = EMBEDDER.detect(make_face(1))[0]
    assert face.aligned.shape == (112, 112, 3), f"got {face.aligned.shape}"
    assert np.allclose(face.landmarks_aligned[:5], ARCFACE_112_TEMPLATE, atol=6.0)


@check("Alignment is scale invariant")
def t_scale_invariance():
    small = EMBEDDER.detect(make_face(1, size=320))[0]
    large = EMBEDDER.detect(make_face(1, size=700))[0]
    difference = float(
        np.abs(small.aligned.astype(np.float32) / 255 - large.aligned.astype(np.float32) / 255).mean()
    )
    assert difference < 0.12, f"mean abs difference {difference:.3f}"


@check("Degenerate keypoints do not crash alignment")
def t_degenerate():
    matrix = umeyama_similarity(np.zeros((5, 2)), ARCFACE_112_TEMPLATE)
    assert np.isfinite(matrix).all()


# ----------------------------------------------------------------- image ---

@check("Valid JPEG and PNG decode to BGR arrays")
def t_decode():
    settings = lenient_settings()
    for encoder in (encode_jpeg, encode_png):
        decoded = decode_image(encoder(make_face(1)), settings)
        assert decoded.ndim == 3 and decoded.shape[2] == 3
        assert decoded.dtype == np.uint8


@check("Non-image payloads are rejected")
def t_reject_non_image():
    settings = lenient_settings()
    for payload in (b"", b"#!/bin/sh\nrm -rf /", b"%PDF-1.7", b"<html>err</html>", b"GIF89a" + b"\x00" * 100):
        try:
            decode_image(payload, settings)
        except Exception as exc:
            assert exc.__class__.__name__ == "ImageError", f"wrong error for {payload[:12]!r}"
        else:
            raise AssertionError(f"accepted a non-image payload: {payload[:16]!r}")


@check("Truncated image is rejected")
def t_truncated():
    data = encode_jpeg(make_face(1))
    try:
        decode_image(data[: len(data) // 3], lenient_settings())
    except Exception as exc:
        assert exc.__class__.__name__ == "ImageError"
    else:
        raise AssertionError("accepted a truncated JPEG")


@check("Decompression-bomb guard fires")
def t_bomb():
    try:
        decode_image(encode_png(make_face(1, size=500)), lenient_settings(max_image_pixels=10_000))
    except Exception as exc:
        assert exc.__class__.__name__ == "ImageError"
    else:
        raise AssertionError("no bomb guard")


# --------------------------------------------------------------- quality ---

@check("Good image passes the quality gate")
def t_quality_good():
    settings = Settings(_env_file=None)
    face = EMBEDDER.detect(make_face(1, size=520))[0]
    report = assess_quality(face, settings)
    assert report.usable, f"hard failures: {report.hard_failures}"


@check("Heavy blur fails the quality gate")
def t_quality_blur():
    settings = Settings(_env_file=None)
    faces = EMBEDDER.detect(make_face(1, blur=16.0))
    assert faces, "detector lost the blurred face entirely, which also counts"
    assert not assess_quality(faces[0], settings).usable


@check("Quality scores stay in range across conditions")
def t_quality_bounds():
    settings = Settings(_env_file=None)
    for image in (
        make_face(1), make_face(2, blur=3.0), make_face(3, brightness=0.3),
        make_face(4, brightness=1.9), make_face(5, noise=40), make_face(6, glasses=True),
    ):
        for face in EMBEDDER.detect(image):
            report = assess_quality(face, settings)
            assert 0 <= report.composite_score <= 100
            assert 0 <= report.exposure_score <= 1
            assert 0 <= report.occlusion_score <= 1


@check("Darker image scores lower on exposure")
def t_exposure_ordering():
    settings = Settings(_env_file=None)
    bright = assess_quality(EMBEDDER.detect(make_face(1))[0], settings)
    dark = assess_quality(EMBEDDER.detect(make_face(1, brightness=0.22))[0], settings)
    assert dark.exposure_score < bright.exposure_score


@check("Quality gates are configurable")
def t_quality_configurable():
    face = EMBEDDER.detect(make_face(1))[0]
    assert assess_quality(face, Settings(min_quality_score=1, _env_file=None)).usable
    assert not assess_quality(face, Settings(min_quality_score=99.5, _env_file=None)).usable


# ----------------------------------------------------------- calibration ---

@check("LLR is monotone over the full cosine range")
def t_monotone():
    scores = np.linspace(-1.0, 1.0, 601)
    for backend in ("opencv", "insightface"):
        profile = default_profile(backend)
        llrs = np.array([profile.log_likelihood_ratio(float(s)) for s in scores])
        differences = np.diff(llrs)
        worst = int(np.argmin(differences))
        assert (differences >= -1e-9).all(), (
            f"{backend} decreases at score {scores[worst]:.3f}"
        )


@check("Maximal dissimilarity is evidence against, not for")
def t_tail():
    for backend in ("opencv", "insightface"):
        profile = default_profile(backend)
        assert profile.log_likelihood_ratio(-1.0) < 0
        assert profile.calibrated_score(-1.0) < 5.0, "cosine -1 must not score high"


@check("Monotone even when genuine variance dwarfs impostor variance")
def t_monotone_pathological():
    profile = CalibrationProfile(
        name="t", backend="x", genuine_mean=0.5, genuine_std=0.34,
        impostor_mean=0.04, impostor_std=0.03,
    )
    llrs = [profile.log_likelihood_ratio(float(s)) for s in np.linspace(-1, 1, 400)]
    assert (np.diff(llrs) >= -1e-9).all()


@check("LLR is bounded by MAX_ABS_LLR")
def t_bounded():
    profile = default_profile("opencv")
    for score in np.linspace(-1, 1, 200):
        assert -MAX_ABS_LLR <= profile.log_likelihood_ratio(float(score)) <= MAX_ABS_LLR


@check("SFace crossover matches the published 0.363 threshold")
def t_threshold():
    threshold = default_profile("opencv").decision_threshold(0.5)
    assert abs(threshold - 0.363) < 0.01, f"threshold {threshold:.4f}"


@check("Prior shifts the posterior but not the evidence")
def t_prior():
    profile = default_profile("opencv")
    low = profile.posterior(0.40, 0.01)
    mid = profile.posterior(0.40, 0.5)
    high = profile.posterior(0.40, 0.99)
    assert low < mid < high, f"{low:.3f} {mid:.3f} {high:.3f}"
    assert 0 <= low and high <= 1


@check("Gaussian fitting recovers known parameters")
def t_fit_gaussian():
    rng = np.random.default_rng(7)
    mg, sg, mi, si = fit_gaussian(rng.normal(0.62, 0.11, 5000), rng.normal(0.08, 0.07, 5000))
    assert abs(mg - 0.62) < 0.01 and abs(sg - 0.11) < 0.01
    assert abs(mi - 0.08) < 0.01 and abs(si - 0.07) < 0.01


@check("Logistic fitting separates the classes")
def t_fit_logistic():
    rng = np.random.default_rng(11)
    a, b, prior = fit_logistic(rng.normal(0.65, 0.10, 2500), rng.normal(0.10, 0.08, 2500))
    assert a > 0, "slope must be positive"
    profile = CalibrationProfile(
        name="f", backend="x", method="logistic",
        logistic_a=a, logistic_b=b, logistic_train_prior=prior,
    )
    assert profile.posterior(0.70, 0.5) > 0.9
    assert profile.posterior(0.05, 0.5) < 0.1
    llrs = [profile.log_likelihood_ratio(float(s)) for s in np.linspace(-1, 1, 200)]
    assert (np.diff(llrs) >= -1e-9).all(), "logistic LLR must also be monotone"


@check("Shipped calibration defaults are marked unvalidated")
def t_unvalidated():
    for backend in ("opencv", "insightface"):
        assert default_profile(backend).fitted is False


@check("A calibration fitted for another backend is refused")
def t_backend_mismatch():
    import tempfile
    from app.scoring.calibration import load_profile

    with tempfile.TemporaryDirectory() as directory:
        CalibrationProfile(name="prod", backend="insightface", fitted=True).save(
            Path(directory) / "opencv.prod.json"
        )
        loaded = load_profile(Path(directory), "prod", "opencv")
        assert loaded.fitted is False, "applied a mismatched calibration"


# ----------------------------------------------------------------- bands ---

@check("Band thresholds order correctly and are configurable")
def t_bands():
    settings = Settings(_env_file=None)
    assert bands.band_for(95, settings).key == "very_high"
    assert bands.band_for(75, settings).key == "high"
    assert bands.band_for(55, settings).key == "moderate"
    assert bands.band_for(10, settings).key == "low"
    custom = Settings(band_very_high=95, band_high=90, band_moderate=80, _env_file=None)
    assert bands.band_for(92, custom).key == "high"


@check("No band text asserts identity")
def t_band_language():
    settings = Settings(_env_file=None)
    forbidden = [
        "definitely the same", "is the same person", "proves",
        "confirmed identity", "match confirmed", "identity verified",
    ]
    for score in (5, 30, 55, 75, 95):
        band = bands.band_for(score, settings)
        text = f"{band.label} {band.statement} {band.guidance}".lower()
        for phrase in forbidden:
            assert phrase not in text, f"band {band.key} says '{phrase}'"


@check("Low band does not assert different people")
def t_low_band():
    settings = Settings(_env_file=None)
    assert "does not establish" in bands.band_for(10, settings).guidance.lower()


# --------------------------------------------------------------- regions ---

@check("Identical faces yield near-perfect region similarity")
def t_regions_identical():
    face = EMBEDDER.detect(make_face(1))[0]
    for region in compare_regions(face.aligned, face.aligned):
        assert region.appearance_similarity > 0.98, f"{region.key} {region.appearance_similarity:.3f}"


@check("Nose ranked least stable, periocular most stable")
def t_stability_ranking():
    assert REGION_STABILITY["nose"] == min(REGION_STABILITY.values())
    assert REGION_STABILITY["periocular"] == max(REGION_STABILITY.values())


@check("A beard moves the jaw region more than the eye region")
def t_beard():
    clean = EMBEDDER.detect(make_face(1))[0]
    bearded = EMBEDDER.detect(make_face(1, beard=True))[0]
    regions = {r.key: r for r in compare_regions(clean.aligned, bearded.aligned)}
    assert regions["jaw_chin"].change_magnitude > regions["periocular"].change_magnitude, (
        f"jaw {regions['jaw_chin'].change_magnitude:.3f} vs "
        f"periocular {regions['periocular'].change_magnitude:.3f}"
    )


@check("Region narrative never claims a cause")
def t_narrative():
    a = EMBEDDER.detect(make_face(1))[0]
    b = EMBEDDER.detect(make_face(7, beard=True))[0]
    text = analyse_regions(a, b, "High Similarity", 4.0).narrative.lower()
    for phrase in ("has had", "underwent", "surgery was", "definitely", "proves"):
        assert phrase not in text, f"narrative says '{phrase}'"


@check("High pose difference marks geometry unreliable")
def t_geometry_pose():
    face = EMBEDDER.detect(make_face(1))[0]
    analysis = analyse_regions(face, face, "High Similarity", 40.0)
    assert analysis.geometry_reliable is False
    assert "projection artefact" in analysis.geometry_caveat


@check("Region analysis is labelled descriptive-only")
def t_region_disclaimer():
    face = EMBEDDER.detect(make_face(1))[0]
    payload = analyse_regions(face, face, "High Similarity", 3.0).to_dict()
    assert "descriptive only" in payload["interpretation"]
    assert "not included in the similarity score" in payload["interpretation"]


# ------------------------------------------------------------- templates ---

@check("Templates are normalised and quality-weighted")
def t_templates():
    from app.pipeline.quality import QualityReport

    def quality(score):
        return QualityReport(
            face_width_px=180, blur_score=90, exposure_score=0.8, brightness=128,
            contrast=50, clipped_highlight_fraction=0, clipped_shadow_fraction=0,
            yaw_deg=0, pitch_deg=0, roll_deg=0, detection_score=0.99,
            suspected_sunglasses=False, suspected_mask=False, occlusion_score=1.0,
            composite_score=score, usable=True,
        )

    good, bad = np.array([1.0, 0.0]), np.array([0.0, 1.0])
    template = build_template([good, bad], [quality(95), quality(30)])
    assert template.embedding[0] > template.embedding[1], "quality weighting inverted"
    assert abs(np.linalg.norm(template.embedding) - 1.0) < 1e-6
    assert abs(sum(template.weights) - 1.0) < 1e-6


@check("Cosine similarity is bounded and correct at the extremes")
def t_cosine():
    v = np.array([0.3, -0.5, 0.8])
    assert abs(cosine_similarity(v, v) - 1.0) < 1e-9
    assert abs(cosine_similarity(v, -v) + 1.0) < 1e-9
    assert cosine_similarity(np.zeros(4), np.ones(4)) == 0.0


# --------------------------------------------------------- orchestration ---

@check("No face detected raises rather than scoring")
def t_no_face():
    blank = np.full((300, 300, 3), 18, dtype=np.uint8)
    try:
        process_side([blank], EMBEDDER, lenient_settings(), "older photograph")
    except NoFaceError:
        return
    raise AssertionError("scored an image with no face")


@check("Multiple faces demand explicit selection")
def t_multi_face():
    image = make_face(1, second_face=True)
    if len(EMBEDDER.detect(image)) < 2:
        raise AssertionError("fixture did not produce two detectable faces")
    try:
        process_side([image], EMBEDDER, lenient_settings(), "older photograph")
    except MultipleFacesError as exc:
        assert len(exc.faces) >= 2
        assert all("thumbnail" in f for f in exc.faces)
        return
    raise AssertionError("silently picked a face")


@check("Poor quality refuses instead of producing a number")
def t_quality_refusal():
    try:
        process_side(
            [make_face(1)], EMBEDDER, lenient_settings(min_quality_score=99.9),
            "older photograph",
        )
    except QualityError as exc:
        assert "Insufficient image quality" in exc.message
        assert exc.details["reasons"]
        return
    raise AssertionError("produced a score for an image below the quality gate")


@check("Full analysis returns every documented field")
def t_full_payload():
    result = analyse_pair(
        [make_face(1)], [make_face(1, brightness=1.1)],
        EMBEDDER, lenient_settings(), default_profile("opencv"),
    )
    for key in (
        "similarity_score", "confidence_level", "statement", "guidance", "disclaimer",
        "face_detected_old", "face_detected_new", "image_quality_old",
        "image_quality_new", "pose_difference", "warnings", "analysis", "scoring",
        "calibration", "model", "templates", "region_analysis",
        "uncertainty_sources", "visualisations",
    ):
        assert key in result, f"missing field: {key}"
    assert 0 <= result["similarity_score"] <= 100
    assert -1 <= result["scoring"]["raw_embedding_similarity"] <= 1


@check("Uncalibrated state is surfaced in warnings and uncertainty")
def t_uncalibrated_surfaced():
    result = analyse_pair(
        [make_face(1)], [make_face(1)], EMBEDDER,
        lenient_settings(), default_profile("opencv"),
    )
    assert result["calibration"]["is_validated"] is False
    assert any("unvalidated" in w.lower() for w in result["warnings"])
    assert any(s["factor"] == "Uncalibrated model" for s in result["uncertainty_sources"])


@check("Look-alike caveat is always present")
def t_lookalike():
    result = analyse_pair(
        [make_face(1)], [make_face(1)], EMBEDDER,
        lenient_settings(), default_profile("opencv"),
    )
    assert any("Look-alikes" in s["factor"] for s in result["uncertainty_sources"])


@check("Multi-photo aggregation reports counts, weights and pairwise matrix")
def t_multi_photo():
    result = analyse_pair(
        [make_face(1), make_face(1, brightness=1.15), make_face(1, noise=8)],
        [make_face(1, brightness=0.9), make_face(1, blur=1.0)],
        EMBEDDER, lenient_settings(), default_profile("opencv"),
    )
    templates = result["templates"]
    assert templates["old_photo_count"] == 3 and templates["new_photo_count"] == 2
    assert len(templates["pairwise"]["values"]) == 3
    assert len(templates["pairwise"]["values"][0]) == 2
    assert abs(sum(templates["old_weights"]) - 1.0) < 0.01


@check("Visualisations are embedded data URIs, never fetchable links")
def t_visualisations():
    result = analyse_pair(
        [make_face(1)], [make_face(1)], EMBEDDER,
        lenient_settings(), default_profile("opencv"),
    )
    for key, value in result["visualisations"].items():
        if key == "heatmap_caption":
            continue
        assert value.startswith("data:image/"), f"{key} is not a data URI"
        assert "http" not in value[:40]


@check("Serialised response never asserts identity")
def t_response_language():
    import json

    result = analyse_pair(
        [make_face(1)], [make_face(1, brightness=1.02)],
        EMBEDDER, lenient_settings(), default_profile("opencv"),
    )
    text = json.dumps(result).lower()
    for phrase in (
        "definitely the same person", "is the same person", "proves identity",
        "identity confirmed", "verified as the same",
    ):
        assert phrase not in text, f"response contains '{phrase}'"
    assert "should not be treated as proof of identity" in text


@check("Same identity scores above different identities (plumbing only)")
def t_ordering():
    settings, profile = lenient_settings(), default_profile("opencv")
    same = analyse_pair(
        [make_face(1)], [make_face(1, brightness=1.05)], EMBEDDER, settings, profile
    )["scoring"]["raw_embedding_similarity"]
    different = analyse_pair(
        [make_face(1)], [make_face(42)], EMBEDDER, settings, profile
    )["scoring"]["raw_embedding_similarity"]
    assert same > different, f"same={same:.4f} different={different:.4f}"


# ----------------------------------------------------------------- store ---

@check("Ephemeral store expires, drops and caps entries")
def t_store():
    import time

    store = EphemeralStore(Settings(session_ttl_seconds=1, session_max_entries=3, _env_file=None))
    token = store.put([make_face(1, size=120)], [make_face(1, size=120)])
    assert store.get(token) is not None
    store.drop(token)
    assert store.get(token) is None, "drop did not remove the entry"

    tokens = [store.put([make_face(i, size=120)], [make_face(i, size=120)]) for i in range(5)]
    assert store.size <= 3, f"capacity exceeded: {store.size}"
    assert len(set(tokens)) == 5, "tokens collided"
    assert all(len(t) >= 40 for t in tokens), "tokens too short to be unguessable"

    fresh = store.put([make_face(1, size=120)], [make_face(1, size=120)])
    time.sleep(1.1)
    assert store.get(fresh) is None, "entry outlived its TTL"


@check("Retention can be disabled entirely")
def t_store_disabled():
    store = EphemeralStore(Settings(retain_for_face_selection=False, _env_file=None))
    assert store.put([make_face(1, size=120)], [make_face(1, size=120)]) == ""
    assert store.size == 0


# --------------------------------------------------------- database search ---

def database_settings(tmp_dir: Path, **overrides) -> Settings:
    base = dict(
        enable_database_search=True,
        database_search_dir=tmp_dir,
        database_search_cache_dir=tmp_dir / "_cache",
    )
    base.update(overrides)
    return lenient_settings(**base)


class _StubEmbedder(FaceEmbedder):
    """Always "detects" exactly one face with a fixed embedding.

    A plumbing double standing in for a *second*, distinct backend in a
    fallback scenario - not a detector. Used only to prove that a query
    routed to the fallback backend gets searched against that backend's own
    index, never the primary's; the real fallback is always a full OpenCV
    backend, which selfcheck must not depend on downloading weights for.
    """

    def __init__(self, backend: str, vector: np.ndarray):
        self._backend = backend
        self._vector = l2_normalise(vector)

    @property
    def info(self) -> ModelInfo:
        return ModelInfo(
            backend=self._backend,
            detector="stub",
            recognizer="stub",
            embedding_dim=len(self._vector),
            license="N/A - test fixture",
            commercial_use=False,
        )

    def detect(self, image_bgr: np.ndarray, max_faces: int = 8) -> list[DetectedFace]:
        height, width = image_bgr.shape[:2]
        return [
            DetectedFace(
                index=0,
                bbox=(0.0, 0.0, float(width), float(height)),
                detection_score=1.0,
                keypoints_5=np.zeros((5, 2), dtype=np.float32),
                aligned=np.zeros((112, 112, 3), dtype=np.uint8),
            )
        ]

    def embed(self, aligned_bgr: np.ndarray) -> np.ndarray:
        return self._vector


@check("Database search refuses when disabled")
def t_db_search_disabled():
    with tempfile.TemporaryDirectory() as tmp:
        settings = database_settings(Path(tmp), enable_database_search=False)
        try:
            run_database_search(encode_jpeg(make_face(1)), settings, EMBEDDER, default_profile("mock"))
        except DatabaseSearchDisabledError:
            return
        raise AssertionError("ran a search while disabled")


@check("Index build embeds library photos and skips faceless or corrupt ones")
def t_db_index_build():
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        (root / "alice.jpg").write_bytes(encode_jpeg(make_face(1)))
        (root / "bob.jpg").write_bytes(encode_jpeg(make_face(2)))
        (root / "carol.jpg").write_bytes(encode_jpeg(make_face(3)))
        (root / "blank.jpg").write_bytes(encode_jpeg(np.full((200, 200, 3), 18, dtype=np.uint8)))
        (root / "corrupt.jpg").write_bytes(b"not an image")

        index = build_index(database_settings(root), EMBEDDER, force=True)
        assert len(index.entries) == 3, f"expected 3 indexed photos, got {len(index.entries)}"
        assert index.skipped_no_face == 1
        assert index.skipped_error == 1


@check("Incremental rebuild picks up a changed file without re-embedding the rest")
def t_db_index_incremental():
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        (root / "alice.jpg").write_bytes(encode_jpeg(make_face(1)))
        (root / "bob.jpg").write_bytes(encode_jpeg(make_face(2)))
        settings = database_settings(root)

        first = build_index(settings, EMBEDDER, force=True)
        bob_entry = next(e for e in first.entries if "bob" in e.path)

        # Replace bob.jpg's content with a different identity.
        (root / "bob.jpg").write_bytes(encode_jpeg(make_face(9)))
        second = build_index(settings, EMBEDDER, force=False)
        assert len(second.entries) == 2

        new_bob = next(e for e in second.entries if "bob" in e.path)
        assert new_bob.embedding != bob_entry.embedding, "changed file kept its stale embedding"

        alice_entry_before = next(e for e in first.entries if "alice" in e.path)
        alice_entry_after = next(e for e in second.entries if "alice" in e.path)
        assert alice_entry_before.embedding == alice_entry_after.embedding, (
            "unchanged file was needlessly re-embedded"
        )


@check("Search ranks the matching identity highest")
def t_db_search_ranks_match():
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        (root / "alice.jpg").write_bytes(encode_jpeg(make_face(1)))
        (root / "bob.jpg").write_bytes(encode_jpeg(make_face(30)))
        (root / "carol.jpg").write_bytes(encode_jpeg(make_face(70)))
        database_search.reset()
        settings = database_settings(root)

        # A small brightness shift, not a large identity gap: the mock
        # embedding is pixel-reduction based (see fixtures.MockEmbedder), so
        # its identity signal is subtle enough that JPEG re-encoding can bury
        # a strong lighting change. This still exercises the real question -
        # does a different photo of the *same* face rank above other faces.
        query = encode_jpeg(make_face(30, brightness=1.02))
        result = run_database_search(query, settings, EMBEDDER, default_profile("mock"))

        assert result["indexed_photo_count"] == 3
        assert result["matches"], "no matches returned"
        assert "bob" in result["matches"][0]["filename"], (
            f"expected bob.jpg on top, got {result['matches'][0]['filename']}"
        )
        assert result["matches"][0]["raw_similarity"] > result["matches"][-1]["raw_similarity"]
        database_search.reset()


@check("Multiple faces in the query demand explicit selection")
def t_db_search_multi_face():
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        (root / "alice.jpg").write_bytes(encode_jpeg(make_face(1)))
        database_search.reset()
        settings = database_settings(root)

        query = encode_jpeg(make_face(1, second_face=True))
        try:
            run_database_search(query, settings, EMBEDDER, default_profile("mock"))
        except MultipleFacesError as exc:
            assert len(exc.faces) >= 2
            return
        raise AssertionError("silently picked a face in the query photo")


@check("Query photo with no face is refused, not scored")
def t_db_search_no_face():
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        (root / "alice.jpg").write_bytes(encode_jpeg(make_face(1)))
        database_search.reset()
        # Fallback disabled: the mock backend isn't "opencv", so
        # detect_with_fallback would otherwise try to load the *real* OpenCV
        # backend here, which needs downloaded weights selfcheck must not
        # require. A blank image has no face for either backend to find, so
        # this does not change what the check demonstrates.
        settings = database_settings(root, enable_detection_fallback=False)

        blank = encode_jpeg(np.full((200, 200, 3), 18, dtype=np.uint8))
        try:
            run_database_search(blank, settings, EMBEDDER, default_profile("mock"))
        except NoFaceError:
            return
        raise AssertionError("scored a query photo with no face")


@check("A query the primary backend can't see searches the fallback's own index")
def t_db_search_query_fallback():
    import app.models.registry as registry

    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        (root / "alice.jpg").write_bytes(encode_jpeg(make_face(1)))
        database_search.reset()
        settings = database_settings(root, enable_detection_fallback=True)

        stub = _StubEmbedder("stub-fallback", np.ones(16, dtype=np.float32))
        original_getter = registry.get_fallback_embedder
        registry.get_fallback_embedder = lambda _settings: stub
        try:
            # EMBEDDER (the mock, standing in for the primary) cannot find a
            # face in a blank canvas; the patched-in stub always can, so this
            # only succeeds if the search actually switched backends rather
            # than refusing outright.
            blank = encode_jpeg(np.full((200, 200, 3), 18, dtype=np.uint8))
            result = run_database_search(blank, settings, EMBEDDER, default_profile("mock"))
        finally:
            registry.get_fallback_embedder = original_getter
            database_search.reset()

        assert result["indexed_photo_count"] == 1, "the fallback's own index was not built"
        assert result["matches"], "fallback search returned no matches"
        assert any("fallback" in w.lower() for w in result["warnings"]), (
            "no warning explained that the fallback backend was used"
        )


@check("Compare detail behind a match reuses the full pairwise pipeline")
def t_db_compare_detail():
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        (root / "alice.jpg").write_bytes(encode_jpeg(make_face(1)))
        database_search.reset()
        settings = database_settings(root)

        query = encode_jpeg(make_face(1, brightness=1.03))
        result = run_database_compare(
            image_bytes=query,
            path=str(root / "alice.jpg"),
            settings=settings,
            embedder=EMBEDDER,
            profile=default_profile("mock"),
        )
        for key in ("similarity_score", "region_analysis", "uncertainty_sources", "scoring"):
            assert key in result, f"missing field: {key}"
        database_search.reset()


@check("Compare detail refuses a path outside the database folder")
def t_db_compare_path_traversal():
    with tempfile.TemporaryDirectory() as tmp, tempfile.TemporaryDirectory() as other:
        root = Path(tmp)
        (root / "alice.jpg").write_bytes(encode_jpeg(make_face(1)))
        outside = Path(other) / "secret.jpg"
        outside.write_bytes(encode_jpeg(make_face(2)))
        database_search.reset()
        settings = database_settings(root)

        query = encode_jpeg(make_face(1))
        try:
            run_database_compare(
                image_bytes=query,
                path=str(outside),
                settings=settings,
                embedder=EMBEDDER,
                profile=default_profile("mock"),
            )
        except DatabaseSearchPathError:
            return
        raise AssertionError("read a file outside the configured database folder")


@check("Device capture auto-selects the most prominent face and logs the result")
def t_db_device_capture():
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        (root / "alice.jpg").write_bytes(encode_jpeg(make_face(1)))
        database_search.reset()
        database_search.clear_captures()
        settings = database_settings(root)

        # Two faces in the query: a device has no one to ask which one was
        # meant, so this must succeed by picking the largest rather than
        # raising MultipleFacesError the way the interactive search does.
        query = encode_jpeg(make_face(1, second_face=True))
        result = run_device_capture_search(
            image_bytes=query,
            settings=settings,
            embedder=EMBEDDER,
            profile=default_profile("mock"),
            device_id="esp32-front-door",
        )
        assert "matches" in result

        logged = recent_captures(limit=5)
        assert len(logged) == 1, f"expected 1 logged capture, got {len(logged)}"
        assert logged[0]["device_id"] == "esp32-front-door"
        assert logged[0]["query_preview"].startswith("data:image/")
        database_search.reset()
        database_search.clear_captures()


def main() -> int:
    global VERBOSE
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()
    VERBOSE = args.verbose

    print("\nFacet pipeline self-check")
    print("=" * 66)
    print("Mock backend: no model weights required.")
    print("Verifies plumbing and score semantics, NOT recognition accuracy.\n")

    groups = {
        "Alignment": [t_umeyama, t_alignment, t_scale_invariance, t_degenerate],
        "Image handling": [t_decode, t_reject_non_image, t_truncated, t_bomb],
        "Quality gating": [
            t_quality_good, t_quality_blur, t_quality_bounds,
            t_exposure_ordering, t_quality_configurable,
        ],
        "Calibration": [
            t_monotone, t_tail, t_monotone_pathological, t_bounded, t_threshold,
            t_prior, t_fit_gaussian, t_fit_logistic, t_unvalidated, t_backend_mismatch,
        ],
        "Result language": [t_bands, t_band_language, t_low_band],
        "Region analysis": [
            t_regions_identical, t_stability_ranking, t_beard, t_narrative,
            t_geometry_pose, t_region_disclaimer,
        ],
        "Templates": [t_templates, t_cosine],
        "Orchestration": [
            t_no_face, t_multi_face, t_quality_refusal, t_full_payload,
            t_uncalibrated_surfaced, t_lookalike, t_multi_photo,
            t_visualisations, t_response_language, t_ordering,
        ],
        "Ephemeral store": [t_store, t_store_disabled],
        "Database search": [
            t_db_search_disabled, t_db_index_build, t_db_index_incremental,
            t_db_search_ranks_match, t_db_search_multi_face, t_db_search_no_face,
            t_db_search_query_fallback, t_db_compare_detail, t_db_compare_path_traversal,
            t_db_device_capture,
        ],
    }

    for group, tests in groups.items():
        print(f"\n{group}")
        print("-" * 66)
        for test in tests:
            test()

    total = len(PASSED) + len(FAILED)
    print("\n" + "=" * 66)
    print(f"{len(PASSED)}/{total} checks passed")
    if FAILED:
        print("\nFailures:")
        for name, detail in FAILED:
            print(f"  - {name}")
            if not VERBOSE:
                print(f"      {detail}")
        return 1
    print("\nPipeline is sound. Add model weights and run pytest for the full suite.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
