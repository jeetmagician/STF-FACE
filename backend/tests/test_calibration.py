"""Calibration and scoring tests."""

from __future__ import annotations

import numpy as np
import pytest

from app.config import Settings
from app.scoring import bands
from app.scoring.calibration import (
    MAX_ABS_LLR,
    CalibrationProfile,
    default_profile,
    fit_gaussian,
    fit_logistic,
    load_profile,
)


class TestMonotonicity:
    """The single most important property: more similar must never score lower.

    A Gaussian LLR with unequal variances is quadratic and turns around in the
    low-score tail, which without correction makes a cosine of -1.0 read as
    strong support FOR a match. These tests pin the correction down.
    """

    @pytest.mark.parametrize("backend", ["opencv", "insightface"])
    def test_llr_is_monotone_over_full_cosine_range(self, backend):
        profile = default_profile(backend)
        scores = np.linspace(-1.0, 1.0, 401)
        llrs = [profile.log_likelihood_ratio(float(s)) for s in scores]
        differences = np.diff(llrs)
        assert (differences >= -1e-9).all(), (
            f"{backend} LLR decreases at score "
            f"{scores[int(np.argmin(differences))]:.3f}"
        )

    @pytest.mark.parametrize("backend", ["opencv", "insightface"])
    def test_maximal_dissimilarity_is_evidence_against(self, backend):
        profile = default_profile(backend)
        assert profile.log_likelihood_ratio(-1.0) == pytest.approx(-MAX_ABS_LLR)
        assert profile.calibrated_score(-1.0) < 5.0

    def test_monotone_even_when_genuine_variance_is_much_larger(self):
        # The pathological configuration: sg >> si opens the parabola sharply.
        profile = CalibrationProfile(
            name="t", backend="x", genuine_mean=0.5, genuine_std=0.30,
            impostor_mean=0.05, impostor_std=0.04,
        )
        scores = np.linspace(-1.0, 1.0, 401)
        llrs = [profile.log_likelihood_ratio(float(s)) for s in scores]
        assert (np.diff(llrs) >= -1e-9).all()

    def test_vertex_is_detected_when_it_exists(self):
        profile = default_profile("opencv")
        vertex = profile._gaussian_vertex()
        assert vertex is not None
        # Below the turning point the LLR must be flat, not rising.
        assert profile.log_likelihood_ratio(vertex - 0.5) == pytest.approx(
            profile.log_likelihood_ratio(vertex), abs=1e-9
        )


class TestLikelihoodRatio:
    def test_llr_is_bounded(self):
        profile = default_profile("opencv")
        for score in np.linspace(-1, 1, 101):
            assert -MAX_ABS_LLR <= profile.log_likelihood_ratio(float(score)) <= MAX_ABS_LLR

    def test_llr_sign_matches_evidence_direction(self):
        profile = default_profile("opencv")
        assert profile.log_likelihood_ratio(0.85) > 0
        assert profile.log_likelihood_ratio(0.02) < 0

    def test_opencv_crossover_matches_published_threshold(self):
        """SFace's published reference cosine threshold is 0.363."""
        profile = default_profile("opencv")
        assert profile.decision_threshold(0.5) == pytest.approx(0.363, abs=0.01)

    def test_threshold_is_inside_the_monotone_region(self):
        for backend in ("opencv", "insightface"):
            profile = default_profile(backend)
            vertex = profile._gaussian_vertex()
            assert profile.decision_threshold(0.5) > vertex


class TestPrior:
    """The prior must genuinely move the answer, and must not touch the LLR."""

    def test_prior_shifts_posterior(self):
        profile = default_profile("opencv")
        score = 0.40
        low = profile.posterior(score, prior=0.01)
        neutral = profile.posterior(score, prior=0.5)
        high = profile.posterior(score, prior=0.99)
        assert low < neutral < high

    def test_llr_is_prior_free(self):
        profile = default_profile("opencv")
        # The LLR takes no prior at all - that is the point of reporting it.
        assert profile.log_likelihood_ratio(0.4) == profile.log_likelihood_ratio(0.4)

    def test_posterior_bounded(self):
        profile = default_profile("opencv")
        for score in np.linspace(-1, 1, 51):
            for prior in (0.001, 0.1, 0.5, 0.9, 0.999):
                assert 0.0 <= profile.posterior(float(score), prior) <= 1.0

    def test_extreme_prior_rejected_at_settings_level(self):
        with pytest.raises(ValueError):
            Settings(prior_same_person=0.0, _env_file=None)
        with pytest.raises(ValueError):
            Settings(prior_same_person=1.0, _env_file=None)


class TestFitting:
    def test_gaussian_recovers_known_parameters(self):
        rng = np.random.default_rng(7)
        genuine = rng.normal(0.62, 0.11, 4000)
        impostor = rng.normal(0.08, 0.07, 4000)
        mg, sg, mi, si = fit_gaussian(genuine, impostor)
        assert mg == pytest.approx(0.62, abs=0.01)
        assert sg == pytest.approx(0.11, abs=0.01)
        assert mi == pytest.approx(0.08, abs=0.01)
        assert si == pytest.approx(0.07, abs=0.01)

    def test_logistic_separates_classes(self):
        rng = np.random.default_rng(11)
        genuine = rng.normal(0.65, 0.10, 2000)
        impostor = rng.normal(0.10, 0.08, 2000)
        a, b, train_prior = fit_logistic(genuine, impostor)
        assert a > 0  # higher score must mean more likely genuine
        assert train_prior == pytest.approx(0.5, abs=0.02)

        profile = CalibrationProfile(
            name="fitted", backend="x", method="logistic",
            logistic_a=a, logistic_b=b, logistic_train_prior=train_prior,
        )
        assert profile.posterior(0.70, 0.5) > 0.9
        assert profile.posterior(0.05, 0.5) < 0.1

    def test_logistic_method_is_monotone(self):
        rng = np.random.default_rng(3)
        a, b, prior = fit_logistic(
            rng.normal(0.6, 0.12, 1500), rng.normal(0.1, 0.09, 1500)
        )
        profile = CalibrationProfile(
            name="f", backend="x", method="logistic",
            logistic_a=a, logistic_b=b, logistic_train_prior=prior,
        )
        llrs = [profile.log_likelihood_ratio(float(s)) for s in np.linspace(-1, 1, 200)]
        assert (np.diff(llrs) >= -1e-9).all()


class TestProfilePersistence:
    def test_roundtrip(self, tmp_path):
        original = CalibrationProfile(
            name="prod", backend="opencv", fitted=True,
            n_genuine_pairs=900, n_impostor_pairs=1200,
            dataset_description="internal set",
        )
        path = tmp_path / "opencv.prod.json"
        original.save(path)
        loaded = load_profile(tmp_path, "prod", "opencv")
        assert loaded.fitted is True
        assert loaded.n_genuine_pairs == 900
        assert loaded.dataset_description == "internal set"

    def test_missing_file_falls_back_to_unvalidated(self, tmp_path):
        profile = load_profile(tmp_path, "nonexistent", "opencv")
        assert profile.fitted is False
        assert profile.name == "default-unvalidated"

    def test_backend_mismatch_is_refused(self, tmp_path):
        """A calibration fitted for one model must never be applied to another."""
        wrong = CalibrationProfile(name="prod", backend="insightface", fitted=True)
        wrong.save(tmp_path / "opencv.prod.json")
        loaded = load_profile(tmp_path, "prod", "opencv")
        assert loaded.fitted is False
        assert loaded.name == "default-unvalidated"

    def test_corrupt_file_falls_back(self, tmp_path):
        (tmp_path / "opencv.broken.json").write_text("{not json", encoding="utf-8")
        assert load_profile(tmp_path, "broken", "opencv").fitted is False

    def test_shipped_defaults_are_marked_unfitted(self):
        for backend in ("opencv", "insightface"):
            assert default_profile(backend).fitted is False


class TestBands:
    def test_band_ordering(self):
        settings = Settings(_env_file=None)
        assert bands.band_for(95, settings).key == "very_high"
        assert bands.band_for(75, settings).key == "high"
        assert bands.band_for(55, settings).key == "moderate"
        assert bands.band_for(10, settings).key == "low"

    def test_thresholds_are_configurable(self):
        settings = Settings(band_very_high=95, band_high=90, band_moderate=80, _env_file=None)
        assert bands.band_for(92, settings).key == "high"
        assert bands.band_for(85, settings).key == "moderate"

    def test_no_band_asserts_identity(self):
        """The core ethical constraint, enforced as a test."""
        settings = Settings(_env_file=None)
        forbidden = [
            "definitely the same",
            "is the same person",
            "proves",
            "confirmed identity",
            "match confirmed",
            "same individual",
            "identity verified",
        ]
        for score in (5, 30, 55, 75, 95):
            band = bands.band_for(score, settings)
            text = f"{band.label} {band.statement} {band.guidance}".lower()
            for phrase in forbidden:
                assert phrase not in text, f"band {band.key} contains '{phrase}'"

    def test_high_bands_mention_lookalikes(self):
        settings = Settings(_env_file=None)
        for score in (75, 95):
            guidance = bands.band_for(score, settings).guidance.lower()
            assert "twin" in guidance or "look-alike" in guidance or "similar-" in guidance

    def test_low_band_does_not_assert_different_people(self):
        settings = Settings(_env_file=None)
        guidance = bands.band_for(10, settings).guidance.lower()
        assert "does not establish" in guidance

    def test_evidence_statement_reflects_direction(self):
        assert "different-people" in bands.describe_evidence_strength(-3.0)
        assert "same-person" in bands.describe_evidence_strength(3.0)
        assert "uninformative" in bands.describe_evidence_strength(0.1)
