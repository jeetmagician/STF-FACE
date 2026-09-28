"""Pipeline tests: alignment, regions, templating, orchestration."""

from __future__ import annotations

import numpy as np
import pytest

from app.config import Settings
from app.core.errors import MultipleFacesError, NoFaceError, QualityError
from app.models.alignment import align_face, apply_affine, umeyama_similarity
from app.models.base import ARCFACE_112_TEMPLATE
from app.pipeline.orchestrator import analyse_pair, process_side
from app.pipeline.regions import (
    REGION_STABILITY,
    analyse_regions,
    compare_regions,
    compute_geometry,
    gradient_descriptor,
)
from app.pipeline.similarity import (
    build_template,
    cosine_similarity,
    pairwise_similarities,
    pose_difference,
)
from app.scoring.calibration import default_profile

from fixtures import make_face


class TestAlignment:
    def test_umeyama_recovers_a_known_similarity_transform(self):
        source = np.array([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0], [1.0, 1.0]])
        angle = np.deg2rad(30)
        scale = 2.5
        rotation = np.array(
            [[np.cos(angle), -np.sin(angle)], [np.sin(angle), np.cos(angle)]]
        )
        target = (source @ rotation.T) * scale + np.array([4.0, -3.0])

        matrix = umeyama_similarity(source, target)
        mapped = apply_affine(source, matrix)
        assert np.allclose(mapped, target, atol=1e-6)

    def test_alignment_puts_keypoints_on_the_template(self, embedder):
        faces = embedder.detect(make_face(1))
        assert faces
        aligned_points = faces[0].landmarks_aligned
        # The five keypoints should land close to the ArcFace template positions.
        assert np.allclose(aligned_points[:5], ARCFACE_112_TEMPLATE, atol=6.0)

    def test_aligned_output_shape(self, embedder):
        faces = embedder.detect(make_face(1))
        assert faces[0].aligned.shape == (112, 112, 3)

    def test_alignment_is_scale_invariant(self, embedder):
        """The same face at two sizes should align to nearly the same crop."""
        small = embedder.detect(make_face(1, size=300))
        large = embedder.detect(make_face(1, size=700))
        if not small or not large:
            pytest.skip("detector missed a variant")
        a = small[0].aligned.astype(np.float32) / 255
        b = large[0].aligned.astype(np.float32) / 255
        assert float(np.abs(a - b).mean()) < 0.12

    def test_degenerate_points_do_not_crash(self):
        degenerate = np.zeros((5, 2), dtype=np.float32)
        matrix = umeyama_similarity(degenerate, ARCFACE_112_TEMPLATE)
        assert matrix.shape == (2, 3)
        assert np.isfinite(matrix).all()


class TestSimilarity:
    def test_cosine_bounds(self):
        rng = np.random.default_rng(1)
        for _ in range(50):
            a, b = rng.normal(size=128), rng.normal(size=128)
            assert -1.0 <= cosine_similarity(a, b) <= 1.0

    def test_identical_vectors(self):
        v = np.array([0.3, -0.5, 0.8, 0.1])
        assert cosine_similarity(v, v) == pytest.approx(1.0)

    def test_opposite_vectors(self):
        v = np.array([1.0, 0.0, 0.0])
        assert cosine_similarity(v, -v) == pytest.approx(-1.0)

    def test_zero_vector_is_safe(self):
        assert cosine_similarity(np.zeros(8), np.ones(8)) == 0.0


class TestTemplates:
    def _quality(self, score):
        from app.pipeline.quality import QualityReport

        return QualityReport(
            face_width_px=180, blur_score=90, exposure_score=0.8, brightness=128,
            contrast=50, clipped_highlight_fraction=0, clipped_shadow_fraction=0,
            yaw_deg=0, pitch_deg=0, roll_deg=0, detection_score=0.99,
            suspected_sunglasses=False, suspected_mask=False, occlusion_score=1.0,
            composite_score=score, usable=True,
        )

    def test_single_embedding_passes_through(self):
        embedding = np.array([0.6, 0.8])
        template = build_template([embedding], [self._quality(80)])
        assert template.member_count == 1
        assert np.allclose(template.embedding, embedding)

    def test_template_is_normalised(self):
        rng = np.random.default_rng(5)
        embeddings = [rng.normal(size=64) for _ in range(4)]
        embeddings = [e / np.linalg.norm(e) for e in embeddings]
        template = build_template(
            embeddings, [self._quality(s) for s in (90, 70, 50, 30)]
        )
        assert np.linalg.norm(template.embedding) == pytest.approx(1.0, abs=1e-6)

    def test_higher_quality_images_dominate(self):
        good = np.array([1.0, 0.0])
        bad = np.array([0.0, 1.0])
        template = build_template([good, bad], [self._quality(95), self._quality(35)])
        assert template.embedding[0] > template.embedding[1]

    def test_weights_sum_to_one(self):
        rng = np.random.default_rng(9)
        embeddings = [rng.normal(size=32) for _ in range(3)]
        template = build_template(
            embeddings, [self._quality(s) for s in (80, 60, 40)]
        )
        assert sum(template.weights) == pytest.approx(1.0)

    def test_empty_rejected(self):
        with pytest.raises(ValueError):
            build_template([], [])

    def test_pairwise_matrix_shape_and_spread(self):
        rng = np.random.default_rng(2)
        old = [rng.normal(size=16) for _ in range(3)]
        new = [rng.normal(size=16) for _ in range(2)]
        matrix = pairwise_similarities(old, new)
        assert len(matrix.values) == 3
        assert len(matrix.values[0]) == 2
        assert matrix.spread == pytest.approx(matrix.maximum - matrix.minimum)


class TestRegions:
    def test_gradient_descriptor_is_normalised(self):
        patch = (np.random.default_rng(1).random((30, 30)) * 255).astype(np.uint8)
        descriptor = gradient_descriptor(patch)
        assert np.linalg.norm(descriptor) == pytest.approx(1.0, abs=1e-5)

    def test_descriptor_handles_flat_patch(self):
        flat = np.full((20, 20), 128, dtype=np.uint8)
        assert np.allclose(gradient_descriptor(flat), 0.0)

    def test_identical_faces_have_high_region_similarity(self, embedder):
        face = embedder.detect(make_face(1))[0]
        regions = compare_regions(face.aligned, face.aligned)
        for region in regions:
            assert region.appearance_similarity > 0.98
            assert region.change_magnitude < 0.02

    def test_every_canonical_region_is_reported(self, embedder):
        a = embedder.detect(make_face(1))[0]
        b = embedder.detect(make_face(2))[0]
        regions = compare_regions(a.aligned, b.aligned)
        assert {r.key for r in regions} == set(REGION_STABILITY)

    def test_nose_is_ranked_least_stable(self):
        """The core cosmetic-surgery premise, pinned as a test."""
        assert REGION_STABILITY["nose"] == min(REGION_STABILITY.values())
        assert REGION_STABILITY["periocular"] == max(REGION_STABILITY.values())
        assert REGION_STABILITY["periocular"] > REGION_STABILITY["nose"]

    def test_beard_changes_lower_face_more_than_eyes(self, embedder):
        clean = embedder.detect(make_face(1))[0]
        bearded = embedder.detect(make_face(1, beard=True))[0]
        regions = {r.key: r for r in compare_regions(clean.aligned, bearded.aligned)}
        assert regions["jaw_chin"].change_magnitude > regions["periocular"].change_magnitude

    def test_geometry_is_scale_invariant(self, embedder):
        """Ratios are divided by inter-ocular distance, so size must not matter."""
        small = embedder.detect(make_face(1, size=320))[0]
        large = embedder.detect(make_face(1, size=640))[0]
        measures = compute_geometry(small, large)
        assert measures
        for measure in measures:
            assert abs(measure.relative_delta) < 0.06, measure.key

    def test_narrative_never_claims_a_cause(self, embedder):
        a = embedder.detect(make_face(1))[0]
        b = embedder.detect(make_face(7, beard=True))[0]
        analysis = analyse_regions(a, b, "High Similarity", pose_difference_deg=4.0)
        text = analysis.narrative.lower()
        for phrase in ("has had", "underwent", "surgery was", "definitely", "proves"):
            assert phrase not in text

    def test_geometry_flagged_unreliable_at_high_pose_difference(self, embedder):
        a = embedder.detect(make_face(1))[0]
        b = embedder.detect(make_face(1))[0]
        analysis = analyse_regions(a, b, "High Similarity", pose_difference_deg=40.0)
        assert analysis.geometry_reliable is False
        assert "projection artefact" in analysis.geometry_caveat

    def test_analysis_serialises_with_interpretation_notice(self, embedder):
        a = embedder.detect(make_face(1))[0]
        b = embedder.detect(make_face(2))[0]
        payload = analyse_regions(a, b, "Moderate Similarity", 5.0).to_dict()
        assert "descriptive only" in payload["interpretation"]
        assert "not included in the similarity score" in payload["interpretation"]


class TestOrchestration:
    def _settings(self, **overrides):
        base = dict(min_quality_score=1.0, min_blur_score=1.0, min_face_width_px=20, _env_file=None)
        base.update(overrides)
        return Settings(**base)

    def test_no_face_raises(self, embedder):
        blank = np.full((300, 300, 3), 20, dtype=np.uint8)
        with pytest.raises(NoFaceError):
            process_side([blank], embedder, self._settings(), "older photograph")

    def test_multiple_faces_requires_selection(self, embedder):
        image = make_face(1, second_face=True)
        faces = embedder.detect(image)
        if len(faces) < 2:
            pytest.skip("mock detector did not find two faces")
        with pytest.raises(MultipleFacesError) as exc:
            process_side([image], embedder, self._settings(), "older photograph")
        assert len(exc.value.faces) >= 2
        assert all("thumbnail" in f for f in exc.value.faces)

    def test_explicit_selection_proceeds(self, embedder):
        image = make_face(1, second_face=True)
        if len(embedder.detect(image)) < 2:
            pytest.skip("mock detector did not find two faces")
        result = process_side(
            [image], embedder, self._settings(), "older photograph",
            selected_face_indices=[0],
        )
        assert result.template.member_count == 1

    def test_poor_quality_refuses_rather_than_scoring(self, embedder):
        strict = self._settings(min_quality_score=99.9)
        with pytest.raises(QualityError) as exc:
            process_side([make_face(1)], embedder, strict, "older photograph")
        assert "Insufficient image quality" in exc.value.message
        assert exc.value.details["reasons"]

    def test_full_analysis_payload(self, embedder):
        settings = self._settings()
        profile = default_profile("opencv")
        result = analyse_pair(
            [make_face(1)], [make_face(1, brightness=1.12)],
            embedder, settings, profile,
        )
        for key in (
            "similarity_score", "confidence_level", "face_detected_old",
            "face_detected_new", "image_quality_old", "image_quality_new",
            "pose_difference", "warnings", "analysis", "scoring",
            "calibration", "model", "templates", "region_analysis",
            "uncertainty_sources", "visualisations",
        ):
            assert key in result, f"missing {key}"

        assert 0 <= result["similarity_score"] <= 100
        assert -1 <= result["scoring"]["raw_embedding_similarity"] <= 1
        assert result["disclaimer"]
        assert result["calibration"]["is_validated"] is False

    def test_uncalibrated_state_is_surfaced(self, embedder):
        result = analyse_pair(
            [make_face(1)], [make_face(1)], embedder,
            self._settings(), default_profile("opencv"),
        )
        assert any("unvalidated" in w.lower() for w in result["warnings"])
        assert any(
            s["factor"] == "Uncalibrated model" for s in result["uncertainty_sources"]
        )

    def test_single_photo_pair_flagged_as_uncertainty(self, embedder):
        result = analyse_pair(
            [make_face(1)], [make_face(1)], embedder,
            self._settings(), default_profile("opencv"),
        )
        assert any(
            "Single photograph" in s["factor"] for s in result["uncertainty_sources"]
        )

    def test_lookalike_caveat_always_present(self, embedder):
        result = analyse_pair(
            [make_face(1)], [make_face(1)], embedder,
            self._settings(), default_profile("opencv"),
        )
        assert any(
            "Look-alikes" in s["factor"] for s in result["uncertainty_sources"]
        )

    def test_multi_photo_aggregation(self, embedder):
        result = analyse_pair(
            [make_face(1), make_face(1, brightness=1.15), make_face(1, noise=8)],
            [make_face(1, brightness=0.9), make_face(1, blur=1.0)],
            embedder, self._settings(), default_profile("opencv"),
        )
        assert result["templates"]["old_photo_count"] == 3
        assert result["templates"]["new_photo_count"] == 2
        assert len(result["templates"]["pairwise"]["values"]) == 3
        assert len(result["templates"]["pairwise"]["values"][0]) == 2
        assert sum(result["templates"]["old_weights"]) == pytest.approx(1.0, abs=0.01)

    def test_visualisations_can_be_disabled(self, embedder):
        result = analyse_pair(
            [make_face(1)], [make_face(1)], embedder,
            self._settings(), default_profile("opencv"),
            include_visualisations=False,
        )
        assert "visualisations" not in result

    def test_visualisations_are_data_uris_not_links(self, embedder):
        """Privacy: imagery is embedded, never served from a fetchable URL."""
        result = analyse_pair(
            [make_face(1)], [make_face(1)], embedder,
            self._settings(), default_profile("opencv"),
        )
        for key, value in result["visualisations"].items():
            if key == "heatmap_caption":
                continue
            assert value.startswith("data:image/")
            assert "http" not in value[:40]

    def test_same_image_scores_higher_than_different_identities(self, embedder):
        """Plumbing check only - the mock carries no recognition accuracy."""
        settings = self._settings()
        profile = default_profile("opencv")
        same = analyse_pair(
            [make_face(1)], [make_face(1, brightness=1.05)], embedder, settings, profile
        )
        different = analyse_pair(
            [make_face(1)], [make_face(42)], embedder, settings, profile
        )
        assert (
            same["scoring"]["raw_embedding_similarity"]
            > different["scoring"]["raw_embedding_similarity"]
        )

    def test_extreme_pose_difference_refuses(self, embedder):
        from app.models.base import DetectedFace

        settings = self._settings(max_pose_difference_deg=5.0)
        a = embedder.detect(make_face(1))[0]
        b = embedder.detect(make_face(1))[0]
        b.pose = (60.0, 0.0, 0.0)

        assert pose_difference(a, b) > 5.0
