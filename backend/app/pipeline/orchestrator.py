"""Pipeline orchestration.

Runs: decode -> detect -> align -> landmarks -> quality gate -> embed ->
template -> similarity -> calibrate -> band -> region analysis -> render.

The quality gate sits deliberately *before* embedding. If the inputs cannot
support a defensible comparison, the system refuses rather than producing a
number that looks authoritative and is not.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np

from app.config import Settings
from app.core.errors import MultipleFacesError, NoFaceError, QualityError
from app.models.base import DetectedFace, FaceEmbedder
from app.pipeline import visualize
from app.pipeline.quality import QualityReport, assess_quality
from app.pipeline.regions import analyse_regions
from app.pipeline.similarity import (
    Template,
    build_template,
    cosine_similarity,
    pairwise_similarities,
    pose_difference,
    representative_faces,
)
from app.scoring import bands
from app.scoring.calibration import CalibrationProfile

logger = logging.getLogger(__name__)


@dataclass
class SideResult:
    """Everything derived from one side (old or new) of the comparison."""

    faces: list[DetectedFace]
    qualities: list[QualityReport]
    template: Template
    representative_index: int
    images: list[np.ndarray]

    @property
    def representative_face(self) -> DetectedFace:
        return self.faces[self.representative_index]

    @property
    def representative_quality(self) -> QualityReport:
        return self.qualities[self.representative_index]


def process_side(
    images: list[np.ndarray],
    embedder: FaceEmbedder,
    settings: Settings,
    label: str,
    selected_face_indices: list[int] | None = None,
    session_token: str | None = None,
) -> SideResult:
    """Detect, gate and embed every image on one side of the comparison."""
    faces: list[DetectedFace] = []
    qualities: list[QualityReport] = []
    hard_failures: list[str] = []

    for position, image in enumerate(images):
        image_label = label if len(images) == 1 else f"{label} (photo {position + 1})"
        detected = embedder.detect(image, max_faces=settings.max_faces_returned)

        if not detected:
            raise NoFaceError(
                f"No face was detected in the {image_label}. The face may be too "
                "small, too dark, heavily occluded, or turned too far from the camera."
            )

        if len(detected) > 1:
            requested = (
                selected_face_indices[position]
                if selected_face_indices is not None and position < len(selected_face_indices)
                else None
            )
            if requested is None:
                thumbnails = [
                    {
                        "index": face.index,
                        "bbox": [round(v, 1) for v in face.bbox],
                        "detection_score": round(face.detection_score, 3),
                        "width_px": round(face.width, 1),
                        "thumbnail": visualize.render_face_thumbnail(image, face),
                    }
                    for face in detected
                ]
                raise MultipleFacesError(
                    f"{len(detected)} faces were detected in the {image_label}. "
                    "Select which face to compare.",
                    faces=thumbnails,
                    session_token=session_token,
                )
            if not 0 <= requested < len(detected):
                raise NoFaceError(
                    f"The selected face index {requested} does not exist in the "
                    f"{image_label}."
                )
            face = detected[requested]
        else:
            face = detected[0]

        quality = assess_quality(face, settings, label=image_label)
        hard_failures.extend(quality.hard_failures)
        faces.append(face)
        qualities.append(quality)

    if hard_failures:
        if settings.strict_quality_gating:
            raise QualityError(
                "Insufficient image quality for reliable comparison.",
                details={
                    "side": label,
                    "reasons": hard_failures,
                    "per_image": [q.to_dict() for q in qualities],
                },
            )
        # Downgrade: score the pair anyway, but keep the reasons visible as
        # warnings rather than silently dropping them.
        for quality in qualities:
            quality.warnings = quality.hard_failures + quality.warnings
            quality.hard_failures = []
            quality.usable = True

    aligned_batch = [face.aligned for face in faces]
    embeddings_array = embedder.embed_batch(aligned_batch)
    embeddings = [embeddings_array[i] for i in range(len(faces))]
    for face, embedding in zip(faces, embeddings):
        face.embedding = embedding

    template = build_template(embeddings, qualities)
    return SideResult(
        faces=faces,
        qualities=qualities,
        template=template,
        representative_index=representative_faces(faces, qualities),
        images=images,
    )


def _collect_warnings(
    old: SideResult,
    new: SideResult,
    pose_delta: float,
    settings: Settings,
    profile: CalibrationProfile,
    pairwise,
    hairstyle_likely: bool = False,
) -> list[str]:
    warnings: list[str] = []

    if not profile.fitted:
        warnings.append(bands.UNCALIBRATED_WARNING)

    for quality in old.qualities + new.qualities:
        warnings.extend(quality.warnings)

    if pose_delta > settings.advisory_pose_difference_deg:
        warnings.append(
            f"Head pose differs by roughly {pose_delta:.0f}° between the "
            "photographs, which typically depresses the similarity score even "
            "for genuine pairs."
        )

    if old.template.is_multi or new.template.is_multi:
        if pairwise.spread > 0.25:
            warnings.append(
                f"The individual photo-to-photo similarities vary widely "
                f"(spread {pairwise.spread:.2f}). The images disagree with one "
                "another, so the pooled score is less dependable than its single "
                "number suggests."
            )

    suspected = [
        q
        for q in old.qualities + new.qualities
        if q.suspected_sunglasses or q.suspected_mask
    ]
    if suspected:
        warnings.append(
            "Possible occlusion was detected. Occlusion detection here is "
            "heuristic and can misfire on heavy shadow, dark-framed glasses or "
            "facial hair."
        )

    if hairstyle_likely:
        warnings.append(
            "A hairstyle or hair colour change looks like the main driver of the "
            "forehead-region difference, while the periocular region stays "
            "stable. The aligned comparison crop includes a thin strip of "
            "hairline, so this can have a small effect on the similarity score - "
            "smaller than typical day-to-day photo variation, but not zero."
        )

    return warnings


def analyse_pair(
    old_images: list[np.ndarray],
    new_images: list[np.ndarray],
    embedder: FaceEmbedder,
    settings: Settings,
    profile: CalibrationProfile,
    old_face_indices: list[int] | None = None,
    new_face_indices: list[int] | None = None,
    session_token: str | None = None,
    include_visualisations: bool = True,
) -> dict:
    """Run the full comparison and return the API payload."""
    old = process_side(
        old_images, embedder, settings, "older photograph", old_face_indices, session_token
    )
    new = process_side(
        new_images, embedder, settings, "newer photograph", new_face_indices, session_token
    )

    raw_similarity = cosine_similarity(old.template.embedding, new.template.embedding)
    pairwise = pairwise_similarities(
        [f.embedding for f in old.faces], [f.embedding for f in new.faces]
    )

    llr = profile.log_likelihood_ratio(raw_similarity)
    calibrated = profile.calibrated_score(raw_similarity, settings.prior_same_person)
    band = bands.band_for(calibrated, settings)

    old_face = old.representative_face
    new_face = new.representative_face
    pose_delta = pose_difference(old_face, new_face)

    if pose_delta > settings.max_pose_difference_deg and settings.strict_quality_gating:
        raise QualityError(
            "Insufficient image quality for reliable comparison.",
            details={
                "side": "pair",
                "reasons": [
                    f"Head pose differs by {pose_delta:.0f}° between the two "
                    f"photographs, beyond the {settings.max_pose_difference_deg:.0f}° "
                    "limit for a defensible comparison."
                ],
                "pose_difference_deg": round(pose_delta, 1),
            },
        )

    region_analysis = analyse_regions(
        old_face, new_face, band.label, pose_delta
    )

    warnings = _collect_warnings(
        old, new, pose_delta, settings, profile, pairwise, region_analysis.hairstyle_likely
    )

    payload: dict = {
        "similarity_score": round(calibrated, 1),
        "confidence_level": band.label,
        "confidence_key": band.key,
        "statement": band.statement,
        "guidance": band.guidance,
        "disclaimer": bands.UNIVERSAL_DISCLAIMER,
        "method_statement": bands.METHOD_STATEMENT,
        "face_detected_old": True,
        "face_detected_new": True,
        "image_quality_old": old.representative_quality.to_dict(),
        "image_quality_new": new.representative_quality.to_dict(),
        "pose_difference": round(pose_delta, 1),
        "warnings": warnings,
        "analysis": region_analysis.narrative,
        "scoring": {
            "raw_embedding_similarity": round(raw_similarity, 4),
            "metric": "cosine",
            "calibrated_similarity_score": round(calibrated, 1),
            "log_likelihood_ratio": round(llr, 3),
            "likelihood_ratio": round(float(np.exp(llr)), 2),
            "evidence_statement": bands.describe_evidence_strength(llr),
            "prior_used": settings.prior_same_person,
            "prior_note": (
                "The percentage is derived from the likelihood ratio combined with "
                "an assumed prior probability of "
                f"{settings.prior_same_person:.2f}. Change PRIOR_SAME_PERSON to "
                "reflect how your pairs are actually selected; the likelihood "
                "ratio itself does not depend on it."
            ),
            "decision_threshold_raw": round(
                profile.decision_threshold(settings.prior_same_person), 4
            ),
            "bands": {
                "very_high": settings.band_very_high,
                "high": settings.band_high,
                "moderate": settings.band_moderate,
            },
        },
        "calibration": {
            "profile": profile.name,
            "method": profile.method,
            "is_validated": profile.fitted,
            "n_genuine_pairs": profile.n_genuine_pairs,
            "n_impostor_pairs": profile.n_impostor_pairs,
            "fitted_at": profile.fitted_at,
            "dataset_description": profile.dataset_description,
            "notes": profile.notes,
            "metrics": profile.metrics,
        },
        "model": {
            "backend": embedder.info.backend,
            "detector": embedder.info.detector,
            "recognizer": embedder.info.recognizer,
            "embedding_dim": embedder.info.embedding_dim,
            "license": embedder.info.license,
            "commercial_use_permitted": embedder.info.commercial_use,
            "notes": embedder.info.notes,
        },
        "templates": {
            "old_photo_count": old.template.member_count,
            "new_photo_count": new.template.member_count,
            "old_weights": [round(w, 3) for w in old.template.weights],
            "new_weights": [round(w, 3) for w in new.template.weights],
            "aggregation": "quality-weighted mean of L2-normalised embeddings",
            "pairwise": pairwise.to_dict(),
        },
        "region_analysis": region_analysis.to_dict(),
        "uncertainty_sources": _uncertainty_sources(
            old, new, pose_delta, profile, pairwise, region_analysis.hairstyle_likely
        ),
    }

    if include_visualisations:
        payload["visualisations"] = {
            "old_annotated": visualize.to_data_uri(
                visualize.annotate_detection(
                    old.images[old.representative_index], old_face
                )
            ),
            "new_annotated": visualize.to_data_uri(
                visualize.annotate_detection(
                    new.images[new.representative_index], new_face
                )
            ),
            "old_aligned": visualize.to_data_uri(visualize.render_aligned(old_face)),
            "new_aligned": visualize.to_data_uri(visualize.render_aligned(new_face)),
            "change_heatmap": visualize.to_data_uri(
                visualize.render_region_heatmap(new_face, region_analysis.regions)
            ),
            "heatmap_caption": (
                "Warmer areas differ more in appearance between the two aligned "
                "faces. This reflects the region measurements in the table below - "
                "it is not a map of what the recognition model attended to, because "
                "no such map is available for this model."
            ),
        }

    return payload


def analyse_pair_with_fallback(
    old_images: list[np.ndarray],
    new_images: list[np.ndarray],
    embedder: FaceEmbedder,
    settings: Settings,
    profile: CalibrationProfile,
    old_face_indices: list[int] | None = None,
    new_face_indices: list[int] | None = None,
    session_token: str | None = None,
    include_visualisations: bool = True,
) -> dict:
    """Run `analyse_pair`, retrying once with the OpenCV backend if the
    primary backend detects zero faces.

    Only triggers on a genuine detection failure (NoFaceError), never on
    quality gates or multiple-face disambiguation - those outcomes are
    meaningful regardless of backend and retrying would just be confusing.
    Both images in a pair are always embedded by the *same* backend: mixing
    embeddings from two different networks would make the cosine similarity
    meaningless, so a fallback always restarts the whole comparison, never
    just the image that failed.
    """
    try:
        return analyse_pair(
            old_images=old_images,
            new_images=new_images,
            embedder=embedder,
            settings=settings,
            profile=profile,
            old_face_indices=old_face_indices,
            new_face_indices=new_face_indices,
            session_token=session_token,
            include_visualisations=include_visualisations,
        )
    except NoFaceError as original_error:
        if not settings.enable_detection_fallback or embedder.info.backend == "opencv":
            raise

        try:
            from app.models.registry import get_fallback_embedder
            from app.scoring.calibration import load_profile

            logger.info(
                "Primary backend (%s) detected no face; retrying with fallback "
                "OpenCV backend.",
                embedder.info.backend,
            )
            fallback_embedder = get_fallback_embedder(settings)
            fallback_profile = load_profile(
                settings.calibration_dir,
                settings.calibration_profile,
                fallback_embedder.info.backend,
            )
            result = analyse_pair(
                old_images=old_images,
                new_images=new_images,
                embedder=fallback_embedder,
                settings=settings,
                profile=fallback_profile,
                old_face_indices=old_face_indices,
                new_face_indices=new_face_indices,
                session_token=session_token,
                include_visualisations=include_visualisations,
            )
        except NoFaceError:
            # The fallback backend found nothing either: the original failure
            # is the more informative one to surface.
            raise original_error from None
        except Exception as exc:
            # Fallback unavailable for some other reason (missing weights,
            # import error, ...). Do not let that mask the real answer with a
            # confusing 500 - report the original detection failure instead.
            logger.warning(
                "Fallback backend unavailable (%s); returning original detection failure.",
                exc,
            )
            raise original_error from None

        result["warnings"] = [
            f"The primary model ({embedder.info.backend}) could not detect a face "
            "in one of these photographs. This comparison used the fallback "
            f"{fallback_embedder.info.backend} model instead, which is less "
            "accurate on large age gaps and post-surgical pairs."
        ] + result["warnings"]
        return result


def _uncertainty_sources(
    old: SideResult,
    new: SideResult,
    pose_delta: float,
    profile: CalibrationProfile,
    pairwise,
    hairstyle_likely: bool = False,
) -> list[dict[str, str]]:
    """The major things that could make this result wrong, ranked."""
    sources: list[dict[str, str]] = []

    if not profile.fitted:
        sources.append(
            {
                "factor": "Uncalibrated model",
                "severity": "high",
                "detail": (
                    "No validation data has been fitted, so the mapping from raw "
                    "similarity to percentage is an assumption rather than a "
                    "measurement. This is currently the largest source of error."
                ),
            }
        )

    worst_quality = min(
        q.composite_score for q in old.qualities + new.qualities
    )
    if worst_quality < 55:
        sources.append(
            {
                "factor": "Image quality",
                "severity": "high" if worst_quality < 40 else "moderate",
                "detail": (
                    f"The weakest image scores {worst_quality:.0f}/100 on the "
                    "composite quality measure. Low quality compresses genuine and "
                    "impostor scores toward each other."
                ),
            }
        )

    if pose_delta > 15:
        sources.append(
            {
                "factor": "Pose difference",
                "severity": "high" if pose_delta > 30 else "moderate",
                "detail": (
                    f"Roughly {pose_delta:.0f}° of head-pose difference. Pose is one "
                    "of the strongest nuisance factors in face comparison."
                ),
            }
        )

    if pairwise.spread > 0.25:
        sources.append(
            {
                "factor": "Inconsistent photographs",
                "severity": "moderate",
                "detail": (
                    f"Photo-to-photo similarity varies by {pairwise.spread:.2f}, so "
                    "the pooled score conceals real disagreement between images."
                ),
            }
        )

    if old.template.member_count == 1 and new.template.member_count == 1:
        sources.append(
            {
                "factor": "Single photograph per side",
                "severity": "moderate",
                "detail": (
                    "One image per side gives no way to distinguish a genuine "
                    "facial characteristic from an artefact of that particular "
                    "photograph. Additional images materially improve reliability."
                ),
            }
        )

    sources.append(
        {
            "factor": "Look-alikes and relatives",
            "severity": "inherent",
            "detail": (
                "Close relatives, and identical twins especially, can produce high "
                "similarity scores. No face-comparison model resolves this, and a "
                "high score never rules it out."
            ),
        }
    )

    if hairstyle_likely:
        sources.append(
            {
                "factor": "Hairstyle or hair colour change",
                "severity": "low",
                "detail": (
                    "The forehead region differs substantially while the "
                    "periocular region does not, a pattern typical of a hairstyle "
                    "or colour change. The aligned comparison crop includes a thin "
                    "strip of hairline, so this can slightly depress the "
                    "similarity score even for the same person."
                ),
            }
        )

    return sources
