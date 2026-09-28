"""API routes."""

from __future__ import annotations

import json
import logging

from fastapi import APIRouter, Depends, File, Form, Request, UploadFile

from app.config import Settings
from app.core.errors import ImageError, MultipleFacesError, NoFaceError, SessionError
from app.core.store import EphemeralStore
from app.models.base import FaceEmbedder
from app.models.registry import get_load_error, is_loaded
from app.pipeline import visualize
from app.pipeline import database_search
from app.pipeline.database_search import run_database_search
from app.pipeline.detection import detect_with_fallback
from app.pipeline.loader import decode_image
from app.pipeline.orchestrator import analyse_pair_with_fallback
from app.pipeline.quality import assess_quality
from app.schemas import (
    AnalyzeResponse,
    DatabaseSearchResponse,
    DatabaseSearchStatus,
    DetectResponse,
    HealthResponse,
)
from app.scoring.calibration import CalibrationProfile, load_profile
from app.api.deps import (
    calibration_dependency,
    embedder_dependency,
    settings_dependency,
    store_dependency,
)

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/api")

VERSION = "1.0.0"


# ---------------------------------------------------------------- health ---

@router.get("/health", response_model=HealthResponse, tags=["system"])
async def health(
    request: Request, settings: Settings = Depends(settings_dependency)
) -> dict:
    """Liveness plus an honest statement of what this instance can and cannot do."""
    notes: list[str] = []
    model_info = None
    calibration_validated = False
    profile_name = settings.calibration_profile

    loaded = is_loaded()
    if loaded:
        embedder = embedder_dependency()
        info = embedder.info
        model_info = {
            "backend": info.backend,
            "detector": info.detector,
            "recognizer": info.recognizer,
            "embedding_dim": info.embedding_dim,
            "license": info.license,
            "commercial_use_permitted": info.commercial_use,
            "notes": info.notes,
        }
        profile = load_profile(
            settings.calibration_dir, settings.calibration_profile, info.backend
        )
        calibration_validated = profile.fitted
        profile_name = profile.name

        if not info.commercial_use:
            notes.append(
                "The active model's pretrained weights are licensed for "
                "non-commercial research use only."
            )
        if not calibration_validated:
            notes.append(
                "Calibration is unvalidated. Similarity percentages are not "
                "meaningful until fitted on representative data."
            )
    else:
        error = get_load_error()
        notes.append(
            f"Model not yet loaded. {error}" if error else "Model loads on first request."
        )

    if settings.api_key is None:
        notes.append("No API key configured: this instance accepts unauthenticated requests.")

    store: EphemeralStore | None = getattr(request.app.state, "store", None)

    return {
        "status": "ok",
        "version": VERSION,
        "model_loaded": loaded,
        "model": model_info,
        "calibration_validated": calibration_validated,
        "calibration_profile": profile_name,
        "active_sessions": store.size if store else 0,
        "notes": notes,
    }


# ---------------------------------------------------------------- detect ---

@router.post("/face-detect", response_model=DetectResponse, tags=["analysis"])
async def face_detect(
    image: UploadFile = File(..., description="A single photograph."),
    settings: Settings = Depends(settings_dependency),
    embedder: FaceEmbedder = Depends(embedder_dependency),
) -> dict:
    """Detect faces in one image and report per-face quality.

    Used by the interface to show a live "usable face detected" state before the
    user commits to a full analysis. Computes no embeddings.

    If the primary backend finds no face at all, retries once with the
    fallback OpenCV backend before reporting failure - this mirrors
    `analyse_pair_with_fallback`, so the live upload check and the full
    analysis never disagree about whether a photo is usable.
    """
    data = await image.read()
    decoded = decode_image(data, settings, label="image")

    faces, used_embedder, used_fallback = detect_with_fallback(embedder, decoded, settings)
    detector_backend = used_embedder.info.backend

    if not faces:
        raise NoFaceError(
            "No face was detected in this image. The face may be too small, too "
            "dark, heavily occluded, or turned too far from the camera."
        )

    payload_faces = []
    warnings: list[str] = []
    for face in faces:
        quality = assess_quality(face, settings, label="image")
        warnings.extend(quality.hard_failures)
        warnings.extend(quality.warnings)
        payload_faces.append(
            {
                "index": face.index,
                "bbox": [round(v, 1) for v in face.bbox],
                "detection_score": round(face.detection_score, 3),
                "width_px": round(face.width, 1),
                "quality": quality.to_dict(),
                "thumbnail": visualize.render_face_thumbnail(decoded, face),
            }
        )

    if len(faces) > 1:
        warnings.append(
            f"{len(faces)} faces were detected. You will need to choose which one "
            "to compare."
        )

    if used_fallback:
        warnings.insert(
            0,
            f"The primary model ({embedder.info.backend}) could not detect a face "
            f"in this photograph. This preview used the fallback {detector_backend} "
            "model instead, which is less accurate on large age gaps and "
            "post-surgical pairs.",
        )

    preview = visualize.to_data_uri(visualize.annotate_detection(decoded, faces[0]))

    # Deduplicate while preserving order.
    seen: set[str] = set()
    unique_warnings = [w for w in warnings if not (w in seen or seen.add(w))]

    return {
        "face_count": len(faces),
        "faces": payload_faces,
        "session_token": None,
        "preview": preview,
        "warnings": unique_warnings,
    }


# ------------------------------------------------------- database search ---

@router.get(
    "/database-search/status", response_model=DatabaseSearchStatus, tags=["analysis"]
)
async def database_search_status(
    settings: Settings = Depends(settings_dependency),
) -> dict:
    """Whether one-to-many search is turned on, and the current index size.

    Cheap: reports the last-built index rather than building one, so the
    interface can poll this before committing to a search. Never raises on a
    missing model - an unloaded or disabled instance should still be able to
    report its own status.
    """
    embedder = None
    if settings.enable_database_search and is_loaded():
        try:
            embedder = embedder_dependency()
        except Exception:
            embedder = None
    return database_search.status(settings, embedder)


@router.post(
    "/database-search", response_model=DatabaseSearchResponse, tags=["analysis"]
)
async def database_search_route(
    image: UploadFile = File(..., description="The photo to search for."),
    face_index: int | None = Form(
        None, description="Which detected face to search for, if the photo has several."
    ),
    top_n: int | None = Form(None, description="How many ranked matches to return."),
    refresh_index: bool = Form(
        False, description="Rescan the database folder for new/changed files first."
    ),
    settings: Settings = Depends(settings_dependency),
    embedder: FaceEmbedder = Depends(embedder_dependency),
    profile: CalibrationProfile = Depends(calibration_dependency),
) -> dict:
    """Search one photo against the local database folder and rank matches.

    Off unless `ENABLE_DATABASE_SEARCH=true` and `DATABASE_SEARCH_DIR` point
    at a real folder - see docs/ETHICS.md for why this is opt-in and scoped
    to a folder the operator names explicitly, never a public or shared
    image set.
    """
    data = await image.read()
    return run_database_search(
        image_bytes=data,
        settings=settings,
        embedder=embedder,
        profile=profile,
        face_index=face_index,
        top_n=top_n,
        refresh_index=refresh_index,
    )


# --------------------------------------------------------------- analyze ---

def _parse_index_list(raw: str | None, count: int, label: str) -> list[int] | None:
    """Parse the face-selection JSON array sent alongside the uploads."""
    if not raw:
        return None
    try:
        parsed = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise ImageError(
            f"{label} must be a JSON array of integers, e.g. [0].",
            code="invalid_selection",
        ) from exc
    if not isinstance(parsed, list) or not all(isinstance(v, int) for v in parsed):
        raise ImageError(
            f"{label} must be a JSON array of integers, e.g. [0].",
            code="invalid_selection",
        )
    if len(parsed) != count:
        raise ImageError(
            f"{label} has {len(parsed)} entries but {count} photographs were "
            "supplied for that side.",
            code="invalid_selection",
        )
    return parsed


@router.post("/analyze", response_model=AnalyzeResponse, tags=["analysis"])
async def analyze(
    old_photos: list[UploadFile] = File(
        ..., description="One or more photographs of the earlier appearance."
    ),
    new_photos: list[UploadFile] = File(
        ..., description="One or more photographs of the later appearance."
    ),
    old_face_indices: str | None = Form(
        None, description="JSON array selecting a face per old photo, e.g. [0,1]."
    ),
    new_face_indices: str | None = Form(None),
    include_visualisations: bool = Form(True),
    settings: Settings = Depends(settings_dependency),
    embedder: FaceEmbedder = Depends(embedder_dependency),
    profile: CalibrationProfile = Depends(calibration_dependency),
    store: EphemeralStore = Depends(store_dependency),
) -> dict:
    """Compare two sets of photographs and return a calibrated similarity assessment.

    Accepts multiple photographs per side; they are pooled into a
    quality-weighted template. A single photograph per side is the degenerate
    case of the same code path.
    """
    if len(old_photos) > settings.max_files_per_side:
        raise ImageError(
            f"At most {settings.max_files_per_side} photographs are accepted per side.",
            code="too_many_files",
        )
    if len(new_photos) > settings.max_files_per_side:
        raise ImageError(
            f"At most {settings.max_files_per_side} photographs are accepted per side.",
            code="too_many_files",
        )

    old_images = [
        decode_image(await f.read(), settings, label=f"older photograph {i + 1}")
        for i, f in enumerate(old_photos)
    ]
    new_images = [
        decode_image(await f.read(), settings, label=f"newer photograph {i + 1}")
        for i, f in enumerate(new_photos)
    ]

    old_indices = _parse_index_list(old_face_indices, len(old_images), "old_face_indices")
    new_indices = _parse_index_list(new_face_indices, len(new_images), "new_face_indices")

    # Stash the decoded images so that, if analysis stops to ask which face to
    # use, the answer does not require a re-upload. On any other outcome the
    # copies are discarded immediately.
    token = store.put(old_images, new_images) if settings.retain_for_face_selection else ""

    try:
        result = analyse_pair_with_fallback(
            old_images=old_images,
            new_images=new_images,
            embedder=embedder,
            settings=settings,
            profile=profile,
            old_face_indices=old_indices,
            new_face_indices=new_indices,
            session_token=token or None,
            include_visualisations=include_visualisations,
        )
    except MultipleFacesError:
        # The only case where the retained copies are still needed: the caller
        # will resume with a face selection.
        raise
    except Exception:
        if token:
            store.drop(token)
        raise

    if token:
        store.drop(token)
    return result


@router.post("/analyze/resume", response_model=AnalyzeResponse, tags=["analysis"])
async def analyze_resume(
    session_token: str = Form(...),
    old_face_indices: str | None = Form(None),
    new_face_indices: str | None = Form(None),
    include_visualisations: bool = Form(True),
    settings: Settings = Depends(settings_dependency),
    embedder: FaceEmbedder = Depends(embedder_dependency),
    profile: CalibrationProfile = Depends(calibration_dependency),
    store: EphemeralStore = Depends(store_dependency),
) -> dict:
    """Continue an analysis after the user has chosen which face to compare."""
    entry = store.get(session_token)
    if entry is None:
        raise SessionError(
            "That selection has expired. Uploaded images are held for at most "
            f"{settings.session_ttl_seconds // 60} minutes. Please upload again."
        )

    old_indices = _parse_index_list(
        old_face_indices, len(entry.old_images), "old_face_indices"
    )
    new_indices = _parse_index_list(
        new_face_indices, len(entry.new_images), "new_face_indices"
    )

    result = analyse_pair_with_fallback(
        old_images=entry.old_images,
        new_images=entry.new_images,
        embedder=embedder,
        settings=settings,
        profile=profile,
        old_face_indices=old_indices,
        new_face_indices=new_indices,
        session_token=session_token,
        include_visualisations=include_visualisations,
    )
    store.drop(session_token)
    return result


@router.delete("/session/{session_token}", tags=["privacy"])
async def drop_session(
    session_token: str, store: EphemeralStore = Depends(store_dependency)
) -> dict:
    """Discard retained images immediately.

    The interface calls this when the user navigates away, so retention ends at
    the moment of abandonment rather than at TTL expiry.
    """
    store.drop(session_token)
    return {"status": "discarded"}
