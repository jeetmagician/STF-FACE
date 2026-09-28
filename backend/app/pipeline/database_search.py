"""One-to-many search of a query photo against a local folder of photographs.

Deliberately scoped: `run_database_search` only ever reads the single folder
the operator points at explicitly (`Settings.database_search_dir`), the
feature is off unless `Settings.enable_database_search` is set, and no
indexed file is ever exposed by path or URL - matches come back as a
server-rendered thumbnail plus the file path as text, the same policy the
rest of this service applies to uploaded photographs. It is a
personal-library lookup tool, not infrastructure for identifying people at
scale. See docs/ETHICS.md.

The index is built with, and only ever searched with, the active
MODEL_BACKEND - never the OpenCV fallback. A photo (query or library) that
only the fallback can find a face in is skipped rather than mixed in: cosine
similarity between embeddings from two different networks is meaningless,
and calibration is fitted per backend, so a mixed-backend index could not be
scored honestly. See `pipeline.detection.detect_with_fallback`, which is
deliberately not used here.
"""

from __future__ import annotations

import json
import logging
import threading
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path

import cv2
import numpy as np

from app.config import Settings
from app.core.errors import (
    DatabaseSearchConfigError,
    DatabaseSearchDisabledError,
    MultipleFacesError,
    NoFaceError,
)
from app.models.base import FaceEmbedder
from app.pipeline import visualize
from app.pipeline.loader import decode_image
from app.scoring import bands
from app.scoring.calibration import CalibrationProfile

logger = logging.getLogger(__name__)

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}


@dataclass
class IndexEntry:
    path: str
    mtime: float
    size: int
    embedding: list[float]
    detection_score: float
    bbox: list[float]


@dataclass
class DatabaseIndex:
    backend: str
    embedding_dim: int
    entries: list[IndexEntry] = field(default_factory=list)
    skipped_no_face: int = 0
    skipped_error: int = 0
    built_at: float = 0.0

    def matrix(self) -> np.ndarray:
        if not self.entries:
            return np.zeros((0, self.embedding_dim), dtype=np.float32)
        return np.array([e.embedding for e in self.entries], dtype=np.float32)


_index: DatabaseIndex | None = None
_index_lock = threading.Lock()


def _cache_path(settings: Settings, backend: str) -> Path:
    settings.database_search_cache_dir.mkdir(parents=True, exist_ok=True)
    return settings.database_search_cache_dir / f"index_{backend}.json"


def _load_cache(settings: Settings, backend: str) -> dict[str, IndexEntry]:
    path = _cache_path(settings, backend)
    if not path.exists():
        return {}
    try:
        raw = json.loads(path.read_text())
        if raw.get("backend") != backend:
            return {}
        return {e["path"]: IndexEntry(**e) for e in raw.get("entries", [])}
    except Exception as exc:
        logger.warning("Could not read database-search cache (%s); rebuilding.", exc)
        return {}


def _save_cache(settings: Settings, index: DatabaseIndex) -> None:
    path = _cache_path(settings, index.backend)
    payload = {
        "backend": index.backend,
        "embedding_dim": index.embedding_dim,
        "built_at": index.built_at,
        "entries": [asdict(e) for e in index.entries],
    }
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(payload))
    tmp.replace(path)


def _iter_image_files(root: Path, max_images: int):
    count = 0
    for candidate in sorted(root.rglob("*")):
        if count >= max_images:
            logger.warning(
                "database_search_max_images (%d) reached; remaining files "
                "under %s were not indexed.",
                max_images,
                root,
            )
            break
        if candidate.is_file() and candidate.suffix.lower() in IMAGE_EXTENSIONS:
            yield candidate
            count += 1


def build_index(
    settings: Settings, embedder: FaceEmbedder, force: bool = False
) -> DatabaseIndex:
    """Build or incrementally refresh the on-disk embedding cache.

    Only files whose (path, size, mtime) changed since the last build are
    re-detected and re-embedded; everything else is reused from cache. A
    photo no face can be found in is skipped, not retried on every call -
    pass `force=True` (e.g. after adding many photos at once) to rebuild
    from scratch and give skipped photos another chance.
    """
    root = settings.database_search_dir
    if root is None or not root.is_dir():
        raise DatabaseSearchConfigError(
            f"DATABASE_SEARCH_DIR is not set to an existing folder ({root})."
        )

    backend = embedder.info.backend
    cached = {} if force else _load_cache(settings, backend)

    entries: list[IndexEntry] = []
    skipped_no_face = 0
    skipped_error = 0

    for file_path in _iter_image_files(root, settings.database_search_max_images):
        try:
            stat = file_path.stat()
        except OSError:
            continue
        key = str(file_path)
        prior = cached.get(key)
        if (
            prior is not None
            and prior.mtime == stat.st_mtime
            and prior.size == stat.st_size
        ):
            entries.append(prior)
            continue

        try:
            decoded = decode_image(
                file_path.read_bytes(), settings, label=file_path.name
            )
            faces = embedder.detect(decoded, max_faces=1)
            if not faces:
                skipped_no_face += 1
                continue
            face = faces[0]
            embedding = embedder.embed(face.aligned)
            entries.append(
                IndexEntry(
                    path=key,
                    mtime=stat.st_mtime,
                    size=stat.st_size,
                    embedding=[float(x) for x in embedding],
                    detection_score=float(face.detection_score),
                    bbox=[round(v, 1) for v in face.bbox],
                )
            )
        except Exception as exc:
            skipped_error += 1
            logger.warning(
                "Skipping %s during database-search indexing: %s", file_path, exc
            )

    index = DatabaseIndex(
        backend=backend,
        embedding_dim=embedder.info.embedding_dim,
        entries=entries,
        skipped_no_face=skipped_no_face,
        skipped_error=skipped_error,
        built_at=time.time(),
    )
    _save_cache(settings, index)
    logger.info(
        "Database-search index ready: %d photo(s) indexed, %d skipped (no "
        "face), %d skipped (error), backend=%s.",
        len(entries),
        skipped_no_face,
        skipped_error,
        backend,
    )
    return index


def get_index(
    settings: Settings, embedder: FaceEmbedder, refresh: bool = False
) -> DatabaseIndex:
    """In-memory singleton, rebuilt (incrementally) on first use or on request."""
    global _index
    with _index_lock:
        if _index is not None and not refresh and _index.backend == embedder.info.backend:
            return _index
        _index = build_index(settings, embedder, force=False)
        return _index


def reset() -> None:
    """Drop the cached index. Used by tests."""
    global _index
    with _index_lock:
        _index = None


def status(settings: Settings) -> dict:
    """Report configuration and, if one has been built, the current index -
    without triggering a build. Building can be slow on a large folder, so
    the status check the UI polls before a search must stay cheap."""
    with _index_lock:
        idx = _index
    directory = settings.database_search_dir
    return {
        "enabled": settings.enable_database_search,
        "directory": str(directory) if directory else None,
        "directory_exists": bool(directory and directory.is_dir()),
        "indexed_photo_count": len(idx.entries) if idx else None,
        "skipped_no_face": idx.skipped_no_face if idx else None,
        "skipped_error": idx.skipped_error if idx else None,
        "backend": idx.backend if idx else None,
        "built_at": idx.built_at if idx else None,
    }


def _thumbnail_from_bbox(image_bgr: np.ndarray, bbox: list[float], size: int = 128) -> str:
    """Crop a padded thumbnail around a stored bounding box.

    Mirrors `visualize.render_face_thumbnail`'s crop, but from a bbox alone -
    library photos are re-read from disk at search time (only for the
    handful of results actually returned), and re-running full detection on
    them just to get a `DetectedFace` object would be wasted work.
    """
    x1, y1, x2, y2 = bbox
    pad_x = (x2 - x1) * 0.18
    pad_y = (y2 - y1) * 0.18
    height, width = image_bgr.shape[:2]

    cx1 = max(0, int(round(x1 - pad_x)))
    cy1 = max(0, int(round(y1 - pad_y)))
    cx2 = min(width, int(round(x2 + pad_x)))
    cy2 = min(height, int(round(y2 + pad_y)))

    crop = image_bgr if cx2 <= cx1 or cy2 <= cy1 else image_bgr[cy1:cy2, cx1:cx2]
    crop = cv2.resize(crop, (size, size), interpolation=cv2.INTER_AREA)
    return visualize.to_data_uri(crop, max_side=size)


def run_database_search(
    image_bytes: bytes,
    settings: Settings,
    embedder: FaceEmbedder,
    profile: CalibrationProfile,
    face_index: int | None = None,
    top_n: int | None = None,
    refresh_index: bool = False,
) -> dict:
    if not settings.enable_database_search:
        raise DatabaseSearchDisabledError(
            "Database search is disabled on this instance. Set "
            "ENABLE_DATABASE_SEARCH=true and DATABASE_SEARCH_DIR to turn it on."
        )

    decoded = decode_image(image_bytes, settings, label="query photo")
    faces = embedder.detect(decoded, max_faces=settings.max_faces_returned)
    if not faces:
        raise NoFaceError(
            "No face was detected in the query photo. The face may be too "
            "small, too dark, heavily occluded, or turned too far from the "
            f"camera. Database search needs the active model ({embedder.info.backend}) "
            "to find the face directly - it does not use the fallback backend, "
            "because mixing embeddings from two different models would make "
            "every similarity score in the results meaningless."
        )

    if len(faces) > 1:
        if face_index is None:
            thumbnails = [
                {
                    "index": face.index,
                    "bbox": [round(v, 1) for v in face.bbox],
                    "detection_score": round(face.detection_score, 3),
                    "width_px": round(face.width, 1),
                    "thumbnail": visualize.render_face_thumbnail(decoded, face),
                }
                for face in faces
            ]
            raise MultipleFacesError(
                f"{len(faces)} faces were detected in the query photo. Select "
                "which face to search for.",
                faces=thumbnails,
            )
        if not 0 <= face_index < len(faces):
            raise NoFaceError(
                f"The selected face index {face_index} does not exist in the query photo."
            )
        face = faces[face_index]
    else:
        face = faces[0]

    query_embedding = embedder.embed(face.aligned)

    index = get_index(settings, embedder, refresh=refresh_index)

    requested_top_n = top_n or settings.database_search_top_n
    requested_top_n = max(1, min(requested_top_n, settings.database_search_max_results))

    matrix = index.matrix()
    query_preview = visualize.render_face_thumbnail(decoded, face, size=160)

    if matrix.shape[0] == 0:
        return {
            "query_preview": query_preview,
            "indexed_photo_count": 0,
            "skipped_no_face": index.skipped_no_face,
            "skipped_error": index.skipped_error,
            "matches": [],
            "warnings": [
                "No searchable photographs were found in the database folder "
                f"({settings.database_search_dir})."
            ],
        }

    query_norm = query_embedding / max(float(np.linalg.norm(query_embedding)), 1e-10)
    library_norms = matrix / np.clip(
        np.linalg.norm(matrix, axis=1, keepdims=True), 1e-10, None
    )
    similarities = library_norms @ query_norm

    order = np.argsort(-similarities)[:requested_top_n]

    matches = []
    for rank, position in enumerate(order):
        entry = index.entries[int(position)]
        raw_similarity = float(similarities[position])
        calibrated = profile.calibrated_score(raw_similarity, settings.prior_same_person)
        band = bands.band_for(calibrated, settings)

        try:
            library_image = decode_image(
                Path(entry.path).read_bytes(), settings, label="library photo"
            )
            thumbnail = _thumbnail_from_bbox(library_image, entry.bbox)
        except Exception as exc:
            logger.warning("Could not render thumbnail for %s: %s", entry.path, exc)
            thumbnail = ""

        matches.append(
            {
                "rank": rank + 1,
                "path": entry.path,
                "filename": Path(entry.path).name,
                "raw_similarity": round(raw_similarity, 4),
                "similarity_score": round(calibrated, 1),
                "confidence_level": band.label,
                "confidence_key": band.key,
                "thumbnail": thumbnail,
            }
        )

    warnings = [
        "This searches only the local folder configured as DATABASE_SEARCH_DIR "
        "on this machine, not any external or shared data. A high score here "
        "carries the same caveats as any other comparison in this tool: "
        "look-alikes and relatives can score highly, and this is not proof of "
        "identity.",
    ]
    if not profile.fitted:
        warnings.append(bands.UNCALIBRATED_WARNING)
    if index.skipped_no_face:
        warnings.append(
            f"{index.skipped_no_face} photo(s) in the database folder had no "
            "detectable face and were excluded from the search."
        )

    return {
        "query_preview": query_preview,
        "indexed_photo_count": len(index.entries),
        "skipped_no_face": index.skipped_no_face,
        "skipped_error": index.skipped_error,
        "matches": matches,
        "warnings": warnings,
    }
