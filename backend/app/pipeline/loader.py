"""Secure image decoding.

Every uploaded byte-string passes through `decode_image`. Nothing is written to
disk at any point, so there are no temporary files to leak or clean up.
"""

from __future__ import annotations

import io
import logging

import cv2
import numpy as np
from PIL import Image, ImageOps, UnidentifiedImageError

from app.config import Settings
from app.core.errors import ImageError

logger = logging.getLogger(__name__)

# Magic-byte signatures. Trusting the client's Content-Type alone would let a
# caller mislabel an arbitrary payload as an image.
_SIGNATURES: tuple[tuple[bytes, str], ...] = (
    (b"\xff\xd8\xff", "jpeg"),
    (b"\x89PNG\r\n\x1a\n", "png"),
    (b"BM", "bmp"),
    (b"II*\x00", "tiff"),
    (b"MM\x00*", "tiff"),
)

ALLOWED_FORMATS = {"jpeg", "png", "bmp", "tiff", "webp"}


def sniff_format(data: bytes) -> str | None:
    for signature, name in _SIGNATURES:
        if data.startswith(signature):
            return name
    # WebP: "RIFF" .... "WEBP"
    if len(data) >= 12 and data[0:4] == b"RIFF" and data[8:12] == b"WEBP":
        return "webp"
    return None


def decode_image(data: bytes, settings: Settings, label: str = "image") -> np.ndarray:
    """Validate and decode image bytes into a BGR numpy array.

    Raises ImageError with a user-safe message on any failure. Strips all
    metadata (including EXIF and GPS) by re-encoding through a raw pixel array,
    after honouring EXIF orientation.
    """
    if not data:
        raise ImageError(f"The {label} file is empty.", code="empty_file")

    if len(data) > settings.max_upload_bytes:
        limit_mb = settings.max_upload_bytes / (1024 * 1024)
        raise ImageError(
            f"The {label} file exceeds the {limit_mb:.0f} MB limit.",
            code="file_too_large",
        )

    fmt = sniff_format(data)
    if fmt is None:
        raise ImageError(
            f"The {label} file is not a supported image format. "
            "Accepted formats: JPEG, PNG, WebP, BMP, TIFF.",
            code="unsupported_format",
        )

    # Guard against decompression bombs before any pixel work happens.
    previous_limit = Image.MAX_IMAGE_PIXELS
    Image.MAX_IMAGE_PIXELS = settings.max_image_pixels
    try:
        with Image.open(io.BytesIO(data)) as probe:
            probe.verify()  # structural check; consumes the file object

        with Image.open(io.BytesIO(data)) as pil_image:
            if pil_image.format is None or pil_image.format.lower() not in ALLOWED_FORMATS:
                raise ImageError(
                    f"The {label} file format ({pil_image.format}) is not supported.",
                    code="unsupported_format",
                )

            width, height = pil_image.size
            if width * height > settings.max_image_pixels:
                raise ImageError(
                    f"The {label} image has too many pixels to process safely.",
                    code="image_too_large",
                )
            if max(width, height) > settings.max_image_dimension:
                raise ImageError(
                    f"The {label} image exceeds "
                    f"{settings.max_image_dimension}px on its longest side.",
                    code="image_too_large",
                )
            if min(width, height) < settings.min_image_dimension:
                raise ImageError(
                    f"The {label} image is too small "
                    f"(minimum {settings.min_image_dimension}px per side).",
                    code="image_too_small",
                )

            # Apply EXIF orientation, then drop every other metadata field by
            # converting to raw RGB pixels.
            oriented = ImageOps.exif_transpose(pil_image)
            rgb = oriented.convert("RGB")
            array = np.asarray(rgb, dtype=np.uint8)

    except ImageError:
        raise
    except (UnidentifiedImageError, OSError, SyntaxError) as exc:
        logger.info("Rejected unreadable %s upload: %s", label, exc)
        raise ImageError(
            f"The {label} file could not be read. It may be corrupted or truncated.",
            code="corrupted_image",
        ) from exc
    except Image.DecompressionBombError as exc:
        raise ImageError(
            f"The {label} image was rejected as unsafely large.",
            code="image_too_large",
        ) from exc
    except ValueError as exc:
        raise ImageError(
            f"The {label} file could not be decoded.", code="corrupted_image"
        ) from exc
    finally:
        Image.MAX_IMAGE_PIXELS = previous_limit

    if array.ndim != 3 or array.shape[2] != 3:
        raise ImageError(
            f"The {label} image has an unexpected channel layout.",
            code="corrupted_image",
        )

    return cv2.cvtColor(array, cv2.COLOR_RGB2BGR)


def encode_preview_png(image_bgr: np.ndarray, max_side: int = 512) -> bytes:
    """Encode a BGR array to PNG bytes, downscaled for transport."""
    height, width = image_bgr.shape[:2]
    scale = min(1.0, max_side / max(height, width))
    if scale < 1.0:
        image_bgr = cv2.resize(
            image_bgr,
            (int(round(width * scale)), int(round(height * scale))),
            interpolation=cv2.INTER_AREA,
        )
    ok, buffer = cv2.imencode(".png", image_bgr)
    if not ok:  # pragma: no cover - defensive
        raise ImageError("Failed to encode preview image.", code="encoding_failed")
    return buffer.tobytes()
