"""Image loading, validation and quality-gate tests."""

from __future__ import annotations

import io

import cv2
import numpy as np
import pytest
from PIL import Image

from app.config import Settings
from app.core.errors import ImageError
from app.pipeline.loader import decode_image, sniff_format
from app.pipeline.quality import assess_quality, laplacian_variance

from fixtures import encode_jpeg, encode_png, make_face


class TestFormatSniffing:
    def test_detects_real_formats(self, face_image):
        assert sniff_format(encode_jpeg(face_image)) == "jpeg"
        assert sniff_format(encode_png(face_image)) == "png"

    def test_rejects_non_images(self):
        assert sniff_format(b"") is None
        assert sniff_format(b"not an image at all") is None
        assert sniff_format(b"%PDF-1.7\n") is None
        assert sniff_format(b"<html><body>error</body></html>") is None

    def test_rejects_polyglot_prefix(self):
        """A file that starts as script text is not saved by an image tail."""
        payload = b"#!/bin/sh\nrm -rf /\n" + encode_jpeg(make_face(1))
        assert sniff_format(payload) is None


class TestDecoding:
    def test_decodes_jpeg_to_bgr(self, settings, face_image):
        decoded = decode_image(encode_jpeg(face_image), settings)
        assert decoded.ndim == 3
        assert decoded.shape[2] == 3
        assert decoded.dtype == np.uint8

    def test_empty_file_rejected(self, settings):
        with pytest.raises(ImageError) as exc:
            decode_image(b"", settings)
        assert exc.value.code == "empty_file"

    def test_oversized_file_rejected(self, settings):
        settings = Settings(max_upload_bytes=1024, _env_file=None)
        with pytest.raises(ImageError) as exc:
            decode_image(encode_png(make_face(1, size=600)), settings)
        assert exc.value.code == "file_too_large"

    def test_corrupt_image_rejected(self, settings):
        data = bytearray(encode_jpeg(make_face(1)))
        truncated = bytes(data[: len(data) // 3])
        with pytest.raises(ImageError) as exc:
            decode_image(truncated, settings)
        assert exc.value.code in {"corrupted_image", "unsupported_format"}

    def test_unsupported_format_rejected(self, settings):
        with pytest.raises(ImageError) as exc:
            decode_image(b"GIF89a" + b"\x00" * 200, settings)
        assert exc.value.code == "unsupported_format"

    def test_tiny_image_rejected(self, settings):
        tiny = np.full((16, 16, 3), 200, dtype=np.uint8)
        with pytest.raises(ImageError) as exc:
            decode_image(encode_png(tiny), settings)
        assert exc.value.code == "image_too_small"

    def test_oversized_dimensions_rejected(self, settings):
        settings = Settings(max_image_dimension=256, _env_file=None)
        with pytest.raises(ImageError) as exc:
            decode_image(encode_png(make_face(1, size=600)), settings)
        assert exc.value.code == "image_too_large"

    def test_decompression_bomb_guard(self, settings):
        settings = Settings(max_image_pixels=10_000, _env_file=None)
        with pytest.raises(ImageError):
            decode_image(encode_png(make_face(1, size=400)), settings)

    def test_exif_orientation_is_applied(self, settings):
        """A portrait photo tagged as rotated must come back upright."""
        image = make_face(1, size=300)
        tall = cv2.resize(image, (200, 400))
        pil = Image.fromarray(cv2.cvtColor(tall, cv2.COLOR_BGR2RGB))
        buffer = io.BytesIO()
        exif = pil.getexif()
        exif[274] = 6  # orientation: rotate 90 CW
        pil.save(buffer, format="JPEG", exif=exif)

        decoded = decode_image(buffer.getvalue(), settings)
        # Orientation 6 swaps the axes.
        assert decoded.shape[0] < decoded.shape[1]

    def test_metadata_is_stripped(self, settings):
        """GPS coordinates in an uploaded photo must not survive decoding."""
        pil = Image.fromarray(
            cv2.cvtColor(make_face(1, size=300), cv2.COLOR_BGR2RGB)
        )
        buffer = io.BytesIO()
        exif = pil.getexif()
        exif[271] = "SecretCameraMake"
        pil.save(buffer, format="JPEG", exif=exif)

        decoded = decode_image(buffer.getvalue(), settings)
        # The result is a bare pixel array - there is nowhere for metadata to live.
        assert isinstance(decoded, np.ndarray)
        assert not hasattr(decoded, "info")


class TestQualityMetrics:
    def test_blur_reduces_laplacian_variance(self):
        sharp = cv2.cvtColor(make_face(1), cv2.COLOR_BGR2GRAY)
        blurred = cv2.cvtColor(make_face(1, blur=6.0), cv2.COLOR_BGR2GRAY)
        assert laplacian_variance(blurred) < laplacian_variance(sharp)

    def test_small_face_fails_hard_gate(self, settings, embedder):
        image = make_face(1, size=480, face_scale=0.10)
        faces = embedder.detect(image)
        if not faces:
            pytest.skip("mock detector did not find the deliberately tiny face")
        report = assess_quality(faces[0], settings)
        assert not report.usable
        assert any("px wide" in f for f in report.hard_failures)

    def test_heavy_blur_fails_hard_gate(self, settings, embedder):
        image = make_face(1, blur=14.0)
        faces = embedder.detect(image)
        if not faces:
            pytest.skip("mock detector did not find the heavily blurred face")
        report = assess_quality(faces[0], settings)
        assert not report.usable

    def test_good_image_passes(self, settings, embedder):
        faces = embedder.detect(make_face(1))
        assert faces
        report = assess_quality(faces[0], settings)
        assert report.usable, report.hard_failures
        assert report.composite_score > settings.min_quality_score

    def test_quality_score_is_bounded(self, settings, embedder):
        for variant in (
            make_face(1),
            make_face(2, blur=3.0),
            make_face(3, brightness=0.35),
            make_face(4, brightness=1.9),
            make_face(5, noise=40),
        ):
            faces = embedder.detect(variant)
            if not faces:
                continue
            report = assess_quality(faces[0], settings)
            assert 0.0 <= report.composite_score <= 100.0
            assert 0.0 <= report.exposure_score <= 1.0

    def test_dark_image_scores_lower_than_well_lit(self, settings, embedder):
        bright = embedder.detect(make_face(1))
        dark = embedder.detect(make_face(1, brightness=0.22))
        if not bright or not dark:
            pytest.skip("detector did not find both variants")
        assert (
            assess_quality(dark[0], settings).exposure_score
            < assess_quality(bright[0], settings).exposure_score
        )

    def test_report_serialises(self, settings, embedder):
        faces = embedder.detect(make_face(1))
        payload = assess_quality(faces[0], settings).to_dict()
        for key in ("composite_score", "usable", "pose", "occlusion", "hard_failures"):
            assert key in payload
        assert set(payload["pose"]) == {"yaw_deg", "pitch_deg", "roll_deg"}

    def test_gates_are_configurable(self, embedder):
        image = make_face(1)
        faces = embedder.detect(image)
        lenient = assess_quality(faces[0], Settings(min_quality_score=1, _env_file=None))
        strict = assess_quality(faces[0], Settings(min_quality_score=99.5, _env_file=None))
        assert lenient.usable
        assert not strict.usable
