"""Typed application errors.

Each carries a stable machine-readable `code` so the frontend can branch on
failure type without string-matching prose.
"""

from __future__ import annotations


class FacetError(Exception):
    """Base class for expected, user-facing failures."""

    http_status = 400
    default_code = "error"

    def __init__(self, message: str, code: str | None = None) -> None:
        super().__init__(message)
        self.message = message
        self.code = code or self.default_code

    def to_dict(self) -> dict[str, str]:
        return {"error": self.code, "detail": self.message}


class ImageError(FacetError):
    """An upload could not be validated or decoded."""

    http_status = 422
    default_code = "invalid_image"


class NoFaceError(FacetError):
    http_status = 422
    default_code = "no_face_detected"


class MultipleFacesError(FacetError):
    """Several faces found and none was selected: the caller must disambiguate."""

    http_status = 409
    default_code = "multiple_faces_detected"

    def __init__(self, message: str, faces: list[dict], session_token: str | None = None):
        super().__init__(message)
        self.faces = faces
        self.session_token = session_token

    def to_dict(self) -> dict:
        payload = super().to_dict()
        payload["faces"] = self.faces
        if self.session_token:
            payload["session_token"] = self.session_token
        return payload


class QualityError(FacetError):
    """Image quality is too poor for a defensible comparison."""

    http_status = 422
    default_code = "insufficient_quality"

    def __init__(self, message: str, details: dict | None = None):
        super().__init__(message)
        self.details = details or {}

    def to_dict(self) -> dict:
        payload = super().to_dict()
        payload["quality"] = self.details
        return payload


class ModelUnavailableError(FacetError):
    http_status = 503
    default_code = "model_unavailable"


class RateLimitError(FacetError):
    http_status = 429
    default_code = "rate_limited"

    def __init__(self, message: str, retry_after: int = 60):
        super().__init__(message)
        self.retry_after = retry_after


class AuthError(FacetError):
    http_status = 401
    default_code = "unauthorised"


class SessionError(FacetError):
    http_status = 404
    default_code = "session_expired"
