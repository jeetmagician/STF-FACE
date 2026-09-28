"""Security middleware: authentication, rate limiting, response headers.

The rate limiter is an in-process sliding window. That is the right choice for
a single-instance prototype and the wrong choice behind a load balancer - see
docs/DEPLOYMENT.md for the Redis-backed swap.
"""

from __future__ import annotations

import time
from collections import defaultdict, deque
from threading import Lock

from fastapi import Request
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.responses import JSONResponse, Response

from app.config import Settings


class SlidingWindowLimiter:
    def __init__(self) -> None:
        self._hits: dict[str, deque[float]] = defaultdict(deque)
        self._lock = Lock()

    def check(self, key: str, limit: int, window_seconds: int) -> tuple[bool, int]:
        """Return (allowed, retry_after_seconds)."""
        now = time.monotonic()
        cutoff = now - window_seconds
        with self._lock:
            bucket = self._hits[key]
            while bucket and bucket[0] < cutoff:
                bucket.popleft()
            if len(bucket) >= limit:
                retry_after = int(max(1, window_seconds - (now - bucket[0])))
                return False, retry_after
            bucket.append(now)

            # Opportunistic cleanup so idle keys do not accumulate forever.
            if len(self._hits) > 4096:
                stale = [k for k, v in self._hits.items() if not v or v[-1] < cutoff]
                for k in stale:
                    self._hits.pop(k, None)

            return True, 0


limiter = SlidingWindowLimiter()


def client_key(request: Request) -> str:
    """Identify the caller for rate-limiting.

    X-Forwarded-For is only trusted when the app is knowingly behind a proxy;
    otherwise a caller could spoof it to evade limits.
    """
    forwarded = request.headers.get("x-forwarded-for")
    if forwarded and getattr(request.app.state, "trust_proxy_headers", False):
        return forwarded.split(",")[0].strip()
    return request.client.host if request.client else "unknown"


class SecurityHeadersMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request: Request, call_next):
        response: Response = await call_next(request)
        response.headers["X-Content-Type-Options"] = "nosniff"
        response.headers["X-Frame-Options"] = "DENY"
        response.headers["Referrer-Policy"] = "no-referrer"
        response.headers["Cross-Origin-Resource-Policy"] = "same-origin"
        response.headers["Permissions-Policy"] = "camera=(), microphone=(), geolocation=()"
        # Uploaded imagery must never be cached by an intermediary.
        response.headers["Cache-Control"] = "no-store, no-cache, must-revalidate"
        response.headers["Pragma"] = "no-cache"
        return response


class RateLimitMiddleware(BaseHTTPMiddleware):
    def __init__(self, app, settings: Settings) -> None:
        super().__init__(app)
        self.settings = settings

    async def dispatch(self, request: Request, call_next):
        path = request.url.path
        if path.startswith("/api/health") or request.method == "OPTIONS":
            return await call_next(request)

        if path.startswith("/api/analyze") or path in (
            "/api/database-search",
            "/api/database-search/device-capture",
        ):
            limit = self.settings.rate_limit_analyze_requests
            window = self.settings.rate_limit_analyze_window_seconds
            scope = "analyze"
        else:
            limit = self.settings.rate_limit_requests
            window = self.settings.rate_limit_window_seconds
            scope = "general"

        allowed, retry_after = limiter.check(
            f"{scope}:{client_key(request)}", limit, window
        )
        if not allowed:
            return JSONResponse(
                status_code=429,
                content={
                    "error": "rate_limited",
                    "detail": (
                        "Too many requests. Wait a moment before trying again."
                    ),
                },
                headers={"Retry-After": str(retry_after)},
            )

        return await call_next(request)


class ApiKeyMiddleware(BaseHTTPMiddleware):
    def __init__(self, app, settings: Settings) -> None:
        super().__init__(app)
        self.settings = settings

    async def dispatch(self, request: Request, call_next):
        if self.settings.api_key is None:
            return await call_next(request)

        path = request.url.path
        if (
            path.startswith("/api/health")
            or request.method == "OPTIONS"
            or not path.startswith("/api/")
        ):
            return await call_next(request)

        presented = request.headers.get("x-api-key")
        if not presented or not _constant_time_equals(presented, self.settings.api_key):
            return JSONResponse(
                status_code=401,
                content={"error": "unauthorised", "detail": "A valid X-API-Key header is required."},
            )
        return await call_next(request)


def _constant_time_equals(a: str, b: str) -> bool:
    """Compare without leaking length-prefix information through timing."""
    import hmac

    return hmac.compare_digest(a.encode("utf-8"), b.encode("utf-8"))
