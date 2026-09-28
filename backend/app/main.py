"""FastAPI application entrypoint."""

from __future__ import annotations

import asyncio
import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from app.api.routes import VERSION, router
from app.config import get_settings
from app.core.errors import FacetError
from app.core.security import (
    ApiKeyMiddleware,
    RateLimitMiddleware,
    SecurityHeadersMiddleware,
)
from app.core.store import EphemeralStore

logging.basicConfig(
    level=get_settings().log_level,
    format="%(asctime)s %(levelname)-8s %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)

DESCRIPTION = """
A facial **similarity** assessment service.

This API reports how similar two sets of photographs are, as measured by a
face-recognition model under a stated calibration. It does not determine
identity, and no response should be read as establishing that two photographs
depict the same person.

Uploaded images are processed in memory and never written to disk.
""".strip()


async def _sweep_sessions(app: FastAPI) -> None:
    """Purge expired retained images even if no request arrives to trigger it."""
    settings = get_settings()
    interval = max(30, settings.session_ttl_seconds // 2)
    while True:
        try:
            await asyncio.sleep(interval)
            removed = app.state.store.purge_expired()
            if removed:
                logger.info("Purged %d expired image session(s).", removed)
        except asyncio.CancelledError:
            raise
        except Exception as exc:  # pragma: no cover - defensive
            logger.warning("Session sweep failed: %s", exc)


@asynccontextmanager
async def lifespan(app: FastAPI):
    settings = get_settings()
    app.state.store = EphemeralStore(settings)
    app.state.trust_proxy_headers = settings.is_production

    if settings.is_production and settings.api_key is None:
        logger.warning(
            "Running in production with no API_KEY set. The API is open to "
            "anyone who can reach it."
        )

    # Load the model eagerly so the first user request is not slowed by it, but
    # do not abort startup if it fails - /api/health should stay reachable to
    # report the problem.
    try:
        from app.models.registry import get_embedder

        embedder = get_embedder(settings)
        embedder.warmup()
        logger.info(
            "Model ready: %s / %s (%s)",
            embedder.info.detector,
            embedder.info.recognizer,
            embedder.info.license,
        )
    except Exception as exc:
        logger.error("Model failed to load at startup: %s", exc)

    sweeper = asyncio.create_task(_sweep_sessions(app))
    try:
        yield
    finally:
        sweeper.cancel()
        try:
            await sweeper
        except asyncio.CancelledError:
            pass
        app.state.store.clear()
        logger.info("Cleared all retained images on shutdown.")


def create_app() -> FastAPI:
    settings = get_settings()

    app = FastAPI(
        title=f"{settings.app_name} API",
        description=DESCRIPTION,
        version=VERSION,
        lifespan=lifespan,
        docs_url="/docs" if not settings.is_production else None,
        redoc_url=None,
    )

    app.add_middleware(SecurityHeadersMiddleware)
    app.add_middleware(ApiKeyMiddleware, settings=settings)
    app.add_middleware(RateLimitMiddleware, settings=settings)
    app.add_middleware(
        CORSMiddleware,
        allow_origins=settings.cors_origins,
        allow_credentials=False,
        allow_methods=["GET", "POST", "DELETE", "OPTIONS"],
        allow_headers=["Content-Type", "X-API-Key"],
        max_age=600,
    )

    @app.exception_handler(FacetError)
    async def handle_facet_error(request: Request, exc: FacetError) -> JSONResponse:
        headers = {}
        if hasattr(exc, "retry_after"):
            headers["Retry-After"] = str(exc.retry_after)
        return JSONResponse(
            status_code=exc.http_status, content=exc.to_dict(), headers=headers
        )

    @app.exception_handler(Exception)
    async def handle_unexpected(request: Request, exc: Exception) -> JSONResponse:
        # Log the detail, return a generic message: internal exception text can
        # leak paths and library versions.
        logger.exception("Unhandled error on %s", request.url.path)
        return JSONResponse(
            status_code=500,
            content={
                "error": "internal_error",
                "detail": "An unexpected error occurred while processing the request.",
            },
        )

    app.include_router(router)

    @app.get("/", include_in_schema=False)
    async def root() -> dict:
        return {
            "service": settings.app_name,
            "version": VERSION,
            "docs": "/docs" if not settings.is_production else None,
            "notice": (
                "This service reports facial similarity under a model. It does "
                "not determine identity."
            ),
        }

    return app


app = create_app()
