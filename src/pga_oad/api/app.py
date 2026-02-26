"""FastAPI application factory."""
from __future__ import annotations

from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from ..db.engine import init_db
from .routes import users_router, picks_router, analytics_router, markets_router


def create_app() -> FastAPI:
    """Create and configure the FastAPI application."""

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        init_db()
        yield

    app = FastAPI(
        title="PGA One-and-Done Pro",
        description=(
            "AI-powered PGA Tour One-and-Done pick optimizer. "
            "Blends DataGolf skill models, Kalshi prediction markets, "
            "DraftKings odds, recent form, and course history into a "
            "proprietary multi-signal ensemble."
        ),
        version="1.0.0",
        docs_url="/docs",
        redoc_url="/redoc",
        lifespan=lifespan,
    )

    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],  # Lock down in production
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # Register route modules
    app.include_router(users_router, prefix="/api/v1")
    app.include_router(picks_router, prefix="/api/v1")
    app.include_router(analytics_router, prefix="/api/v1")
    app.include_router(markets_router, prefix="/api/v1")

    @app.get("/health")
    def health():
        return {"status": "ok", "service": "pga-oad-pro"}

    return app


# Entry point for `uvicorn src.pga_oad.api.app:app`
app = create_app()
