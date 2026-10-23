"""Main FastAPI application factory and ASGI entrypoint for EconoCausal API.

Configures lifespan startup/shutdown hooks, middleware (CORS, Request Timing),
centralized exception handlers, and API routing.
"""

from contextlib import asynccontextmanager
import os
import time
from typing import AsyncGenerator, Optional
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from src.api.routes import router
from src.api.service import CausalPricingService


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
    """Application lifespan manager initializing models on startup."""
    model_path = getattr(app.state, "model_path", None) or os.getenv("ECONOCAUSAL_MODEL_PATH")
    service = CausalPricingService.get_instance(model_path=model_path)
    app.state.service = service
    yield
    # Cleanup on shutdown if needed
    pass


def create_app(model_path: Optional[str] = None) -> FastAPI:
    """Create and configure the FastAPI application instance.

    Parameters
    ----------
    model_path : Optional[str], default=None
        Custom path to serialized joblib pipeline model.

    Returns
    -------
    FastAPI
        Configured FastAPI app ready for deployment.
    """
    app = FastAPI(
        title="EconoCausal Dynamic Pricing API",
        description=(
            "High-performance REST API microservice for causal dynamic pricing using "
            "Double Machine Learning (DML), Conditional Average Treatment Effects (CATE), "
            "and microeconomic Amoroso-Robinson profit optimization with business guardrails."
        ),
        version="1.0.0",
        docs_url="/docs",
        redoc_url="/redoc",
        openapi_url="/openapi.json",
        lifespan=lifespan,
    )

    app.state.model_path = model_path

    # Configure CORS middleware
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # Middleware: Measure and attach request latency header
    @app.middleware("http")
    async def add_process_time_header(request: Request, call_next):
        start_time = time.perf_counter()
        response = await call_next(request)
        process_time_ms = (time.perf_counter() - start_time) * 1000.0
        response.headers["X-Process-Time-Ms"] = f"{process_time_ms:.2f}"
        return response

    # Global Exception Handlers
    @app.exception_handler(ValueError)
    async def value_error_handler(request: Request, exc: ValueError):
        return JSONResponse(
            status_code=400,
            content={
                "error": "Bad Request",
                "detail": str(exc),
            },
        )

    # Mount API routers
    app.include_router(router)

    return app


# Default ASGI application instance for uvicorn
app = create_app()
