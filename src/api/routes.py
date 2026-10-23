"""FastAPI route definitions for EconoCausal Dynamic Pricing Microservice.

Provides RESTful endpoints for real-time inference, batch optimization,
counterfactual simulation, elasticity calculation, and health checks.
"""

from typing import Any, Dict
from fastapi import APIRouter, Depends, HTTPException, status

from src.api.schemas import (
    BatchPricingRequest,
    BatchPricingResponse,
    ElasticityRequest,
    ElasticityResponse,
    ErrorDetail,
    HealthResponse,
    ModelMetadataResponse,
    PortfolioScenarioRequest,
    PortfolioScenarioResponse,
    SingleItemPricingRequest,
    SingleItemPricingResponse,
)
from src.api.service import CausalPricingService


router = APIRouter()


def get_service() -> CausalPricingService:
    """Dependency provider for CausalPricingService singleton."""
    return CausalPricingService.get_instance()


@router.get(
    "/",
    tags=["Root"],
    summary="API Root Information",
    description="Returns microservice identification, active version, and documentation links.",
)
async def api_root() -> Dict[str, Any]:
    """Root endpoint welcoming API consumers and linking to Swagger UI."""
    return {
        "message": "Welcome to EconoCausal Dynamic Pricing REST API Microservice",
        "version": "1.0.0",
        "methodology": "Double Machine Learning (DML) & Conditional Average Treatment Effects (CATE)",
        "docs_url": "/docs",
        "redoc_url": "/redoc",
        "openapi_url": "/openapi.json",
        "health_check": "/health",
    }


@router.get(
    "/health",
    response_model=HealthResponse,
    tags=["Diagnostics"],
    summary="Microservice Health & Readiness Probe",
    description="Reports service liveness, uptime, model loading state, and baseline causal metrics.",
)
async def health_check(
    service: CausalPricingService = Depends(get_service),
) -> HealthResponse:
    """Check microservice health status."""
    return service.get_health()


@router.get(
    "/model/metadata",
    response_model=ModelMetadataResponse,
    responses={503: {"model": ErrorDetail, "description": "Model not loaded"}},
    tags=["Diagnostics"],
    summary="Causal Pipeline Model Metadata",
    description="Inspect trained Double Machine Learning parameters, causal refutation pass rates, and guardrails.",
)
async def get_model_metadata(
    service: CausalPricingService = Depends(get_service),
) -> ModelMetadataResponse:
    """Retrieve detailed model training parameters, ATE elasticity, and refutations."""
    try:
        return service.get_metadata()
    except ValueError as exc:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=str(exc),
        ) from exc


@router.post(
    "/predict/price",
    response_model=SingleItemPricingResponse,
    responses={
        400: {"model": ErrorDetail, "description": "Invalid input parameters"},
        503: {"model": ErrorDetail, "description": "Model not ready"},
    },
    tags=["Inference"],
    summary="Real-Time Single SKU Dynamic Pricing Recommendation",
    description=(
        "Calculates the microeconomically constrained optimal selling price using "
        "heterogeneous CATE elasticity, Lerner markup index (Amoroso-Robinson), and business guardrails."
    ),
)
async def predict_single_sku_price(
    request: SingleItemPricingRequest,
    service: CausalPricingService = Depends(get_service),
) -> SingleItemPricingResponse:
    """Predict optimal price and projected uplift for a single item."""
    try:
        return service.predict_single_item(request)
    except ValueError as exc:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(exc),
        ) from exc
    except Exception as exc:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Inference pipeline execution error: {str(exc)}",
        ) from exc


@router.post(
    "/predict/elasticity",
    response_model=ElasticityResponse,
    responses={
        400: {"model": ErrorDetail, "description": "Invalid input features"},
        503: {"model": ErrorDetail, "description": "Model not ready"},
    },
    tags=["Inference"],
    summary="Heterogeneous Causal Elasticity Estimation theta(X)",
    description="Estimates item-specific price elasticity using the second-stage CATE interaction model.",
)
async def predict_elasticity(
    request: ElasticityRequest,
    service: CausalPricingService = Depends(get_service),
) -> ElasticityResponse:
    """Predict causal elasticity coefficients for one or more items."""
    try:
        return service.predict_elasticity(request)
    except ValueError as exc:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(exc),
        ) from exc
    except Exception as exc:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Elasticity prediction error: {str(exc)}",
        ) from exc


@router.post(
    "/batch/recommend",
    response_model=BatchPricingResponse,
    responses={
        400: {"model": ErrorDetail, "description": "Invalid batch request"},
        503: {"model": ErrorDetail, "description": "Model not ready"},
    },
    tags=["Optimization"],
    summary="Batch Catalog Dynamic Pricing Optimization",
    description="Optimizes selling prices and simulates portfolio revenue and profit uplifts across a catalog batch.",
)
async def batch_recommend_prices(
    request: BatchPricingRequest,
    service: CausalPricingService = Depends(get_service),
) -> BatchPricingResponse:
    """Generate dynamic price recommendations and portfolio uplifts across a catalog batch."""
    try:
        return service.batch_recommend(request)
    except ValueError as exc:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(exc),
        ) from exc
    except Exception as exc:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Batch pricing optimization error: {str(exc)}",
        ) from exc


@router.post(
    "/simulate/portfolio",
    response_model=PortfolioScenarioResponse,
    responses={
        400: {"model": ErrorDetail, "description": "Invalid scenario simulation request"},
        503: {"model": ErrorDetail, "description": "Model not ready"},
    },
    tags=["Simulation"],
    summary="Counterfactual Scenario Simulation Matrix",
    description=(
        "Simulates counterfactual financial performance across multiple strategic policies: "
        "Status Quo, Flat 10% Discount, Flat 10% Markup, Competitor Benchmark, Revenue Maximizer, Profit Maximizer."
    ),
)
async def simulate_portfolio_scenarios(
    request: PortfolioScenarioRequest,
    service: CausalPricingService = Depends(get_service),
) -> PortfolioScenarioResponse:
    """Simulate and compare portfolio performance across competitive and causal strategies."""
    try:
        return service.simulate_portfolio_scenarios(request)
    except ValueError as exc:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(exc),
        ) from exc
    except Exception as exc:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Scenario simulation error: {str(exc)}",
        ) from exc
