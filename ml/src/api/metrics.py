"""Prometheus observability and telemetry metrics for EconoCausal API microservice.

Defines Prometheus collectors for HTTP traffic, latency histograms,
dynamic pricing recommendations, CATE price elasticity distributions,
and microservice runtime health.
"""

import time
from typing import Callable, Optional
from fastapi import Request, Response
from prometheus_client import (
    CONTENT_TYPE_LATEST,
    Counter,
    Gauge,
    Histogram,
    Info,
    generate_latest,
)

# ---------------------------------------------------------------------------
# Prometheus Metric Collectors
# ---------------------------------------------------------------------------

# HTTP Request throughput counter partitioned by method, endpoint, and HTTP status code
REQUEST_COUNT = Counter(
    name="econocausal_http_requests_total",
    documentation="Total count of HTTP requests processed by EconoCausal microservice",
    labelnames=["method", "endpoint", "status_code"],
)

# Request latency histogram with high-resolution latency buckets (5ms to 5000ms)
REQUEST_LATENCY = Histogram(
    name="econocausal_http_request_duration_seconds",
    documentation="HTTP request handling latency in seconds",
    labelnames=["method", "endpoint"],
    buckets=(0.005, 0.010, 0.025, 0.050, 0.100, 0.250, 0.500, 1.000, 2.500, 5.000),
)

# Active concurrent requests currently executing in the ASGI worker
ACTIVE_REQUESTS = Gauge(
    name="econocausal_http_requests_active",
    documentation="Number of HTTP requests currently being processed in-flight",
)

# Dynamic pricing recommendation counters by product category and strategic action
PRICING_RECOMMENDATIONS = Counter(
    name="econocausal_pricing_recommendations_total",
    documentation="Count of causal dynamic pricing recommendations generated",
    labelnames=["product_category", "action"],
)

# Elasticity gauge tracking the latest estimated CATE price elasticity per category
CATE_ELASTICITY = Gauge(
    name="econocausal_cate_elasticity_latest",
    documentation="Latest estimated price elasticity of demand (theta) by category",
    labelnames=["product_category"],
)

# Expected profit uplift percentage gauge from latest causal pricing optimization
PROFIT_UPLIFT = Gauge(
    name="econocausal_profit_uplift_percent_latest",
    documentation="Latest expected percentage profit uplift by product category",
    labelnames=["product_category"],
)

# Model build metadata and causal parameters
MODEL_INFO = Info(
    name="econocausal_model",
    documentation="Causal ML model architecture, baseline ATE, and refutation score",
)

# Confounder distribution drift gauges
CONFOUNDER_MAX_PSI = Gauge(
    name="econocausal_confounder_max_psi",
    documentation="Maximum Population Stability Index (PSI) observed across confounders",
)

CONFOUNDER_MEAN_PSI = Gauge(
    name="econocausal_confounder_mean_psi",
    documentation="Mean Population Stability Index (PSI) across audited confounders",
)

DRIFT_HEALTH_STATUS = Gauge(
    name="econocausal_drift_health_status",
    documentation="Causal MLOps drift status code: 0=HEALTHY, 1=WARNING, 2=RETRAIN_RECOMMENDED, 3=CRITICAL",
)

PROD_REFUTATION_PASS_RATE = Gauge(
    name="econocausal_refutation_prod_pass_rate",
    documentation="Causal refutation test pass rate on production data (0.0 to 1.0)",
)


def record_drift_audit(
    max_psi: float,
    mean_psi: float,
    status_str: str,
    refutation_pass_rate: Optional[float] = None,
) -> None:
    """Record Prometheus telemetry for a causal MLOps drift audit."""
    CONFOUNDER_MAX_PSI.set(max_psi)
    CONFOUNDER_MEAN_PSI.set(mean_psi)

    status_map = {
        "HEALTHY": 0,
        "WARNING": 1,
        "RETRAIN_RECOMMENDED": 2,
        "CRITICAL": 3,
    }
    DRIFT_HEALTH_STATUS.set(status_map.get(status_str.upper(), 0))

    if refutation_pass_rate is not None:
        PROD_REFUTATION_PASS_RATE.set(refutation_pass_rate)



def init_model_info(
    version: str = "1.0.0",
    baseline_ate: float = -1.2678,
    refutation_pass_rate: float = 1.0,
    status: str = "ready",
) -> None:
    """Initialize Prometheus model metadata info gauge on application startup."""
    MODEL_INFO.info(
        {
            "version": version,
            "baseline_ate": f"{baseline_ate:.4f}",
            "refutation_pass_rate": f"{refutation_pass_rate * 100:.1f}%",
            "framework": "DoubleMachineLearning_PLR",
            "status": status,
        }
    )


def record_recommendation(
    product_category: str,
    action: str,
    elasticity: Optional[float] = None,
    profit_uplift_pct: Optional[float] = None,
) -> None:
    """Record Prometheus telemetry for a generated pricing recommendation.

    Parameters
    ----------
    product_category : str
        Product category (e.g. 'health_beauty', 'computers_accessories').
    action : str
        Pricing action ('MARKUP', 'DISCOUNT', 'HOLD').
    elasticity : Optional[float], default=None
        Estimated CATE elasticity of demand.
    profit_uplift_pct : Optional[float], default=None
        Expected profit improvement percentage.
    """
    PRICING_RECOMMENDATIONS.labels(
        product_category=product_category,
        action=action,
    ).inc()

    if elasticity is not None:
        CATE_ELASTICITY.labels(product_category=product_category).set(elasticity)

    if profit_uplift_pct is not None:
        PROFIT_UPLIFT.labels(product_category=product_category).set(profit_uplift_pct)


def export_metrics() -> Response:
    """Generate Prometheus exposition text format response."""
    return Response(
        content=generate_latest(),
        media_type=CONTENT_TYPE_LATEST,
        headers={"Cache-Control": "no-cache, no-store, must-revalidate"},
    )
