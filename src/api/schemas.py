"""Pydantic V2 schemas for EconoCausal Dynamic Pricing REST API.

Provides strongly typed, high-performance request/response data contracts,
input validations, and OpenAPI documentation schema specifications.
"""

from typing import Any, Dict, List, Literal, Optional
from pydantic import BaseModel, ConfigDict, Field, field_validator


class PricingItemInput(BaseModel):
    """Input payload representing a single product SKU for pricing evaluation."""

    model_config = ConfigDict(extra="allow", populate_by_name=True)

    unit_price: float = Field(
        ...,
        gt=0.0,
        description="Current selling unit price of the product (must be > 0).",
        examples=[59.90],
    )
    product_id: Optional[str] = Field(
        default=None,
        description="Unique SKU or product identifier.",
        examples=["bed_bath_table_item_01"],
    )
    product_category_name: Optional[str] = Field(
        default="health_beauty",
        description="Product category name used for CATE heterogeneity moderation.",
        examples=["health_beauty"],
    )
    qty: Optional[float] = Field(
        default=10.0,
        gt=0.0,
        description="Baseline observed sales volume / demand quantity.",
        examples=[15.0],
    )
    unit_cost: Optional[float] = Field(
        default=None,
        gt=0.0,
        description="Direct marginal unit cost. If omitted, inferred from default margin (30%).",
        examples=[41.93],
    )
    comp_price: Optional[float] = Field(
        default=None,
        gt=0.0,
        description="Key competitor benchmark price.",
        examples=[58.50],
    )
    comp_1: Optional[float] = Field(
        default=None,
        gt=0.0,
        description="Primary competitor price (comp_1).",
        examples=[58.50],
    )
    comp_2: Optional[float] = Field(
        default=None,
        gt=0.0,
        description="Secondary competitor price (comp_2).",
        examples=[62.00],
    )
    comp_3: Optional[float] = Field(
        default=None,
        gt=0.0,
        description="Tertiary competitor price (comp_3).",
        examples=[65.00],
    )
    freight_price: Optional[float] = Field(
        default=None,
        ge=0.0,
        description="Shipping / freight price.",
        examples=[12.50],
    )
    product_score: Optional[float] = Field(
        default=None,
        ge=0.0,
        le=5.0,
        description="Product review rating score (0 to 5).",
        examples=[4.2],
    )
    customers: Optional[float] = Field(
        default=None,
        ge=0.0,
        description="Number of potential customers or store traffic.",
        examples=[120.0],
    )
    weekday: Optional[float] = Field(
        default=None,
        ge=0.0,
        description="Weekday index or count.",
        examples=[22.0],
    )
    holiday: Optional[float] = Field(
        default=None,
        ge=0.0,
        description="Holiday count or indicator.",
        examples=[1.0],
    )
    s: Optional[float] = Field(
        default=None,
        description="Seasonal demand index or factor.",
        examples=[15.0],
    )
    lag_price: Optional[float] = Field(
        default=None,
        gt=0.0,
        description="Previous period lagged price.",
        examples=[59.90],
    )

    def to_inference_dict(self) -> Dict[str, Any]:
        """Convert validated Pydantic model to a clean dictionary for the pipeline."""
        d = self.model_dump(exclude_none=True)
        # Ensure comp_price maps to comp_1 if comp_1 not explicitly provided
        if "comp_price" in d and "comp_1" not in d:
            d["comp_1"] = d["comp_price"]
        return d


class SingleItemPricingRequest(BaseModel):
    """Request payload for real-time single-SKU dynamic pricing recommendation."""

    item: PricingItemInput = Field(
        ...,
        description="Product details and confounder features for pricing optimization.",
    )
    objective: Literal["profit", "revenue"] = Field(
        default="profit",
        description="Optimization objective: 'profit' (Amoroso-Robinson) or 'revenue'.",
        examples=["profit"],
    )


class SingleItemPricingResponse(BaseModel):
    """Response payload containing optimal price, uplift simulation, and diagnostics."""

    status: str = Field(default="SUCCESS", description="Execution status.")
    product_id: Optional[str] = Field(default=None, description="Product identifier if provided.")
    action: str = Field(
        ...,
        description="Recommended pricing action: 'MARKUP', 'DISCOUNT', or 'HOLD'.",
        examples=["MARKUP"],
    )
    baseline_price: float = Field(..., description="Original baseline price ($).")
    optimal_price: float = Field(..., description="Recommended optimal price ($).")
    price_change_pct: float = Field(..., description="Percentage price adjustment (%).")
    causal_elasticity: float = Field(
        ...,
        description="Conditional Average Treatment Effect (CATE) price elasticity theta(X).",
    )
    unit_cost: float = Field(..., description="Marginal unit cost ($).")
    baseline_margin_pct: float = Field(..., description="Baseline gross margin percentage (%).")
    projected_margin_pct: float = Field(..., description="Projected gross margin percentage (%).")
    baseline_qty: float = Field(..., description="Baseline demand volume.")
    projected_qty: float = Field(..., description="Simulated demand volume under optimal price.")
    demand_uplift_pct: float = Field(..., description="Projected demand uplift (%).")
    baseline_revenue: float = Field(..., description="Baseline total revenue ($).")
    projected_revenue: float = Field(..., description="Projected revenue ($).")
    revenue_uplift_pct: float = Field(..., description="Projected revenue uplift (%).")
    baseline_profit: float = Field(..., description="Baseline gross profit ($).")
    projected_profit: float = Field(..., description="Projected gross profit ($).")
    profit_uplift_pct: float = Field(..., description="Projected profit uplift (%).")
    bound_hit: Optional[str] = Field(
        default=None,
        description="Active guardrail: 'min_bound_clipped', 'max_bound_clipped', 'margin_floor_hit', or None.",
    )
    objective: str = Field(default="profit", description="Optimization objective used.")
    inference_latency_ms: Optional[float] = Field(
        default=None,
        description="Inference execution duration in milliseconds.",
    )


class BatchPricingRequest(BaseModel):
    """Request payload for batch catalog pricing optimization."""

    items: List[PricingItemInput] = Field(
        ...,
        min_length=1,
        description="List of product SKU records to evaluate and optimize.",
    )
    objective: Literal["profit", "revenue"] = Field(
        default="profit",
        description="Global optimization objective for the catalog.",
    )


class BatchPricingResponse(BaseModel):
    """Aggregate response for batch catalog dynamic pricing."""

    total_items: int = Field(..., description="Number of items evaluated.")
    objective: str = Field(..., description="Optimization objective applied.")
    recommendations: List[SingleItemPricingResponse] = Field(
        ...,
        description="List of individual SKU price recommendations.",
    )
    portfolio_summary: Dict[str, Any] = Field(
        ...,
        description="High-level catalog portfolio baseline vs. projected aggregate uplift.",
    )
    batch_latency_ms: float = Field(..., description="Total batch processing latency in milliseconds.")


class ElasticityRequest(BaseModel):
    """Request payload for raw heterogeneous causal price elasticity estimation."""

    items: List[PricingItemInput] = Field(
        ...,
        min_length=1,
        description="List of item confounder feature dictionaries.",
    )


class ElasticityResponseItem(BaseModel):
    """Individual elasticity estimation result."""

    product_id: Optional[str] = Field(default=None)
    product_category_name: Optional[str] = Field(default=None)
    predicted_elasticity: float = Field(
        ...,
        description="Predicted causal elasticity theta(X). Negative values denote elastic demand.",
    )
    elasticity_classification: str = Field(
        ...,
        description="'Highly Elastic' (|e| > 1.5), 'Elastic' (1.0 < |e| <= 1.5), or 'Inelastic' (|e| <= 1.0).",
    )


class ElasticityResponse(BaseModel):
    """Response containing causal elasticity predictions."""

    total_items: int = Field(...)
    predictions: List[ElasticityResponseItem] = Field(...)
    latency_ms: float = Field(...)


class PortfolioScenarioRequest(BaseModel):
    """Request payload to simulate portfolio scenarios across competitive strategies."""

    items: List[PricingItemInput] = Field(
        ...,
        min_length=1,
        description="Catalog items to simulate across pricing strategies.",
    )
    comp_col: Optional[str] = Field(
        default="comp_1",
        description="Competitor column used for matching benchmark.",
    )


class ScenarioResult(BaseModel):
    """Simulation metrics for a single pricing strategy."""

    strategy: str = Field(..., description="Strategy name.")
    total_revenue: float = Field(..., description="Projected total revenue ($).")
    revenue_uplift_pct: float = Field(..., description="Revenue change vs baseline (%).")
    total_profit: float = Field(..., description="Projected total gross profit ($).")
    profit_uplift_pct: float = Field(..., description="Profit change vs baseline (%).")
    total_units_sold: float = Field(..., description="Total demand units projected.")
    units_uplift_pct: float = Field(..., description="Demand volume change vs baseline (%).")
    average_margin_pct: float = Field(..., description="Average portfolio gross margin (%).")


class PortfolioScenarioResponse(BaseModel):
    """Comprehensive multi-strategy counterfactual simulation response."""

    total_items: int = Field(..., description="Number of items simulated.")
    scenarios: List[ScenarioResult] = Field(
        ...,
        description="Comparative scenario matrix across standard and causal pricing policies.",
    )
    recommended_strategy: str = Field(
        ...,
        description="Strategy delivering highest total gross profit within margin guardrails.",
    )
    simulation_latency_ms: float = Field(..., description="Duration of scenario simulation in milliseconds.")


class HealthResponse(BaseModel):
    """Microservice liveness and readiness probe response."""

    status: str = Field(default="healthy", description="Service health state: 'healthy' or 'degraded'.")
    service: str = Field(default="econocausal-pricing-service")
    version: str = Field(default="1.0.0")
    uptime_seconds: float = Field(..., description="Service uptime in seconds.")
    model_loaded: bool = Field(..., description="Whether the causal pipeline is in memory and fitted.")
    model_path: Optional[str] = Field(default=None, description="Filepath of loaded model.")
    baseline_ate: Optional[float] = Field(
        default=None,
        description="Double Machine Learning Average Treatment Effect (ATE) price elasticity.",
    )
    refutation_pass_rate: Optional[float] = Field(
        default=None,
        description="Causal refutation test battery pass rate (0% to 100%).",
    )
    timestamp: str = Field(..., description="ISO 8601 UTC timestamp.")


class ModelMetadataResponse(BaseModel):
    """Detailed metadata and diagnostic parameters of the active causal model."""

    is_fitted: bool
    model_path: Optional[str]
    trained_at: Optional[str]
    training_samples_total: Optional[int]
    training_samples_train: Optional[int]
    encoded_feature_count: Optional[int]
    baseline_ate: Optional[float]
    baseline_ate_se: Optional[float]
    baseline_ate_pvalue: Optional[float]
    refutation_pass_rate: Optional[float]
    cate_mean_elasticity: Optional[float]
    cate_min_elasticity: Optional[float]
    cate_max_elasticity: Optional[float]
    optimizer_bounds: Dict[str, Any]
    feature_imputation_medians: Dict[str, Any]


class ErrorDetail(BaseModel):
    """Structured error payload for failed requests."""

    error: str = Field(..., description="Error classification.")
    detail: str = Field(..., description="Human-readable explanation of error.")
    timestamp: str = Field(..., description="ISO 8601 UTC timestamp.")


class FeatureDriftDetail(BaseModel):
    """Drift diagnostics for a single confounder or feature."""

    feature_name: str
    psi: float = Field(..., description="Population Stability Index (PSI).")
    drift_level: str = Field(..., description="'STABLE', 'MODERATE_DRIFT', or 'SIGNIFICANT_DRIFT'.")
    drift_detected: bool = Field(..., description="True if PSI >= warning threshold.")
    is_categorical: bool = Field(default=False)
    ks_stat: Optional[float] = Field(default=None, description="Kolmogorov-Smirnov two-sample test statistic.")
    p_value: Optional[float] = Field(default=None, description="KS test p-value.")
    wasserstein_distance: Optional[float] = Field(default=None, description="Earth Mover's Distance.")
    baseline_mean: Optional[float] = Field(default=None)
    target_mean: Optional[float] = Field(default=None)
    mean_shift_pct: Optional[float] = Field(default=None, description="Percentage shift in mean.")


class ElasticityDriftDetail(BaseModel):
    """Drift diagnostics for heterogeneous CATE price elasticity distribution."""

    psi: float
    drift_level: str
    baseline_mean: float
    target_mean: float
    delta_mean: float
    ks_stat: float
    p_value: float
    num_regime_changes: int
    regime_changes: List[Dict[str, Any]] = Field(default_factory=list)


class RefutationAuditDetail(BaseModel):
    """Diagnostics for continuous production refutation re-auditing."""

    pass_rate: float
    tests_passed: int
    total_tests: int
    placebo_treatment: Dict[str, Any]
    random_common_cause: Dict[str, Any]
    data_subset: Dict[str, Any]


class DriftAuditRequest(BaseModel):
    """Request payload to perform continuous confounder and elasticity drift detection."""

    items: List[PricingItemInput] = Field(
        ...,
        min_length=1,
        description="Batch of incoming production items to audit against baseline training distribution.",
    )
    run_refutations: bool = Field(
        default=False,
        description="Whether to run active causal refutation tests on production items (computationally intensive).",
    )
    psi_warning_threshold: float = Field(
        default=0.10,
        ge=0.01,
        le=0.50,
        description="Threshold for moderate drift warning (default 0.10).",
    )
    psi_alert_threshold: float = Field(
        default=0.25,
        ge=0.05,
        le=1.00,
        description="Threshold for significant drift alert triggering retraining (default 0.25).",
    )


class DriftAuditResponse(BaseModel):
    """Comprehensive causal MLOps drift audit report response."""

    timestamp: str
    dataset_baseline_rows: int
    dataset_target_rows: int
    overall_status: str = Field(..., description="'HEALTHY', 'WARNING', 'RETRAIN_RECOMMENDED', or 'CRITICAL'.")
    recommendation: str = Field(..., description="Actionable retraining or operational guidance.")
    max_confounder_psi: float = Field(..., description="Maximum PSI observed among confounders.")
    mean_confounder_psi: float = Field(..., description="Mean PSI across all audited confounders.")
    num_drifting_confounders: int = Field(..., description="Count of confounders exhibiting drift.")
    confounders: Dict[str, FeatureDriftDetail] = Field(..., description="Per-confounder drift breakdown.")
    treatment_drift: Optional[FeatureDriftDetail] = None
    outcome_drift: Optional[FeatureDriftDetail] = None
    elasticity_drift: Optional[ElasticityDriftDetail] = None
    refutation_audit: Optional[RefutationAuditDetail] = None
    audit_latency_ms: float = Field(..., description="Audit computation time in milliseconds.")

