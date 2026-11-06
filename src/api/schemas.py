"""Pydantic V2 schemas for EconoCausal Dynamic Pricing REST API.

Provides strongly typed, high-performance request/response data contracts,
input validations, and OpenAPI documentation schema specifications.
"""

from typing import Any, Dict, List, Literal, Optional
from pydantic import AliasChoices, BaseModel, ConfigDict, Field, field_validator, model_validator


class ConfidenceIntervalSchema(BaseModel):
    """Asymptotic confidence interval parameters for causal treatment effects."""

    lower: float = Field(..., description="Lower bound of confidence interval.")
    upper: float = Field(..., description="Upper bound of confidence interval.")
    confidence_level: float = Field(default=0.95, description="Confidence level (e.g., 0.95 for 95% CI).")


class PriceCurvePoint(BaseModel):
    """Simulated microeconomic demand, revenue, and gross profit point on candidate price spectrum."""

    price: float = Field(..., description="Candidate selling price ($).")
    predicted_demand: float = Field(..., description="Simulated demand quantity at candidate price.")
    predicted_revenue: float = Field(..., description="Projected total revenue at candidate price ($).")
    predicted_profit: float = Field(..., description="Projected gross profit at candidate price ($).")


class PricingItemInput(BaseModel):
    """Input payload representing a single product SKU for pricing evaluation."""

    model_config = ConfigDict(extra="allow", populate_by_name=True)

    unit_price: float = Field(
        ...,
        gt=0.0,
        validation_alias=AliasChoices("unit_price", "price", "basePrice", "currentPrice", "unitPrice"),
        description="Current selling unit price of the product (must be > 0).",
        examples=[59.90],
    )
    product_id: Optional[str] = Field(
        default=None,
        validation_alias=AliasChoices("product_id", "productId", "sku", "id"),
        description="Unique SKU or product identifier.",
        examples=["bed_bath_table_item_01"],
    )
    product_category_name: Optional[str] = Field(
        default="health_beauty",
        validation_alias=AliasChoices("product_category_name", "category", "productCategory"),
        description="Product category name used for CATE heterogeneity moderation.",
        examples=["health_beauty"],
    )
    qty: Optional[float] = Field(
        default=10.0,
        gt=0.0,
        validation_alias=AliasChoices("qty", "unitsSold", "demand", "salesVolume", "quantity"),
        description="Baseline observed sales volume / demand quantity.",
        examples=[15.0],
    )
    unit_cost: Optional[float] = Field(
        default=None,
        gt=0.0,
        validation_alias=AliasChoices("unit_cost", "cost", "costPrice", "marginalCost", "unitCost"),
        description="Direct marginal unit cost. If omitted, inferred from default margin (30%).",
        examples=[41.93],
    )
    comp_price: Optional[float] = Field(
        default=None,
        gt=0.0,
        validation_alias=AliasChoices("comp_price", "competitorPrice", "compPrice", "comp_1"),
        description="Key competitor benchmark price.",
        examples=[58.50],
    )
    comp_1: Optional[float] = Field(
        default=None,
        gt=0.0,
        validation_alias=AliasChoices("comp_1", "comp1"),
        description="Primary competitor price (comp_1).",
        examples=[58.50],
    )
    comp_2: Optional[float] = Field(
        default=None,
        gt=0.0,
        validation_alias=AliasChoices("comp_2", "comp2"),
        description="Secondary competitor price (comp_2).",
        examples=[62.00],
    )
    comp_3: Optional[float] = Field(
        default=None,
        gt=0.0,
        validation_alias=AliasChoices("comp_3", "comp3"),
        description="Tertiary competitor price (comp_3).",
        examples=[65.00],
    )
    freight_price: Optional[float] = Field(
        default=None,
        ge=0.0,
        validation_alias=AliasChoices("freight_price", "freightPrice", "shippingPrice"),
        description="Shipping / freight price.",
        examples=[12.50],
    )
    product_score: Optional[float] = Field(
        default=None,
        ge=0.0,
        le=5.0,
        validation_alias=AliasChoices("product_score", "productScore", "rating", "reviewScore"),
        description="Product review rating score (0 to 5).",
        examples=[4.2],
    )
    customers: Optional[float] = Field(
        default=None,
        ge=0.0,
        validation_alias=AliasChoices("customers", "traffic", "visitorCount"),
        description="Number of potential customers or store traffic.",
        examples=[120.0],
    )
    weekday: Optional[float] = Field(
        default=None,
        ge=0.0,
        validation_alias=AliasChoices("weekday", "dayOfWeek"),
        description="Weekday index or count.",
        examples=[22.0],
    )
    holiday: Optional[float] = Field(
        default=None,
        ge=0.0,
        validation_alias=AliasChoices("holiday", "isHoliday"),
        description="Holiday count or indicator.",
        examples=[1.0],
    )
    s: Optional[float] = Field(
        default=None,
        validation_alias=AliasChoices("s", "seasonalityIndex", "seasonality"),
        description="Seasonal demand index or factor.",
        examples=[15.0],
    )
    lag_price: Optional[float] = Field(
        default=None,
        gt=0.0,
        validation_alias=AliasChoices("lag_price", "lagPrice", "previousPrice"),
        description="Previous period lagged price.",
        examples=[59.90],
    )
    min_price: Optional[float] = Field(
        default=None,
        gt=0.0,
        validation_alias=AliasChoices("min_price", "minPrice", "priceFloor"),
        description="Minimum permissible price floor constraint.",
        examples=[45.00],
    )
    max_price: Optional[float] = Field(
        default=None,
        gt=0.0,
        validation_alias=AliasChoices("max_price", "maxPrice", "priceCeiling"),
        description="Maximum permissible price ceiling constraint.",
        examples=[90.00],
    )
    inventory: Optional[float] = Field(
        default=None,
        ge=0.0,
        validation_alias=AliasChoices("inventory", "inventoryLevel", "stock"),
        description="Current stock inventory count.",
        examples=[100.0],
    )
    is_promotion_active: Optional[bool] = Field(
        default=False,
        validation_alias=AliasChoices("is_promotion_active", "isPromotionActive", "promotionalFlag"),
        description="Whether active promotion or campaign applies.",
        examples=[False],
    )

    @field_validator("product_score")
    @classmethod
    def validate_product_score(cls, v: Optional[float]) -> Optional[float]:
        """Validate product score is strictly bounded in [0.0, 5.0]."""
        if v is not None and not (0.0 <= v <= 5.0):
            raise ValueError(f"product_score must be between 0.0 and 5.0, got {v}")
        return v

    @model_validator(mode="after")
    def validate_price_bounds(self) -> "PricingItemInput":
        """Validate that min_price does not exceed max_price."""
        if self.min_price is not None and self.max_price is not None:
            if self.min_price > self.max_price:
                raise ValueError(
                    f"min_price ({self.min_price}) cannot exceed max_price ({self.max_price})"
                )
        return self

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
    """Response payload containing optimal price, uplift simulation, uncertainty, and diagnostics."""

    model_config = ConfigDict(populate_by_name=True)

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
    treatment_effect: Optional[float] = Field(
        default=None,
        description="Causal treatment effect point estimate (elasticity dQ/dP equivalent).",
    )
    confidence_interval: Optional[ConfidenceIntervalSchema] = Field(
        default=None,
        description="Asymptotic 95% confidence interval [lower, upper] for causal elasticity.",
    )
    std_error: Optional[float] = Field(
        default=None,
        description="Standard error of treatment effect estimate.",
    )
    p_value: Optional[float] = Field(
        default=None,
        description="P-value testing significance against null hypothesis.",
    )
    is_statistically_significant: Optional[bool] = Field(
        default=True,
        description="Whether treatment effect is statistically distinguishable from zero.",
    )
    elasticity_regime: Optional[str] = Field(
        default="Elastic",
        description="Microeconomic elasticity classification: 'Elastic', 'Inelastic', or 'Unit Elastic'.",
    )
    unit_cost: float = Field(..., description="Marginal unit cost ($).")
    baseline_margin_pct: float = Field(..., description="Baseline gross margin percentage (%).")
    projected_margin_pct: float = Field(..., description="Projected gross margin percentage (%).")
    baseline_qty: float = Field(..., description="Baseline demand volume.")
    projected_qty: float = Field(..., description="Simulated demand volume under optimal price.")
    expected_demand: Optional[float] = Field(
        default=None,
        description="Expected demand units under optimal price (alias for projected_qty).",
    )
    demand_uplift_pct: float = Field(..., description="Projected demand uplift (%).")
    baseline_revenue: float = Field(..., description="Baseline total revenue ($).")
    projected_revenue: float = Field(..., description="Projected revenue ($).")
    expected_revenue: Optional[float] = Field(
        default=None,
        description="Expected total revenue under optimal price (alias for projected_revenue).",
    )
    revenue_uplift_pct: float = Field(..., description="Projected revenue uplift (%).")
    baseline_profit: float = Field(..., description="Baseline gross profit ($).")
    projected_profit: float = Field(..., description="Projected gross profit ($).")
    expected_profit: Optional[float] = Field(
        default=None,
        description="Expected gross profit under optimal price (alias for projected_profit).",
    )
    profit_uplift_pct: float = Field(..., description="Projected profit uplift (%).")
    bound_hit: Optional[str] = Field(
        default=None,
        description="Active guardrail: 'min_bound_clipped', 'max_bound_clipped', 'margin_floor_hit', or None.",
    )
    objective: str = Field(default="profit", description="Optimization objective used.")
    controlled_confounders: Optional[List[str]] = Field(
        default=None,
        description="List of confounders controlled in Double Machine Learning model.",
    )
    price_demand_curve: Optional[List[PriceCurvePoint]] = Field(
        default=None,
        description="Multi-point simulated demand, revenue, and profit curve across price spectrum.",
    )
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
    std_error: Optional[float] = Field(
        default=None,
        description="Asymptotic standard error of elasticity estimate.",
    )
    confidence_interval: Optional[ConfidenceIntervalSchema] = Field(
        default=None,
        description="Asymptotic 95% confidence interval [lower, upper] for elasticity.",
    )
    p_value: Optional[float] = Field(
        default=None,
        description="P-value testing null hypothesis of unit elasticity.",
    )
    is_statistically_significant: Optional[bool] = Field(
        default=None,
        description="Whether elasticity is statistically significant.",
    )
    elasticity_regime: Optional[str] = Field(
        default=None,
        description="Regime classification ('Elastic', 'Inelastic', 'Unit Elastic').",
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
    baseline_ate_ci: Optional[Dict[str, float]] = Field(
        default=None,
        description="95% confidence interval for baseline ATE price elasticity.",
    )
    baseline_ate_se: Optional[float] = Field(
        default=None,
        description="Standard error of baseline ATE price elasticity.",
    )
    baseline_ate_pvalue: Optional[float] = Field(
        default=None,
        description="P-value of baseline ATE price elasticity.",
    )
    refutation_pass_rate: Optional[float] = Field(
        default=None,
        description="Causal refutation test battery pass rate (0% to 100%).",
    )
    active_features: Optional[List[str]] = Field(
        default=None,
        description="List of active confounder features in causal pipeline.",
    )
    timestamp: str = Field(..., description="ISO 8601 UTC timestamp.")


class BackendPricingOptimizationRequest(BaseModel):
    """Direct request payload matching Node.js Express backend controllers."""

    model_config = ConfigDict(extra="allow", populate_by_name=True)

    productId: Optional[str] = Field(
        default=None,
        validation_alias=AliasChoices("productId", "product_id"),
        description="MongoDB ObjectId or SKU.",
    )
    title: Optional[str] = Field(
        default="Causal DML Price Optimization",
        description="Experiment title.",
    )
    basePrice: Optional[float] = Field(
        default=120.0,
        gt=0.0,
        validation_alias=AliasChoices("basePrice", "base_price", "unit_price"),
        description="Base catalog benchmark price.",
    )
    costPrice: Optional[float] = Field(
        default=55.0,
        gt=0.0,
        validation_alias=AliasChoices("costPrice", "cost_price", "unit_cost"),
        description="Cost price (Marginal Cost).",
    )
    currentPrice: Optional[float] = Field(
        default=115.0,
        gt=0.0,
        validation_alias=AliasChoices("currentPrice", "current_price"),
        description="Current live selling price.",
    )
    minPrice: Optional[float] = Field(
        default=None,
        gt=0.0,
        validation_alias=AliasChoices("minPrice", "min_price"),
        description="Lower guardrail price.",
    )
    maxPrice: Optional[float] = Field(
        default=None,
        gt=0.0,
        validation_alias=AliasChoices("maxPrice", "max_price"),
        description="Upper guardrail price.",
    )
    competitorPrice: Optional[float] = Field(
        default=None,
        gt=0.0,
        validation_alias=AliasChoices("competitorPrice", "competitor_price", "comp_price"),
        description="Competitor benchmark price.",
    )
    category: Optional[str] = Field(
        default="health_beauty",
        validation_alias=AliasChoices("category", "product_category_name"),
        description="Product category.",
    )
    historicalData: Optional[List[Dict[str, Any]]] = Field(
        default_factory=list,
        validation_alias=AliasChoices("historicalData", "historical_data"),
        description="Historical sales observation records.",
    )

    @model_validator(mode="after")
    def validate_backend_bounds(self) -> "BackendPricingOptimizationRequest":
        """Validate that minPrice does not exceed maxPrice."""
        if self.minPrice is not None and self.maxPrice is not None:
            if self.minPrice > self.maxPrice:
                raise ValueError(
                    f"minPrice ({self.minPrice}) cannot exceed maxPrice ({self.maxPrice})"
                )
        return self


class BackendExperimentData(BaseModel):
    """Data payload matching Node.js PricingExperiment Mongoose document."""

    model_config = ConfigDict(populate_by_name=True)

    title: str
    productId: str
    status: str = "completed"
    methodology: str = "Double Machine Learning (DML) - Orthogonal Residualization"
    sampleSize: int
    causalTreatmentEffect: float
    priceElasticity: float
    confidenceInterval: Dict[str, float]
    pVal: float
    currentPrice: float
    recommendedOptimalPrice: float
    projectedDemandAtOptimal: float
    projectedRevenueAtOptimal: float
    projectedProfitLiftPercentage: float
    controlledConfounders: List[str]
    priceDemandCurve: List[Dict[str, float]]


class BackendPricingOptimizationResponse(BaseModel):
    """Response payload directly consumable by Node.js backend pricingController."""

    success: bool = True
    message: str = "Causal DML analysis calculated successfully"
    data: BackendExperimentData


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

