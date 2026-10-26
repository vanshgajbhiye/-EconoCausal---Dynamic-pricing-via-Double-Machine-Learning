"""Service layer managing the EconoCausal pipeline lifecycle and inference executions.

Provides high-performance, asynchronous-friendly inference methods, latency tracking,
input normalization, and defensive error handling.
"""

import datetime
import os
import time
from typing import Any, Dict, List, Optional
import numpy as np
import pandas as pd

from src.api.metrics import record_drift_audit
from src.api.schemas import (
    BatchPricingRequest,
    BatchPricingResponse,
    DriftAuditRequest,
    DriftAuditResponse,
    ElasticityDriftDetail,
    ElasticityRequest,
    ElasticityResponse,
    ElasticityResponseItem,
    FeatureDriftDetail,
    HealthResponse,
    ModelMetadataResponse,
    PricingItemInput,
    PortfolioScenarioRequest,
    PortfolioScenarioResponse,
    RefutationAuditDetail,
    ScenarioResult,
    SingleItemPricingRequest,
    SingleItemPricingResponse,
)
from src.monitoring.drift import CausalDriftDetector
from src.pipeline import EconoCausalPipeline


DEFAULT_MODEL_PATH = os.path.join("models", "econocausal_pipeline.joblib")
DEFAULT_DATA_PATH = os.path.join("data", "raw", "retail_price.csv")


class CausalPricingService:
    """Singleton service manager for the EconoCausal dynamic pricing microservice."""

    _instance: Optional["CausalPricingService"] = None

    def __init__(self, model_path: Optional[str] = None, baseline_path: Optional[str] = None):
        self.start_time = time.time()
        self.model_path = model_path or os.getenv("ECONOCAUSAL_MODEL_PATH", DEFAULT_MODEL_PATH)
        self.baseline_path = baseline_path or os.getenv("ECONOCAUSAL_DATA_PATH", DEFAULT_DATA_PATH)
        self.pipeline: Optional[EconoCausalPipeline] = None
        self._cached_baseline_df: Optional[pd.DataFrame] = None
        self._total_requests_served = 0
        self._load_pipeline()

    @classmethod
    def get_instance(cls, model_path: Optional[str] = None) -> "CausalPricingService":
        """Get or initialize singleton service instance."""
        if cls._instance is None:
            cls._instance = cls(model_path=model_path)
        return cls._instance

    @classmethod
    def reset_instance(cls, model_path: Optional[str] = None) -> "CausalPricingService":
        """Reset and reinitialize service singleton (useful for isolated testing)."""
        cls._instance = cls(model_path=model_path)
        return cls._instance

    def _load_pipeline(self) -> None:
        """Attempt to load persisted EconoCausalPipeline from disk."""
        if os.path.exists(self.model_path):
            try:
                self.pipeline = EconoCausalPipeline.load(self.model_path)
            except Exception as e:
                # Log or keep pipeline as None for degraded status
                self.pipeline = None
        else:
            self.pipeline = None

    @property
    def is_ready(self) -> bool:
        """Check if pipeline is loaded and fitted."""
        return self.pipeline is not None and self.pipeline.is_fitted_

    def get_health(self) -> HealthResponse:
        """Return microservice health and readiness diagnostics."""
        uptime = time.time() - self.start_time
        timestamp = datetime.datetime.now(datetime.timezone.utc).isoformat()

        if self.is_ready:
            return HealthResponse(
                status="healthy",
                service="econocausal-pricing-service",
                version="1.0.0",
                uptime_seconds=round(uptime, 2),
                model_loaded=True,
                model_path=self.model_path,
                baseline_ate=round(self.pipeline.baseline_ate_, 4) if self.pipeline.baseline_ate_ is not None else None,
                refutation_pass_rate=round(self.pipeline.refutation_pass_rate_, 2) if self.pipeline.refutation_pass_rate_ is not None else None,
                timestamp=timestamp,
            )
        else:
            return HealthResponse(
                status="degraded",
                service="econocausal-pricing-service",
                version="1.0.0",
                uptime_seconds=round(uptime, 2),
                model_loaded=False,
                model_path=self.model_path,
                baseline_ate=None,
                refutation_pass_rate=None,
                timestamp=timestamp,
            )

    def get_metadata(self) -> ModelMetadataResponse:
        """Return comprehensive metadata for the loaded causal model."""
        if not self.is_ready:
            raise ValueError("Model pipeline is not loaded or fitted.")

        meta = self.pipeline.metadata_
        return ModelMetadataResponse(
            is_fitted=self.pipeline.is_fitted_,
            model_path=self.model_path,
            trained_at=meta.get("trained_at"),
            training_samples_total=meta.get("n_samples_total"),
            training_samples_train=meta.get("n_samples_train"),
            encoded_feature_count=meta.get("n_features_encoded"),
            baseline_ate=meta.get("baseline_ate"),
            baseline_ate_se=meta.get("baseline_ate_se"),
            baseline_ate_pvalue=meta.get("baseline_ate_pvalue"),
            refutation_pass_rate=meta.get("refutation_pass_rate"),
            cate_mean_elasticity=meta.get("cate_mean_elasticity"),
            cate_min_elasticity=meta.get("cate_min_elasticity"),
            cate_max_elasticity=meta.get("cate_max_elasticity"),
            optimizer_bounds={
                "min_price_change_pct": self.pipeline.min_price_change_pct,
                "max_price_change_pct": self.pipeline.max_price_change_pct,
                "default_margin": self.pipeline.default_margin,
                "min_margin": self.pipeline.min_margin,
                "charm_pricing": self.pipeline.charm_pricing,
            },
            feature_imputation_medians=self.pipeline.feature_imputation_values_,
        )

    def predict_single_item(
        self, request: SingleItemPricingRequest
    ) -> SingleItemPricingResponse:
        """Execute real-time optimal pricing recommendation for a single SKU."""
        if not self.is_ready:
            raise ValueError("Model pipeline is not loaded or fitted. Cannot serve predictions.")

        t0 = time.perf_counter()
        item_dict = request.item.to_inference_dict()
        pred_dict = self.pipeline.predict_item(item_dict, objective=request.objective)
        latency_ms = round((time.perf_counter() - t0) * 1000.0, 3)

        self._total_requests_served += 1

        return SingleItemPricingResponse(
            status=pred_dict["status"],
            product_id=request.item.product_id,
            action=pred_dict["action"],
            baseline_price=pred_dict["baseline_price"],
            optimal_price=pred_dict["optimal_price"],
            price_change_pct=pred_dict["price_change_pct"],
            causal_elasticity=pred_dict["causal_elasticity"],
            unit_cost=pred_dict["unit_cost"],
            baseline_margin_pct=pred_dict["baseline_margin_pct"],
            projected_margin_pct=pred_dict["projected_margin_pct"],
            baseline_qty=pred_dict["baseline_qty"],
            projected_qty=pred_dict["projected_qty"],
            demand_uplift_pct=pred_dict["demand_uplift_pct"],
            baseline_revenue=pred_dict["baseline_revenue"],
            projected_revenue=pred_dict["projected_revenue"],
            revenue_uplift_pct=pred_dict["revenue_uplift_pct"],
            baseline_profit=pred_dict["baseline_profit"],
            projected_profit=pred_dict["projected_profit"],
            profit_uplift_pct=pred_dict["profit_uplift_pct"],
            bound_hit=pred_dict.get("bound_hit"),
            objective=pred_dict["objective"],
            inference_latency_ms=latency_ms,
        )

    def batch_recommend(self, request: BatchPricingRequest) -> BatchPricingResponse:
        """Execute dynamic pricing optimization over an entire batch of items."""
        if not self.is_ready:
            raise ValueError("Model pipeline is not loaded or fitted. Cannot serve predictions.")

        t0 = time.perf_counter()
        item_dicts = [item.to_inference_dict() for item in request.items]
        batch_df = pd.DataFrame(item_dicts)

        rec_df = self.pipeline.batch_recommend(batch_df, objective=request.objective)

        # Assemble individual responses
        recommendations: List[SingleItemPricingResponse] = []
        for idx, row in rec_df.iterrows():
            item_req = request.items[idx]
            price_change = float(row.get("price_change_pct", 0.0))
            if abs(price_change) <= 1.0 and abs(price_change) > 0.0:
                price_change_pct = price_change * 100.0
            else:
                price_change_pct = price_change

            recommendations.append(
                SingleItemPricingResponse(
                    status="SUCCESS",
                    product_id=item_req.product_id or str(row.get("product_id", f"item_{idx}")),
                    action=str(row.get("pricing_action", "HOLD")),
                    baseline_price=round(float(row["baseline_price"]), 2),
                    optimal_price=round(float(row["optimal_price"]), 2),
                    price_change_pct=round(price_change_pct, 2),
                    causal_elasticity=round(float(row["elasticity"]), 4),
                    unit_cost=round(float(row.get("unit_cost", float(row["baseline_price"]) * 0.70)), 2),
                    baseline_margin_pct=round(float(row.get("baseline_margin", 0.30)) * 100.0, 2),
                    projected_margin_pct=round(float(row.get("projected_margin", 0.35)) * 100.0, 2),
                    baseline_qty=round(float(row["baseline_qty"]), 1),
                    projected_qty=round(float(row["projected_qty"]), 1),
                    demand_uplift_pct=round(float(row["demand_uplift_pct"]) * 100.0 if abs(float(row["demand_uplift_pct"])) < 2.0 else float(row["demand_uplift_pct"]), 2),
                    baseline_revenue=round(float(row["baseline_revenue"]), 2),
                    projected_revenue=round(float(row["projected_revenue"]), 2),
                    revenue_uplift_pct=round(float(row["revenue_uplift_pct"]) * 100.0 if abs(float(row["revenue_uplift_pct"])) < 2.0 else float(row["revenue_uplift_pct"]), 2),
                    baseline_profit=round(float(row["baseline_profit"]), 2),
                    projected_profit=round(float(row["projected_profit"]), 2),
                    profit_uplift_pct=round(float(row["profit_uplift_pct"]) * 100.0 if abs(float(row["profit_uplift_pct"])) < 2.0 else float(row["profit_uplift_pct"]), 2),
                    bound_hit=str(row.get("bound_hit", "")) or None,
                    objective=request.objective,
                    inference_latency_ms=None,
                )
            )

        # Portfolio aggregate metrics
        base_rev = float(rec_df["baseline_revenue"].sum())
        proj_rev = float(rec_df["projected_revenue"].sum())
        base_prof = float(rec_df["baseline_profit"].sum())
        proj_prof = float(rec_df["projected_profit"].sum())

        rev_uplift = ((proj_rev - base_rev) / base_rev * 100.0) if base_rev > 0 else 0.0
        prof_uplift = ((proj_prof - base_prof) / base_prof * 100.0) if base_prof > 0 else 0.0

        portfolio_summary = {
            "total_items": len(recommendations),
            "total_baseline_revenue": round(base_rev, 2),
            "total_projected_revenue": round(proj_rev, 2),
            "portfolio_revenue_uplift_pct": round(rev_uplift, 2),
            "total_baseline_profit": round(base_prof, 2),
            "total_projected_profit": round(proj_prof, 2),
            "portfolio_profit_uplift_pct": round(prof_uplift, 2),
            "markup_count": sum(1 for r in recommendations if r.action == "MARKUP"),
            "discount_count": sum(1 for r in recommendations if r.action == "DISCOUNT"),
            "hold_count": sum(1 for r in recommendations if r.action == "HOLD"),
        }

        batch_latency_ms = round((time.perf_counter() - t0) * 1000.0, 3)
        self._total_requests_served += len(request.items)

        return BatchPricingResponse(
            total_items=len(recommendations),
            objective=request.objective,
            recommendations=recommendations,
            portfolio_summary=portfolio_summary,
            batch_latency_ms=batch_latency_ms,
        )

    def predict_elasticity(self, request: ElasticityRequest) -> ElasticityResponse:
        """Estimate heterogeneous causal elasticity theta(X) for a batch of items."""
        if not self.is_ready:
            raise ValueError("Model pipeline is not loaded or fitted.")

        t0 = time.perf_counter()
        item_dicts = [item.to_inference_dict() for item in request.items]
        df = pd.DataFrame(item_dicts)

        elasticities = self.pipeline.predict_elasticity(df)

        predictions: List[ElasticityResponseItem] = []
        for idx, e_val in enumerate(elasticities):
            item = request.items[idx]
            val = float(e_val)
            abs_val = abs(val)
            if abs_val > 1.5:
                classification = "Highly Elastic"
            elif abs_val > 1.0:
                classification = "Elastic"
            else:
                classification = "Inelastic"

            predictions.append(
                ElasticityResponseItem(
                    product_id=item.product_id,
                    product_category_name=item.product_category_name,
                    predicted_elasticity=round(val, 4),
                    elasticity_classification=classification,
                )
            )

        latency_ms = round((time.perf_counter() - t0) * 1000.0, 3)

        return ElasticityResponse(
            total_items=len(predictions),
            predictions=predictions,
            latency_ms=latency_ms,
        )

    def simulate_portfolio_scenarios(
        self, request: PortfolioScenarioRequest
    ) -> PortfolioScenarioResponse:
        """Simulate and compare catalog performance across competitive and causal strategies."""
        if not self.is_ready:
            raise ValueError("Model pipeline is not loaded or fitted.")

        t0 = time.perf_counter()
        item_dicts = [item.to_inference_dict() for item in request.items]
        df = pd.DataFrame(item_dicts)

        scenarios_df = self.pipeline.simulate_scenarios(df, comp_col=request.comp_col)

        results: List[ScenarioResult] = []
        best_profit = -float("inf")
        best_strat = "Status Quo (Baseline)"

        for _, row in scenarios_df.iterrows():
            strat_name = str(row["Strategy"])
            profit = float(row["Total Profit ($)"])
            if profit > best_profit:
                best_profit = profit
                best_strat = strat_name

            results.append(
                ScenarioResult(
                    strategy=strat_name,
                    total_revenue=round(float(row["Total Revenue ($)"]), 2),
                    revenue_uplift_pct=round(float(row["Revenue Uplift (%)"]), 2),
                    total_profit=round(profit, 2),
                    profit_uplift_pct=round(float(row["Profit Uplift (%)"]), 2),
                    total_units_sold=round(float(row["Total Units Sold"]), 1),
                    units_uplift_pct=round(float(row["Units Uplift (%)"]), 2),
                    average_margin_pct=round(float(row["Average Margin (%)"]), 2),
                )
            )

        latency_ms = round((time.perf_counter() - t0) * 1000.0, 3)

        return PortfolioScenarioResponse(
            total_items=len(request.items),
            scenarios=results,
            recommended_strategy=best_strat,
            simulation_latency_ms=latency_ms,
        )

    def _get_baseline_df(self) -> pd.DataFrame:
        """Load or return cached reference baseline dataset."""
        if self._cached_baseline_df is not None:
            return self._cached_baseline_df
        if os.path.exists(self.baseline_path):
            try:
                self._cached_baseline_df = pd.read_csv(self.baseline_path)
                return self._cached_baseline_df
            except Exception:
                pass
        return pd.DataFrame()

    def audit_drift(self, request: DriftAuditRequest) -> DriftAuditResponse:
        """Audit incoming production batch for covariate and elasticity drift."""
        t0 = time.perf_counter()
        target_df = pd.DataFrame([item.to_inference_dict() for item in request.items])
        baseline_df = self._get_baseline_df()

        if baseline_df.empty:
            baseline_df = target_df.copy()

        detector = CausalDriftDetector(
            psi_warning_threshold=request.psi_warning_threshold,
            psi_alert_threshold=request.psi_alert_threshold,
        )

        report = detector.generate_report(
            baseline_df=baseline_df,
            target_df=target_df,
            pipeline=self.pipeline,
            run_refutations=request.run_refutations,
        )

        refut_pass_rate = None
        if report.refutation_audit is not None:
            refut_pass_rate = report.refutation_audit.get("pass_rate")
        elif self.is_ready:
            refut_pass_rate = self.pipeline.refutation_pass_rate_

        record_drift_audit(
            max_psi=report.max_confounder_psi,
            mean_psi=report.mean_confounder_psi,
            status_str=report.overall_status,
            refutation_pass_rate=refut_pass_rate,
        )

        latency_ms = round((time.perf_counter() - t0) * 1000.0, 3)

        confounders_dict = {
            k: FeatureDriftDetail(**v.to_dict())
            for k, v in report.confounder_drift.items()
        }

        treatment_detail = (
            FeatureDriftDetail(**report.treatment_drift.to_dict())
            if report.treatment_drift
            else None
        )
        outcome_detail = (
            FeatureDriftDetail(**report.outcome_drift.to_dict())
            if report.outcome_drift
            else None
        )
        elasticity_detail = (
            ElasticityDriftDetail(**report.elasticity_drift)
            if report.elasticity_drift
            else None
        )
        refutation_detail = (
            RefutationAuditDetail(**report.refutation_audit)
            if report.refutation_audit
            else None
        )

        return DriftAuditResponse(
            timestamp=report.timestamp,
            dataset_baseline_rows=report.dataset_baseline_rows,
            dataset_target_rows=report.dataset_target_rows,
            overall_status=report.overall_status,
            recommendation=report.recommendation,
            max_confounder_psi=report.max_confounder_psi,
            mean_confounder_psi=report.mean_confounder_psi,
            num_drifting_confounders=report.num_drifting_confounders,
            confounders=confounders_dict,
            treatment_drift=treatment_detail,
            outcome_drift=outcome_detail,
            elasticity_drift=elasticity_detail,
            refutation_audit=refutation_detail,
            audit_latency_ms=latency_ms,
        )

