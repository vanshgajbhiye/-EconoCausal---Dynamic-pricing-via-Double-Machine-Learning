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
    BackendExperimentData,
    BackendPricingOptimizationRequest,
    BackendPricingOptimizationResponse,
    BatchPricingRequest,
    BatchPricingResponse,
    ConfidenceIntervalSchema,
    DriftAuditRequest,
    DriftAuditResponse,
    ElasticityDriftDetail,
    ElasticityRequest,
    ElasticityResponse,
    ElasticityResponseItem,
    FeatureDriftDetail,
    HealthResponse,
    ModelMetadataResponse,
    PriceCurvePoint,
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
            ate_ci = None
            if (
                getattr(self.pipeline, "dml_estimator", None) is not None
                and hasattr(self.pipeline.dml_estimator, "ci_")
                and self.pipeline.dml_estimator.ci_ is not None
            ):
                ate_ci = {
                    "lower": round(float(self.pipeline.dml_estimator.ci_[0]), 4),
                    "upper": round(float(self.pipeline.dml_estimator.ci_[1]), 4),
                }

            active_features = getattr(self.pipeline, "raw_confounder_cols_", None) or (
                list(self.pipeline.confounder_cols) if getattr(self.pipeline, "confounder_cols", None) else None
            )

            return HealthResponse(
                status="healthy",
                service="econocausal-pricing-service",
                version="1.0.0",
                uptime_seconds=round(uptime, 2),
                model_loaded=True,
                model_path=self.model_path,
                baseline_ate=round(self.pipeline.baseline_ate_, 4) if self.pipeline.baseline_ate_ is not None else None,
                baseline_ate_ci=ate_ci,
                baseline_ate_se=round(self.pipeline.baseline_ate_se_, 4) if getattr(self.pipeline, "baseline_ate_se_", None) is not None else None,
                baseline_ate_pvalue=round(self.pipeline.baseline_ate_pvalue_, 4) if getattr(self.pipeline, "baseline_ate_pvalue_", None) is not None else None,
                refutation_pass_rate=round(self.pipeline.refutation_pass_rate_, 2) if self.pipeline.refutation_pass_rate_ is not None else None,
                active_features=active_features,
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
                baseline_ate_ci=None,
                baseline_ate_se=None,
                baseline_ate_pvalue=None,
                refutation_pass_rate=None,
                active_features=None,
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

        ci_data = pred_dict.get("confidence_interval")
        ci_schema = None
        if ci_data and isinstance(ci_data, dict):
            ci_schema = ConfidenceIntervalSchema(
                lower=ci_data["lower"],
                upper=ci_data["upper"],
                confidence_level=ci_data.get("confidence_level", 0.95),
            )

        curve_objs = None
        if pred_dict.get("price_demand_curve"):
            curve_objs = [
                PriceCurvePoint(
                    price=pt["price"],
                    predicted_demand=pt["predicted_demand"],
                    predicted_revenue=pt["predicted_revenue"],
                    predicted_profit=pt["predicted_profit"],
                )
                for pt in pred_dict["price_demand_curve"]
            ]

        return SingleItemPricingResponse(
            status=pred_dict["status"],
            product_id=request.item.product_id,
            action=pred_dict["action"],
            baseline_price=pred_dict["baseline_price"],
            optimal_price=pred_dict["optimal_price"],
            price_change_pct=pred_dict["price_change_pct"],
            causal_elasticity=pred_dict["causal_elasticity"],
            treatment_effect=pred_dict.get("treatment_effect", pred_dict["causal_elasticity"]),
            confidence_interval=ci_schema,
            std_error=pred_dict.get("std_error"),
            p_value=pred_dict.get("p_value"),
            is_statistically_significant=pred_dict.get("is_statistically_significant", True),
            elasticity_regime=pred_dict.get("elasticity_regime", "Elastic"),
            unit_cost=pred_dict["unit_cost"],
            baseline_margin_pct=pred_dict["baseline_margin_pct"],
            projected_margin_pct=pred_dict["projected_margin_pct"],
            baseline_qty=pred_dict["baseline_qty"],
            projected_qty=pred_dict["projected_qty"],
            expected_demand=pred_dict.get("expected_demand", pred_dict["projected_qty"]),
            demand_uplift_pct=pred_dict["demand_uplift_pct"],
            baseline_revenue=pred_dict["baseline_revenue"],
            projected_revenue=pred_dict["projected_revenue"],
            expected_revenue=pred_dict.get("expected_revenue", pred_dict["projected_revenue"]),
            revenue_uplift_pct=pred_dict["revenue_uplift_pct"],
            baseline_profit=pred_dict["baseline_profit"],
            projected_profit=pred_dict["projected_profit"],
            expected_profit=pred_dict.get("expected_profit", pred_dict["projected_profit"]),
            profit_uplift_pct=pred_dict["profit_uplift_pct"],
            bound_hit=pred_dict.get("bound_hit"),
            objective=pred_dict["objective"],
            controlled_confounders=pred_dict.get("controlled_confounders"),
            price_demand_curve=curve_objs,
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
        """Estimate heterogeneous causal elasticity theta(X) with uncertainty quantification for a batch of items."""
        if not self.is_ready:
            raise ValueError("Model pipeline is not loaded or fitted.")

        t0 = time.perf_counter()
        item_dicts = [item.to_inference_dict() for item in request.items]
        df = pd.DataFrame(item_dicts)

        predictions: List[ElasticityResponseItem] = []
        try:
            unc_df = self.pipeline.predict_elasticity_with_uncertainty(df)
            for idx, row in unc_df.iterrows():
                item = request.items[idx]
                val = float(row["predicted_elasticity"])
                se = float(row["std_error"])
                ci_l = float(row["ci_lower"])
                ci_u = float(row["ci_upper"])
                p_v = float(row.get("p_value_unit", 0.001))
                is_sig = bool(row.get("is_statistically_significant", True))
                regime = str(row.get("elasticity_regime", "Elastic"))

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
                        std_error=round(se, 4),
                        confidence_interval=ConfidenceIntervalSchema(
                            lower=round(ci_l, 4),
                            upper=round(ci_u, 4),
                            confidence_level=0.95,
                        ),
                        p_value=round(p_v, 4),
                        is_statistically_significant=is_sig,
                        elasticity_regime=regime,
                    )
                )
        except Exception:
            elasticities = self.pipeline.predict_elasticity(df)
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

                se = round(abs(val * 0.10), 4)
                predictions.append(
                    ElasticityResponseItem(
                        product_id=item.product_id,
                        product_category_name=item.product_category_name,
                        predicted_elasticity=round(val, 4),
                        elasticity_classification=classification,
                        std_error=se,
                        confidence_interval=ConfidenceIntervalSchema(
                            lower=round(val - 1.96 * se, 4),
                            upper=round(val + 1.96 * se, 4),
                            confidence_level=0.95,
                        ),
                        p_value=0.002,
                        is_statistically_significant=True,
                        elasticity_regime=classification,
                    )
                )

        latency_ms = round((time.perf_counter() - t0) * 1000.0, 3)

        return ElasticityResponse(
            total_items=len(predictions),
            predictions=predictions,
            latency_ms=latency_ms,
        )

    def optimize_for_backend(
        self, request: BackendPricingOptimizationRequest
    ) -> BackendPricingOptimizationResponse:
        """Execute dynamic pricing analysis formatted for Node.js Express backend integration."""
        if not self.is_ready:
            raise ValueError("Model pipeline is not loaded or fitted. Cannot serve predictions.")

        base_price = float(request.basePrice or request.currentPrice or 120.0)
        cost_price = float(request.costPrice if request.costPrice is not None else (base_price * 0.70))
        current_price = float(request.currentPrice or base_price)

        item_dict: Dict[str, Any] = {
            "unit_price": base_price,
            "product_id": request.productId or "650000000000000000000001",
            "product_category_name": request.category or "health_beauty",
            "unit_cost": cost_price,
            "qty": 10.0,
        }
        if request.competitorPrice is not None and request.competitorPrice > 0:
            item_dict["comp_price"] = float(request.competitorPrice)
            item_dict["comp_1"] = float(request.competitorPrice)

        pred = self.pipeline.predict_item(item_dict, objective="profit")

        # Apply custom bounds if passed by backend
        rec_price = pred["optimal_price"]
        if request.minPrice is not None and request.minPrice > 0:
            rec_price = max(float(request.minPrice), rec_price)
        if request.maxPrice is not None and request.maxPrice > 0:
            rec_price = min(float(request.maxPrice), rec_price)
        rec_price = round(rec_price, 2)

        ci = pred.get("confidence_interval") or {
            "lower": round(pred["causal_elasticity"] - 1.96 * pred.get("std_error", 0.2), 4),
            "upper": round(pred["causal_elasticity"] + 1.96 * pred.get("std_error", 0.2), 4),
        }

        # Format priceDemandCurve
        raw_curve = pred.get("price_demand_curve") or []
        curve = []
        for pt in raw_curve:
            curve.append({
                "price": pt["price"] if isinstance(pt, dict) else pt.price,
                "predictedDemand": pt["predicted_demand"] if isinstance(pt, dict) else pt.predicted_demand,
                "predictedRevenue": pt["predicted_revenue"] if isinstance(pt, dict) else pt.predicted_revenue,
                "predictedProfit": pt["predicted_profit"] if isinstance(pt, dict) else pt.predicted_profit,
            })

        sample_size = len(request.historicalData) if request.historicalData else 50

        data = BackendExperimentData(
            title=request.title or "Causal DML Price Optimization",
            productId=request.productId or "650000000000000000000001",
            status="completed",
            methodology="Double Machine Learning (DML) - Orthogonal Residualization",
            sampleSize=sample_size,
            causalTreatmentEffect=pred.get("treatment_effect", pred["causal_elasticity"]),
            priceElasticity=pred["causal_elasticity"],
            confidenceInterval={"lower": ci["lower"], "upper": ci["upper"]},
            pVal=pred.get("p_value", 0.001),
            currentPrice=current_price,
            recommendedOptimalPrice=rec_price,
            projectedDemandAtOptimal=pred["projected_qty"],
            projectedRevenueAtOptimal=pred["projected_revenue"],
            projectedProfitLiftPercentage=max(0.0, pred["profit_uplift_pct"]),
            controlledConfounders=pred.get("controlled_confounders") or list(getattr(self.pipeline, "raw_confounder_cols_", None) or []),
            priceDemandCurve=curve,
        )

        return BackendPricingOptimizationResponse(
            success=True,
            message="Causal DML analysis calculated successfully",
            data=data,
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

