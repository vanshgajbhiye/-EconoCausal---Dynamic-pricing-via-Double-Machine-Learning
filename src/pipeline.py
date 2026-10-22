"""End-to-End Orchestration and Inference Pipeline for EconoCausal Dynamic Pricing.

Integrates data ingestion, anti-leakage preprocessing, Double Machine Learning (DML),
Conditional Average Treatment Effects (CATE), causal refutation diagnostics,
and constrained dynamic pricing optimization into a unified, serializable pipeline.

Supports:
- Batch pricing inference and catalog recommendation sheet export
- Real-time / on-demand single-SKU pricing inference with business guardrails
- Counterfactual portfolio uplift simulation across pricing strategies
- Production model persistence (save / load via joblib)
"""

import datetime
import os
from typing import Any, Dict, List, Optional, Tuple, Union
import joblib
import numpy as np
import pandas as pd

from src.data.load_data import (
    DEFAULT_CONFOUNDER_COLS,
    DEFAULT_OUTCOME_COL,
    DEFAULT_TREATMENT_COL,
    load_retail_price_dataset,
    validate_schema,
)
from src.data.preprocess import (
    CausalPreprocessor,
    split_causal_data,
)
from src.models.cate import CATEEstimator
from src.models.optimizer import DynamicPricingOptimizer, ItemOptimizationResult
from src.models.refutation import CausalRefuter
from src.models.train_dml import DoubleMachineLearning


class EconoCausalPipeline:
    """End-to-End Causal Dynamic Pricing Pipeline.

    Encapsulates the complete causal machine learning and dynamic pricing lifecycle:
    1. Preprocessing & Confounder Encoding (StandardScaler + OneHotEncoder)
    2. Double Machine Learning (DML) Baseline ATE Elasticity Estimation
    3. Conditional Average Treatment Effect (CATE) Heterogeneous Elasticity Estimation
    4. Causal Refutation Diagnostics (Placebo, Common Cause, Data Subsetting)
    5. Microeconomic Dynamic Pricing Optimization (Lerner Index / Amoroso-Robinson)
    6. Counterfactual Uplift Simulation (Demand, Revenue, Gross Profit)

    Parameters
    ----------
    treatment_col : str, default="unit_price"
        Column name for price treatment T.
    outcome_col : str, default="qty"
        Column name for demand volume outcome Y.
    confounder_cols : Optional[List[str]], default=None
        List of confounder columns. If None, uses DEFAULT_CONFOUNDER_COLS.
    categorical_confounders : Optional[List[str]], default=("product_category_name",)
        List of categorical confounder columns to one-hot encode.
    log_transform_treatment : bool, default=True
        Whether to model price in log scale for constant elasticity.
    log_transform_outcome : bool, default=True
        Whether to model demand in log scale.
    n_splits : int, default=5
        Number of cross-fitting folds for DML residualization.
    cate_alpha : float, default=1.0
        Ridge penalty for second-stage CATE interaction regression.
    default_margin : float, default=0.30
        Default gross margin fraction (0.30 = 30%) used when unit costs are unknown.
    min_price_change_pct : float, default=-0.30
        Maximum allowable price discount (-0.30 = -30%).
    max_price_change_pct : float, default=0.20
        Maximum allowable price markup (+0.20 = +20%).
    min_margin : float, default=0.05
        Hard minimum floor margin to prevent selling at negative/unsustainable margins.
    charm_pricing : bool, default=True
        Whether to adjust final prices to charm psychological endings (.99).
    random_state : int, default=42
        Random seed for cross-fitting splits and refutations.
    """

    def __init__(
        self,
        treatment_col: str = DEFAULT_TREATMENT_COL,
        outcome_col: str = DEFAULT_OUTCOME_COL,
        confounder_cols: Optional[List[str]] = None,
        categorical_confounders: Optional[List[str]] = None,
        log_transform_treatment: bool = True,
        log_transform_outcome: bool = True,
        n_splits: int = 5,
        cate_alpha: float = 1.0,
        default_margin: float = 0.30,
        min_price_change_pct: float = -0.30,
        max_price_change_pct: float = 0.20,
        min_margin: float = 0.05,
        charm_pricing: bool = True,
        random_state: int = 42,
    ):
        self.treatment_col = treatment_col
        self.outcome_col = outcome_col
        self.confounder_cols = list(confounder_cols) if confounder_cols is not None else None
        self.categorical_confounders = (
            list(categorical_confounders)
            if categorical_confounders is not None
            else ["product_category_name"]
        )
        self.log_transform_treatment = log_transform_treatment
        self.log_transform_outcome = log_transform_outcome
        self.n_splits = n_splits
        self.cate_alpha = cate_alpha
        self.default_margin = default_margin
        self.min_price_change_pct = min_price_change_pct
        self.max_price_change_pct = max_price_change_pct
        self.min_margin = min_margin
        self.charm_pricing = charm_pricing
        self.random_state = random_state

        # Internal component instances
        self.preprocessor: Optional[CausalPreprocessor] = None
        self.dml_estimator: Optional[DoubleMachineLearning] = None
        self.cate_estimator: Optional[CATEEstimator] = None
        self.optimizer: Optional[DynamicPricingOptimizer] = None
        self.refuter: Optional[CausalRefuter] = None

        # Learned attributes & metadata
        self.is_fitted_: bool = False
        self.baseline_ate_: Optional[float] = None
        self.baseline_ate_se_: Optional[float] = None
        self.baseline_ate_pvalue_: Optional[float] = None
        self.refutation_results_: Optional[Dict[str, Any]] = None
        self.refutation_pass_rate_: Optional[float] = None
        self.training_feature_names_: List[str] = []
        self.raw_confounder_cols_: List[str] = []
        self.raw_categorical_cols_: List[str] = []
        self.raw_numerical_cols_: List[str] = []
        self.feature_imputation_values_: Dict[str, Any] = {}
        self.metadata_: Dict[str, Any] = {}

    def fit(
        self,
        df: pd.DataFrame,
        test_size: float = 0.20,
        run_refutations: bool = True,
        refutation_repeats: int = 2,
    ) -> "EconoCausalPipeline":
        """Train the full causal dynamic pricing pipeline on a raw retail dataset.

        Parameters
        ----------
        df : pd.DataFrame
            Raw retail transaction or catalog DataFrame.
        test_size : float, default=0.20
            Fraction of data held out for validation.
        run_refutations : bool, default=True
            Whether to execute the causal refutation test battery.
        refutation_repeats : int, default=2
            Number of iterations for refutation checks.

        Returns
        -------
        EconoCausalPipeline
            Fitted pipeline instance ready for inference and optimization.
        """
        start_time = datetime.datetime.now(datetime.timezone.utc)

        # 1. Schema Validation
        validate_schema(
            df,
            treatment_col=self.treatment_col,
            outcome_col=self.outcome_col,
            confounder_cols=self.confounder_cols,
        )

        # 2. Determine and record confounder columns
        if self.confounder_cols is None:
            active_confounders = [c for c in DEFAULT_CONFOUNDER_COLS if c in df.columns]
        else:
            active_confounders = [c for c in self.confounder_cols if c in df.columns]

        active_categoricals = [c for c in self.categorical_confounders if c in df.columns]
        for cat in active_categoricals:
            if cat not in active_confounders:
                active_confounders.append(cat)

        self.raw_confounder_cols_ = active_confounders
        self.raw_categorical_cols_ = active_categoricals
        self.raw_numerical_cols_ = [c for c in active_confounders if c not in active_categoricals]

        # 3. Store training medians / modes for robust real-time imputation
        for col in self.raw_numerical_cols_:
            self.feature_imputation_values_[col] = float(df[col].median(skipna=True))
        for col in self.raw_categorical_cols_:
            mode_val = df[col].mode(dropna=True)
            self.feature_imputation_values_[col] = str(mode_val.iloc[0]) if len(mode_val) > 0 else "unknown"

        # 4. Anti-leakage Causal Splitting & Preprocessing
        splits = split_causal_data(
            df,
            test_size=test_size,
            random_state=self.random_state,
            treatment_col=self.treatment_col,
            outcome_col=self.outcome_col,
            confounder_cols=self.raw_confounder_cols_,
            categorical_confounders=self.raw_categorical_cols_,
            log_transform_treatment=self.log_transform_treatment,
            log_transform_outcome=self.log_transform_outcome,
        )

        self.preprocessor = splits["preprocessor"]
        self.training_feature_names_ = splits["feature_names"]
        X_train = splits["X_train"]
        T_train = splits["T_train"]
        Y_train = splits["Y_train"]

        # 5. Fit Double Machine Learning Baseline (ATE)
        self.dml_estimator = DoubleMachineLearning(
            n_folds=self.n_splits,
            random_state=self.random_state,
        )
        self.dml_estimator.fit(X_train, T_train, Y_train)
        self.baseline_ate_ = self.dml_estimator.theta_
        self.baseline_ate_se_ = self.dml_estimator.std_error_
        self.baseline_ate_pvalue_ = self.dml_estimator.p_value_

        # 6. Fit Heterogeneous CATE Estimator
        self.cate_estimator = CATEEstimator(
            dml_base=self.dml_estimator,
            alpha=self.cate_alpha,
            include_intercept=True,
            random_state=self.random_state,
        )
        self.cate_estimator.fit(X_train, T_train, Y_train)

        # 7. Optional Causal Refutation Suite
        if run_refutations:
            self.refuter = CausalRefuter(
                model=self.dml_estimator,
                random_state=self.random_state,
            )
            self.refutation_results_ = self.refuter.run_all_refutations(
                X_train, T_train, Y_train, n_repeats=refutation_repeats
            )
            self.refutation_pass_rate_ = float(
                self.refutation_results_.get(
                    "robustness_score_percent",
                    self.refutation_results_.get("robustness_score", 0.0),
                )
            )
        else:
            self.refuter = None
            self.refutation_results_ = None
            self.refutation_pass_rate_ = None

        # 8. Initialize Dynamic Pricing Optimizer
        self.optimizer = DynamicPricingOptimizer(
            default_margin=self.default_margin,
            min_price_change_pct=self.min_price_change_pct,
            max_price_change_pct=self.max_price_change_pct,
            min_margin=self.min_margin,
            charm_pricing=self.charm_pricing,
        )

        end_time = datetime.datetime.now(datetime.timezone.utc)
        self.is_fitted_ = True

        # 9. Store Metadata
        self.metadata_ = {
            "trained_at": end_time.isoformat(),
            "training_duration_sec": (end_time - start_time).total_seconds(),
            "n_samples_total": len(df),
            "n_samples_train": len(X_train),
            "n_samples_val": len(splits["X_test"]),
            "n_features_encoded": len(self.training_feature_names_),
            "baseline_ate": self.baseline_ate_,
            "baseline_ate_se": self.baseline_ate_se_,
            "baseline_ate_pvalue": self.baseline_ate_pvalue_,
            "refutation_pass_rate": self.refutation_pass_rate_,
            "cate_mean_elasticity": float(np.mean(self.cate_estimator.cate_predictions_)),
            "cate_min_elasticity": float(np.min(self.cate_estimator.cate_predictions_)),
            "cate_max_elasticity": float(np.max(self.cate_estimator.cate_predictions_)),
        }

        return self

    def _align_input_dataframe(self, input_df: pd.DataFrame) -> pd.DataFrame:
        """Align and impute missing confounder columns before preprocessing.

        Ensures production resilience when client requests omit non-critical columns.

        Parameters
        ----------
        input_df : pd.DataFrame
            Raw input DataFrame containing some or all confounder columns.

        Returns
        -------
        pd.DataFrame
            Aligned DataFrame containing all expected raw confounders with defaults.
        """
        aligned_df = input_df.copy()

        # Fill or insert missing numerical confounders
        for col in self.raw_numerical_cols_:
            if col not in aligned_df.columns:
                aligned_df[col] = self.feature_imputation_values_.get(col, 0.0)
            else:
                aligned_df[col] = aligned_df[col].fillna(
                    self.feature_imputation_values_.get(col, 0.0)
                )

        # Fill or insert missing categorical confounders
        for col in self.raw_categorical_cols_:
            if col not in aligned_df.columns:
                aligned_df[col] = self.feature_imputation_values_.get(col, "unknown")
            else:
                aligned_df[col] = aligned_df[col].fillna(
                    self.feature_imputation_values_.get(col, "unknown")
                )

        return aligned_df[self.raw_confounder_cols_]

    def predict_elasticity(
        self,
        features: Union[pd.DataFrame, Dict[str, Any]],
    ) -> np.ndarray:
        """Predict heterogeneous causal price elasticity theta(X) for new observations.

        Parameters
        ----------
        features : Union[pd.DataFrame, Dict[str, Any]]
            DataFrame or dict containing item/market features.

        Returns
        -------
        np.ndarray
            Array of predicted causal price elasticity coefficients theta(X).
        """
        if not self.is_fitted_:
            raise ValueError("Pipeline is not fitted yet. Call 'fit' before predicting.")

        if isinstance(features, dict):
            df = pd.DataFrame([features])
        else:
            df = features

        aligned_df = self._align_input_dataframe(df)
        X_trans = self.preprocessor.transform(aligned_df)
        return self.cate_estimator.predict_cate(X_trans)

    def predict_item(
        self,
        item: Dict[str, Any],
        objective: str = "profit",
    ) -> Dict[str, Any]:
        """Real-time on-demand pricing recommendation and uplift simulation for a single SKU.

        Parameters
        ----------
        item : Dict[str, Any]
            Dictionary containing item features. Must include:
            - "unit_price": baseline current selling price
            Optional but recommended:
            - "qty": baseline expected sales volume (defaults to 10.0)
            - "unit_cost": marginal unit cost (defaults to unit_price * (1 - default_margin))
            - "comp_price" or "comp_1": competitor benchmark price
            - "product_category_name": category string for CATE moderation
        objective : str, default="profit"
            Optimization objective: "profit" or "revenue".

        Returns
        -------
        Dict[str, Any]
            Comprehensive structured response with optimal price, action, and uplifts.
        """
        if not self.is_fitted_:
            raise ValueError("Pipeline is not fitted yet. Call 'fit' before predicting.")

        if self.treatment_col not in item:
            raise ValueError(f"Missing required price treatment column '{self.treatment_col}' in input item.")

        base_price = float(item[self.treatment_col])
        if base_price <= 0:
            raise ValueError(f"Base price must be strictly positive, got {base_price}")

        base_qty = float(item.get(self.outcome_col, 10.0))
        unit_cost = float(item["unit_cost"]) if "unit_cost" in item else None

        # Detect competitor price
        comp_price = None
        for comp_key in ["comp_price", "comp_1", "comp_2", "comp_3"]:
            if comp_key in item and item[comp_key] is not None:
                comp_price = float(item[comp_key])
                break

        # 1. Predict Heterogeneous Price Elasticity
        elasticity = float(self.predict_elasticity(item)[0])

        # 2. Calculate Constrained Optimal Price
        optimal_price, bound_hit = self.optimizer.calculate_optimal_price(
            base_price=base_price,
            elasticity=elasticity,
            unit_cost=unit_cost,
            objective=objective,
            comp_price=comp_price,
        )

        # 3. Simulate Uplift
        uplift_res: ItemOptimizationResult = self.optimizer.simulate_item_uplift(
            base_price=base_price,
            base_qty=base_qty,
            new_price=optimal_price,
            elasticity=elasticity,
            unit_cost=unit_cost,
            objective=objective,
            bound_hit=bound_hit,
        )

        # 4. Classify Pricing Action
        price_delta_pct = (optimal_price - base_price) / base_price
        if price_delta_pct > 0.005:
            action = "MARKUP"
        elif price_delta_pct < -0.005:
            action = "DISCOUNT"
        else:
            action = "HOLD"

        return {
            "status": "SUCCESS",
            "action": action,
            "baseline_price": round(base_price, 2),
            "optimal_price": round(optimal_price, 2),
            "price_change_pct": round(price_delta_pct * 100.0, 2),
            "causal_elasticity": round(elasticity, 4),
            "unit_cost": round(uplift_res.unit_cost, 2),
            "baseline_margin_pct": round(uplift_res.baseline_margin * 100.0, 2),
            "projected_margin_pct": round(uplift_res.projected_margin * 100.0, 2),
            "baseline_qty": round(base_qty, 1),
            "projected_qty": round(uplift_res.projected_qty, 1),
            "demand_uplift_pct": round(uplift_res.demand_uplift_pct, 2),
            "baseline_revenue": round(uplift_res.baseline_revenue, 2),
            "projected_revenue": round(uplift_res.projected_revenue, 2),
            "revenue_uplift_pct": round(uplift_res.revenue_uplift_pct, 2),
            "baseline_profit": round(uplift_res.baseline_profit, 2),
            "projected_profit": round(uplift_res.projected_profit, 2),
            "profit_uplift_pct": round(uplift_res.profit_uplift_pct, 2),
            "bound_hit": bound_hit,
            "objective": objective,
        }

    def batch_recommend(
        self,
        catalog_df: pd.DataFrame,
        objective: str = "profit",
        export_path: Optional[str] = None,
    ) -> pd.DataFrame:
        """Generate optimal dynamic price recommendations for an entire product catalog.

        Parameters
        ----------
        catalog_df : pd.DataFrame
            Catalog or transaction DataFrame.
        objective : str, default="profit"
            Optimization objective: "profit" or "revenue".
        export_path : Optional[str], default=None
            If provided, saves recommendation sheet to CSV or JSON.

        Returns
        -------
        pd.DataFrame
            Enriched recommendations sheet with baseline vs optimal prices,
            elasticities, uplifts, and actions.
        """
        if not self.is_fitted_:
            raise ValueError("Pipeline is not fitted yet. Call 'fit' before predicting.")

        df_work = catalog_df.copy()

        # Predict elasticities across the catalog
        elasticities = self.predict_elasticity(df_work)
        df_work["elasticity"] = elasticities

        # Run catalog optimizer
        comp_col = "comp_1" if "comp_1" in df_work.columns else None
        cost_col = "unit_cost" if "unit_cost" in df_work.columns else None

        optimized_df = self.optimizer.optimize_catalog(
            df=df_work,
            price_col=self.treatment_col,
            qty_col=self.outcome_col,
            elasticity_col="elasticity",
            cost_col=cost_col,
            comp_price_col=comp_col,
            objective=objective,
        )

        # Ensure baseline price and baseline quantity columns are present
        if "baseline_price" not in optimized_df.columns and self.treatment_col in optimized_df.columns:
            optimized_df["baseline_price"] = optimized_df[self.treatment_col]
        if "baseline_qty" not in optimized_df.columns and self.outcome_col in optimized_df.columns:
            optimized_df["baseline_qty"] = optimized_df[self.outcome_col]

        # Assemble clean recommendation report DataFrame
        summary_cols = []
        for id_col in ["product_id", "product_category_name"]:
            if id_col in optimized_df.columns:
                summary_cols.append(id_col)

        metric_cols = [
            "baseline_price",
            "optimal_price",
            "price_change_pct",
            "elasticity",
            "baseline_qty",
            "projected_qty",
            "demand_uplift_pct",
            "baseline_revenue",
            "projected_revenue",
            "revenue_uplift_pct",
            "baseline_profit",
            "projected_profit",
            "profit_uplift_pct",
            "baseline_margin",
            "projected_margin",
            "bound_hit",
        ]
        available_metrics = [c for c in metric_cols if c in optimized_df.columns]
        result_df = optimized_df[summary_cols + available_metrics].copy()

        # Add explicit action tag
        actions = []
        for delta in result_df["price_change_pct"]:
            if delta > 0.005:
                actions.append("MARKUP")
            elif delta < -0.005:
                actions.append("DISCOUNT")
            else:
                actions.append("HOLD")
        result_df.insert(len(summary_cols) + 3, "pricing_action", actions)

        # Optional Export
        if export_path is not None:
            os.makedirs(os.path.dirname(os.path.abspath(export_path)), exist_ok=True)
            if export_path.endswith(".json"):
                result_df.to_json(export_path, orient="records", indent=2)
            else:
                result_df.to_csv(export_path, index=False)

        return result_df

    def simulate_scenarios(
        self,
        catalog_df: pd.DataFrame,
        comp_col: Optional[str] = "comp_1",
    ) -> pd.DataFrame:
        """Simulate and compare multiple pricing strategies across the catalog.

        Strategies:
        1. Status Quo
        2. Flat 10% Discount
        3. Flat 10% Markup
        4. Match Key Competitor
        5. Causal Revenue Maximizer
        6. Causal Profit Maximizer

        Parameters
        ----------
        catalog_df : pd.DataFrame
            Catalog DataFrame.
        comp_col : Optional[str], default="comp_1"
            Competitor price column name.

        Returns
        -------
        pd.DataFrame
            Scenario comparison matrix.
        """
        if not self.is_fitted_:
            raise ValueError("Pipeline is not fitted yet. Call 'fit' before simulating.")

        df_work = catalog_df.copy()
        if "elasticity" not in df_work.columns:
            df_work["elasticity"] = self.predict_elasticity(df_work)

        valid_comp = comp_col if comp_col in df_work.columns else None
        return self.optimizer.compare_pricing_scenarios(
            df=df_work,
            price_col=self.treatment_col,
            qty_col=self.outcome_col,
            elasticity_col="elasticity",
            comp_col=valid_comp,
        )

    def save(self, filepath: str) -> str:
        """Serialize and persist the fitted pipeline to disk using joblib.

        Parameters
        ----------
        filepath : str
            Destination file path (e.g. 'models/econocausal_pipeline.joblib').

        Returns
        -------
        str
            Absolute path to saved model artifact.
        """
        if not self.is_fitted_:
            raise ValueError("Cannot save an unfitted pipeline. Call 'fit' first.")

        abs_path = os.path.abspath(filepath)
        os.makedirs(os.path.dirname(abs_path), exist_ok=True)
        joblib.dump(self, abs_path, compress=3)
        return abs_path

    @classmethod
    def load(cls, filepath: str) -> "EconoCausalPipeline":
        """Deserialize and load a persisted EconoCausalPipeline from disk.

        Parameters
        ----------
        filepath : str
            Path to serialized .joblib file.

        Returns
        -------
        EconoCausalPipeline
            Fitted pipeline instance.
        """
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"Pipeline model file not found at: {filepath}")

        loaded_obj = joblib.load(filepath)
        if not isinstance(loaded_obj, cls):
            raise TypeError(
                f"Loaded object is of type {type(loaded_obj).__name__}, expected {cls.__name__}"
            )
        return loaded_obj

    def generate_executive_summary(self) -> str:
        """Generate a formatted ASCII executive summary of the trained pipeline."""
        if not self.is_fitted_:
            return "EconoCausalPipeline is currently UNFITTED."

        ci_str = (
            f"[{self.dml_estimator.ci_[0]:.4f}, {self.dml_estimator.ci_[1]:.4f}]"
            if self.dml_estimator and self.dml_estimator.ci_
            else "N/A"
        )
        refutation_str = (
            f"{self.refutation_pass_rate_:.1f}% ({int(self.refutation_pass_rate_ * 3 / 100)}/3 checks passed)"
            if self.refutation_pass_rate_ is not None
            else "Not Executed"
        )

        lines = [
            "=" * 78,
            "           ECONOCAUSAL END-TO-END PIPELINE EXECUTIVE AUDIT",
            "=" * 78,
            f"Training Timestamp:            {self.metadata_.get('trained_at', 'N/A')}",
            f"Training Samples:              {self.metadata_.get('n_samples_train', 'N/A')} (Total: {self.metadata_.get('n_samples_total', 'N/A')})",
            f"Encoded Confounder Dimension:  {self.metadata_.get('n_features_encoded', 'N/A')} features",
            "-" * 78,
            "CAUSAL IDENTIFICATION & DOUBLE MACHINE LEARNING (DML):",
            f"  - Baseline ATE Elasticity:   {self.baseline_ate_:.4f}",
            f"  - Asymptotic Standard Error: {self.baseline_ate_se_:.4f}",
            f"  - p-value:                   {self.baseline_ate_pvalue_:.4e}",
            f"  - 95% Confidence Interval:   {ci_str}",
            "-" * 78,
            "HETEROGENEOUS TREATMENT EFFECTS (CATE):",
            f"  - Mean CATE Elasticity:      {self.metadata_.get('cate_mean_elasticity', 0.0):.4f}",
            f"  - Min (Most Sensitive):      {self.metadata_.get('cate_min_elasticity', 0.0):.4f}",
            f"  - Max (Least Sensitive):     {self.metadata_.get('cate_max_elasticity', 0.0):.4f}",
            "-" * 78,
            "CAUSAL REFUTATION & MODEL AUDIT:",
            f"  - Robustness Score:          {refutation_str}",
            "-" * 78,
            "DYNAMIC PRICING OPTIMIZER GUARDRAILS:",
            f"  - Pricing Bounds:            [{self.min_price_change_pct * 100.0:+.1f}%, {self.max_price_change_pct * 100.0:+.1f}%]",
            f"  - Default Margin:            {self.default_margin * 100.0:.1f}%",
            f"  - Minimum Margin Floor:      {self.min_margin * 100.0:.1f}%",
            f"  - Charm Pricing (.99):       {'Enabled' if self.charm_pricing else 'Disabled'}",
            "=" * 78,
        ]
        return "\n".join(lines)
