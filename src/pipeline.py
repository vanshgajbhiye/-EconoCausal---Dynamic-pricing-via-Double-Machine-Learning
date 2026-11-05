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
import json
import os
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union
import uuid
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
from src.models.artifacts import (
    EvaluationResults,
    FeatureSchema,
    ModelMetadata,
    TrainingDataVersion,
    extract_feature_schema,
    extract_training_data_version,
)
from src.models.cate import CATEEstimator
from src.models.optimizer import DynamicPricingOptimizer, ItemOptimizationResult
from src.models.refutation import CausalRefuter
from src.models.train_dml import DoubleMachineLearning
from src.utils.reproducibility import (
    get_dependency_versions,
    get_git_info,
    get_system_info,
    set_seed,
)


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
        model_version: str = "1.0.0",
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
        self.model_version = model_version

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
        self.model_metadata_: Optional[ModelMetadata] = None
        self.feature_schema_: Optional[FeatureSchema] = None
        self.training_data_version_: Optional[TrainingDataVersion] = None
        self.evaluation_results_: Optional[EvaluationResults] = None
        self.benchmark_results_: Optional[Any] = None

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
        set_seed(self.random_state)

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

        # 3. Anti-leakage Causal Splitting & Preprocessing
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

        # 4. Store training medians / modes strictly from training partition for zero-leakage runtime imputation
        df_train = df.loc[X_train.index]
        for col in self.raw_numerical_cols_:
            self.feature_imputation_values_[col] = float(df_train[col].median(skipna=True))
        for col in self.raw_categorical_cols_:
            mode_val = df_train[col].mode(dropna=True)
            self.feature_imputation_values_[col] = str(mode_val.iloc[0]) if len(mode_val) > 0 else "unknown"

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

        # 9. Build Structured Metadata, FeatureSchema, Data Provenance, and Evaluation Results
        self.feature_schema_ = extract_feature_schema(
            df=df_train,
            treatment_col=self.treatment_col,
            outcome_col=self.outcome_col,
            raw_confounder_cols=self.raw_confounder_cols_,
            raw_numerical_cols=self.raw_numerical_cols_,
            raw_categorical_cols=self.raw_categorical_cols_,
            encoded_feature_names=self.training_feature_names_,
            imputation_values=self.feature_imputation_values_,
        )

        self.training_data_version_ = extract_training_data_version(
            df=df,
            train_idx=X_train.index.tolist(),
            test_idx=splits["X_test"].index.tolist(),
            test_size=test_size,
            random_state=self.random_state,
        )

        ci_lower = (
            self.baseline_ate_ - 1.96 * self.baseline_ate_se_
            if (self.baseline_ate_ is not None and self.baseline_ate_se_ is not None)
            else 0.0
        )
        ci_upper = (
            self.baseline_ate_ + 1.96 * self.baseline_ate_se_
            if (self.baseline_ate_ is not None and self.baseline_ate_se_ is not None)
            else 0.0
        )

        cate_preds = self.cate_estimator.cate_predictions_
        cate_summary = {
            "mean": float(np.mean(cate_preds)),
            "std": float(np.std(cate_preds)),
            "min": float(np.min(cate_preds)),
            "max": float(np.max(cate_preds)),
            "median": float(np.median(cate_preds)),
            "p25": float(np.percentile(cate_preds, 25)),
            "p75": float(np.percentile(cate_preds, 75)),
            "pct_elastic": float(np.mean(np.abs(cate_preds) > 1.0) * 100.0),
        }

        nuisance_scores = getattr(self.dml_estimator, "nuisance_scores_", {})

        self.evaluation_results_ = EvaluationResults(
            baseline_ate=float(self.baseline_ate_ if self.baseline_ate_ is not None else 0.0),
            baseline_ate_se=float(self.baseline_ate_se_ or 0.0),
            baseline_ate_pvalue=float(self.baseline_ate_pvalue_ or 0.0),
            baseline_ate_ci=(ci_lower, ci_upper),
            cate_summary=cate_summary,
            nuisance_scores=nuisance_scores,
            refutation_pass_rate=self.refutation_pass_rate_,
            refutation_results=self.refutation_results_,
            recorded_at=end_time.isoformat(),
        )

        model_id = f"ec-{uuid.uuid4().hex[:12]}"
        self.model_metadata_ = ModelMetadata(
            model_name="EconoCausalPipeline",
            model_version=self.model_version,
            model_id=model_id,
            created_at=end_time.isoformat(),
            training_duration_sec=(end_time - start_time).total_seconds(),
            random_state=self.random_state,
            algorithm="PartiallyLinearDML_CATE_AmorosoRobinson",
            hyperparameters={
                "n_splits": self.n_splits,
                "cate_alpha": self.cate_alpha,
                "default_margin": self.default_margin,
                "min_price_change_pct": self.min_price_change_pct,
                "max_price_change_pct": self.max_price_change_pct,
                "min_margin": self.min_margin,
                "charm_pricing": self.charm_pricing,
                "log_transform_treatment": self.log_transform_treatment,
                "log_transform_outcome": self.log_transform_outcome,
            },
            dependencies=get_dependency_versions(),
            system_info=get_system_info(),
            git_info=get_git_info(),
        )

        # Retain and enrich self.metadata_ for 100% backward compatibility
        self.metadata_ = {
            "model_version": self.model_version,
            "model_id": model_id,
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
            "cate_mean_elasticity": cate_summary["mean"],
            "cate_min_elasticity": cate_summary["min"],
            "cate_max_elasticity": cate_summary["max"],
            "training_data_sha256": self.training_data_version_.data_sha256,
            "dependencies": self.model_metadata_.dependencies,
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
            "unit_cost",
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
            "comp_1",
            "comp_2",
            "comp_3",
            "comp_price",
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
        result_df["action"] = actions

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

    # Backward and interface compatibility aliases
    recommend_prices = batch_recommend
    simulate_portfolio = simulate_scenarios

    def save(self, filepath: str, save_companion_metadata: bool = True) -> str:
        """Serialize and persist the fitted pipeline to disk using joblib.

        Parameters
        ----------
        filepath : str
            Destination file path (e.g. 'models/econocausal_pipeline.joblib').
        save_companion_metadata : bool, default=True
            Whether to write companion JSON files (metadata.json, feature_schema.json,
            training_data_version.json, evaluation_results.json) in the destination directory.

        Returns
        -------
        str
            Absolute path to saved model artifact.
        """
        if not self.is_fitted_:
            raise ValueError("Cannot save an unfitted pipeline. Call 'fit' first.")

        abs_path = os.path.abspath(filepath)
        os.makedirs(os.path.dirname(abs_path), exist_ok=True)
        if os.path.exists(abs_path):
            try:
                os.remove(abs_path)
            except OSError:
                pass
        joblib.dump(self, abs_path, compress=3)

        if save_companion_metadata:
            base_dir = os.path.dirname(abs_path)
            if self.model_metadata_:
                with open(os.path.join(base_dir, "metadata.json"), "w", encoding="utf-8") as f:
                    json.dump(self.model_metadata_.to_dict(), f, indent=2)
            if self.feature_schema_:
                with open(os.path.join(base_dir, "feature_schema.json"), "w", encoding="utf-8") as f:
                    json.dump(self.feature_schema_.to_dict(), f, indent=2)
            if self.training_data_version_:
                with open(os.path.join(base_dir, "training_data_version.json"), "w", encoding="utf-8") as f:
                    json.dump(self.training_data_version_.to_dict(), f, indent=2)
            if self.evaluation_results_:
                with open(os.path.join(base_dir, "evaluation_results.json"), "w", encoding="utf-8") as f:
                    json.dump(self.evaluation_results_.to_dict(), f, indent=2)

        return abs_path

    def save_versioned(self, artifact_dir: str, version: Optional[str] = None) -> str:
        """Serialize and persist the model into a versioned artifact bundle with manifest and checksums.

        Parameters
        ----------
        artifact_dir : str
            Destination directory for bundle.
        version : Optional[str], default=None
            Version string. If None, uses self.model_version.

        Returns
        -------
        str
            Absolute path to the created artifact directory.
        """
        if not self.is_fitted_:
            raise ValueError("Cannot save an unfitted pipeline. Call 'fit' first.")
        from src.models.artifacts import save_versioned_bundle

        return save_versioned_bundle(self, artifact_dir, version=version or self.model_version)

    @classmethod
    def load_versioned(
        cls, artifact_dir: str, verify_checksums: bool = True, strict_compatibility: bool = False
    ) -> "EconoCausalPipeline":
        """Deserialize and validate a versioned model bundle from disk.

        Parameters
        ----------
        artifact_dir : str
            Path to versioned artifact bundle directory.
        verify_checksums : bool, default=True
            Whether to verify cryptographic SHA-256 hashes against manifest.json.
        strict_compatibility : bool, default=False
            Whether to enforce strict dependency compatibility.

        Returns
        -------
        EconoCausalPipeline
            Restored pipeline instance.
        """
        from src.models.artifacts import load_versioned_bundle

        return load_versioned_bundle(
            artifact_dir,
            verify_checksums=verify_checksums,
            strict_compatibility=strict_compatibility,
        )

    @classmethod
    def load(cls, filepath: str) -> "EconoCausalPipeline":
        """Deserialize and load a persisted EconoCausalPipeline from disk.

        Supports both direct .joblib file paths and versioned bundle directories.

        Parameters
        ----------
        filepath : str
            Path to serialized .joblib file or bundle directory.

        Returns
        -------
        EconoCausalPipeline
            Fitted pipeline instance.
        """
        abs_path = os.path.abspath(filepath)
        if os.path.isdir(abs_path):
            return cls.load_versioned(abs_path)

        if not os.path.exists(abs_path):
            raise FileNotFoundError(f"Pipeline model file not found at: {filepath}")

        loaded_obj = joblib.load(abs_path)
        if not isinstance(loaded_obj, cls):
            raise TypeError(
                f"Loaded object is of type {type(loaded_obj).__name__}, expected {cls.__name__}"
            )

        # Attempt to load companion metadata if present
        base_dir = os.path.dirname(abs_path)
        meta_p = os.path.join(base_dir, "metadata.json")
        if os.path.exists(meta_p) and not getattr(loaded_obj, "model_metadata_", None):
            try:
                with open(meta_p, "r", encoding="utf-8") as f:
                    loaded_obj.model_metadata_ = ModelMetadata.from_dict(json.load(f))
            except Exception:
                pass
        schema_p = os.path.join(base_dir, "feature_schema.json")
        if os.path.exists(schema_p) and not getattr(loaded_obj, "feature_schema_", None):
            try:
                with open(schema_p, "r", encoding="utf-8") as f:
                    loaded_obj.feature_schema_ = FeatureSchema.from_dict(json.load(f))
            except Exception:
                pass
        data_ver_p = os.path.join(base_dir, "training_data_version.json")
        if os.path.exists(data_ver_p) and not getattr(loaded_obj, "training_data_version_", None):
            try:
                with open(data_ver_p, "r", encoding="utf-8") as f:
                    loaded_obj.training_data_version_ = TrainingDataVersion.from_dict(json.load(f))
            except Exception:
                pass
        eval_p = os.path.join(base_dir, "evaluation_results.json")
        if os.path.exists(eval_p) and not getattr(loaded_obj, "evaluation_results_", None):
            try:
                with open(eval_p, "r", encoding="utf-8") as f:
                    loaded_obj.evaluation_results_ = EvaluationResults.from_dict(json.load(f))
            except Exception:
                pass

        return loaded_obj

    def benchmark_inference(
        self,
        df: pd.DataFrame,
        batch_sizes: Sequence[int] = (1, 10, 50, 100),
        n_single_runs: int = 100,
        measure_memory: bool = True,
    ) -> Any:
        """Measure inference latency and memory use on representative batches.

        Parameters
        ----------
        df : pd.DataFrame
            Catalog or transaction dataset for batch benchmarking.
        batch_sizes : Sequence[int], default=(1, 10, 50, 100)
            Representative batch sizes.
        n_single_runs : int, default=100
            Number of single-SKU inference iterations.
        measure_memory : bool, default=True
            Whether to track memory peak and delta with tracemalloc.

        Returns
        -------
        ModelBenchmarkSummary
            Detailed performance benchmark results.
        """
        from src.models.benchmark import run_inference_benchmarks

        summary = run_inference_benchmarks(
            pipeline=self,
            df=df,
            batch_sizes=batch_sizes,
            n_single_runs=n_single_runs,
            measure_memory=measure_memory,
        )
        self.benchmark_results_ = summary
        return summary

    def check_version_compatibility(self) -> Dict[str, Any]:
        """Verify current execution environment against model recorded dependencies."""
        from src.models.artifacts import check_version_compatibility

        meta = self.model_metadata_ if self.model_metadata_ is not None else self.metadata_
        return check_version_compatibility(meta)

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
