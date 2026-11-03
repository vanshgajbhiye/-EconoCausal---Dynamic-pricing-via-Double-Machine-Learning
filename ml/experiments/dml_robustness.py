"""Double Machine Learning (DML) Robustness and Cross-Fitting Evaluation Suite.

Provides systematic diagnostic benchmarks and stress tests for Partially Linear DML
price elasticity estimation:
1. Nuisance Model Comparison: Linear vs Tree-based Ensembles vs Hybrid pairings.
2. Cross-Fitting Sensitivity: Sensitivity across K folds (2 to 10) and multiple random seeds,
   with multi-split median aggregation and split-adjusted standard errors (Chernozhukov et al., 2018).
3. Uncertainty Intervals: Asymptotic normal CIs, multi-split pooled CIs, and percentile bootstrap CIs.
4. Stress Testing:
   - Small-sample behavior (SE inflation, finite-sample stability).
   - Missing-value handling (median, mean, missing indicator, drop strategies under MCAR).
   - Weak treatment variation / weak identification (effective variation ratio, residual variance collapse).
"""

import argparse
from dataclasses import asdict, dataclass, field
import json
import os
import sys
import time
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union
import warnings

# Configure environment and silence loky core-count warning in constrained environments
os.environ.setdefault("LOKY_MAX_CPU_COUNT", str(os.cpu_count() or 4))
warnings.filterwarnings("ignore", category=UserWarning, module="joblib")

# Ensure repository root is on sys.path for direct script execution
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from scipy import stats  # noqa: E402
from sklearn.base import clone  # noqa: E402
from sklearn.ensemble import (  # noqa: E402
    ExtraTreesRegressor,
    GradientBoostingRegressor,
    HistGradientBoostingRegressor,
    RandomForestRegressor,
)
from sklearn.linear_model import ElasticNet, Lasso, Ridge  # noqa: E402

from src.data.load_data import (  # noqa: E402
    DEFAULT_OUTCOME_COL,
    DEFAULT_TREATMENT_COL,
    load_retail_price_dataset,
)
from src.data.preprocess import split_causal_data  # noqa: E402
from src.models.train_dml import DoubleMachineLearning  # noqa: E402


# ==============================================================================
# 1. DATA MODELS AND CONTAINERS
# ==============================================================================


@dataclass
class RobustnessResult:
    """Structured record for a single DML estimation run."""

    experiment: str
    nuisance_name: str
    n_folds: int
    random_state: int
    sample_size: int
    theta: float
    std_error: float
    t_stat: float
    p_value: float
    ci_lower: float
    ci_upper: float
    ci_width: float
    ci_level: float
    r2_y: float
    rmse_y: float
    mae_y: float
    r2_t: float
    rmse_t: float
    mae_t: float
    orthogonality_score: float
    orthogonality_corr: float
    var_residual_t: float
    fit_time_seconds: float
    extra_info: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Convert record to flat dictionary."""
        d = asdict(self)
        extra = d.pop("extra_info", {})
        for k, v in extra.items():
            d[f"extra_{k}"] = v
        return d


class RobustnessReport:
    """Collection and analysis container for DML robustness experiment results."""

    def __init__(self, records: Optional[List[RobustnessResult]] = None):
        self.records: List[RobustnessResult] = records or []

    def add(self, result: RobustnessResult) -> None:
        """Add a single experiment run result."""
        self.records.append(result)

    def extend(self, results: Sequence[RobustnessResult]) -> None:
        """Add multiple experiment run results."""
        self.records.extend(results)

    def to_dataframe(self) -> pd.DataFrame:
        """Export all records to a pandas DataFrame."""
        if not self.records:
            return pd.DataFrame()
        return pd.DataFrame([r.to_dict() for r in self.records])

    def filter_by_experiment(self, experiment_name: str) -> "RobustnessReport":
        """Return a sub-report filtered by experiment identifier."""
        matched = [r for r in self.records if r.experiment == experiment_name]
        return RobustnessReport(matched)

    def get_summary_table(self, experiment: Optional[str] = None) -> str:
        """Generate formatted ASCII table of estimation results."""
        df = self.to_dataframe()
        if df.empty:
            return "No robustness records found."

        if experiment:
            df = df[df["experiment"] == experiment]
            if df.empty:
                return f"No records found for experiment '{experiment}'."

        cols = [
            "experiment",
            "nuisance_name",
            "n_folds",
            "sample_size",
            "theta",
            "std_error",
            "ci_lower",
            "ci_upper",
            "r2_y",
            "r2_t",
        ]
        available_cols = [c for c in cols if c in df.columns]
        display_df = df[available_cols].copy()

        # Format numeric floats
        format_spec = {
            "theta": "{:.4f}".format,
            "std_error": "{:.4f}".format,
            "ci_lower": "{:.4f}".format,
            "ci_upper": "{:.4f}".format,
            "r2_y": "{:.3f}".format,
            "r2_t": "{:.3f}".format,
        }
        for col, fmt in format_spec.items():
            if col in display_df.columns:
                display_df[col] = display_df[col].apply(lambda x: fmt(x) if pd.notnull(x) else "N/A")

        return display_df.to_string(index=False)

    def save_csv(self, filepath: str) -> str:
        """Save results to CSV file, ensuring parent directory exists."""
        os.makedirs(os.path.dirname(os.path.abspath(filepath)), exist_ok=True)
        df = self.to_dataframe()
        df.to_csv(filepath, index=False)
        return filepath

    def save_json(self, filepath: str) -> str:
        """Save results to structured JSON file."""
        os.makedirs(os.path.dirname(os.path.abspath(filepath)), exist_ok=True)
        data = [r.to_dict() for r in self.records]
        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2)
        return filepath


# ==============================================================================
# 2. NUISANCE MODEL REGISTRY & FACTORY
# ==============================================================================


def get_nuisance_model_candidates(
    random_state: int = 42,
) -> Dict[str, Tuple[Any, Any]]:
    """Return dictionary of reasonable nuisance model pairs (model_y, model_t).

    Pairs include:
    - Linear regularized: Ridge, Lasso, ElasticNet
    - Tree ensembles: RandomForest, GradientBoosting, ExtraTrees, HistGradientBoosting
    - Asymmetric Hybrids: RandomForest Y + Ridge T, GradientBoosting Y + Ridge T
    """
    return {
        "Ridge": (
            Ridge(alpha=1.0, random_state=random_state),
            Ridge(alpha=1.0, random_state=random_state),
        ),
        "Lasso": (
            Lasso(alpha=0.01, random_state=random_state, max_iter=2000),
            Lasso(alpha=0.01, random_state=random_state, max_iter=2000),
        ),
        "ElasticNet": (
            ElasticNet(alpha=0.01, l1_ratio=0.5, random_state=random_state, max_iter=2000),
            ElasticNet(alpha=0.01, l1_ratio=0.5, random_state=random_state, max_iter=2000),
        ),
        "RandomForest": (
            RandomForestRegressor(n_estimators=50, max_depth=6, random_state=random_state),
            RandomForestRegressor(n_estimators=50, max_depth=6, random_state=random_state),
        ),
        "GradientBoosting": (
            GradientBoostingRegressor(n_estimators=50, max_depth=3, learning_rate=0.1, random_state=random_state),
            GradientBoostingRegressor(n_estimators=50, max_depth=3, learning_rate=0.1, random_state=random_state),
        ),
        "ExtraTrees": (
            ExtraTreesRegressor(n_estimators=50, max_depth=6, random_state=random_state),
            ExtraTreesRegressor(n_estimators=50, max_depth=6, random_state=random_state),
        ),
        "HistGradientBoosting": (
            HistGradientBoostingRegressor(max_iter=50, max_depth=4, random_state=random_state),
            HistGradientBoostingRegressor(max_iter=50, max_depth=4, random_state=random_state),
        ),
        "Hybrid_RF_Ridge": (
            RandomForestRegressor(n_estimators=50, max_depth=6, random_state=random_state),
            Ridge(alpha=1.0, random_state=random_state),
        ),
        "Hybrid_GB_Ridge": (
            GradientBoostingRegressor(n_estimators=50, max_depth=3, learning_rate=0.1, random_state=random_state),
            Ridge(alpha=1.0, random_state=random_state),
        ),
    }


def fit_and_record_dml(
    X: Union[pd.DataFrame, np.ndarray],
    T: Union[pd.Series, np.ndarray],
    Y: Union[pd.Series, np.ndarray],
    model_y: Optional[Any] = None,
    model_t: Optional[Any] = None,
    n_folds: int = 5,
    random_state: int = 42,
    alpha: float = 0.05,
    experiment_name: str = "dml_run",
    nuisance_name: str = "custom",
    extra_info: Optional[Dict[str, Any]] = None,
) -> Tuple[DoubleMachineLearning, RobustnessResult]:
    """Fit a DoubleMachineLearning estimator and capture full metrics into a RobustnessResult."""
    t0 = time.time()
    dml = DoubleMachineLearning(
        model_y=model_y,
        model_t=model_t,
        n_folds=n_folds,
        random_state=random_state,
        alpha=alpha,
    )
    dml.fit(X, T, Y)
    elapsed = time.time() - t0

    ci = dml.confidence_interval(alpha=alpha)
    ci_lower, ci_upper = ci
    ci_width = ci_upper - ci_lower

    res = RobustnessResult(
        experiment=experiment_name,
        nuisance_name=nuisance_name,
        n_folds=n_folds,
        random_state=random_state,
        sample_size=len(X),
        theta=float(dml.theta_),
        std_error=float(dml.std_error_),
        t_stat=float(dml.t_stat_),
        p_value=float(dml.p_value_),
        ci_lower=float(ci_lower),
        ci_upper=float(ci_upper),
        ci_width=float(ci_width),
        ci_level=1.0 - alpha,
        r2_y=float(dml.nuisance_scores_.get("r2_y", 0.0)),
        rmse_y=float(dml.nuisance_scores_.get("rmse_y", 0.0)),
        mae_y=float(dml.nuisance_scores_.get("mae_y", 0.0)),
        r2_t=float(dml.nuisance_scores_.get("r2_t", 0.0)),
        rmse_t=float(dml.nuisance_scores_.get("rmse_t", 0.0)),
        mae_t=float(dml.nuisance_scores_.get("mae_t", 0.0)),
        orthogonality_score=float(dml.residual_diagnostics_.get("orthogonality_score", 0.0)),
        orthogonality_corr=float(dml.residual_diagnostics_.get("orthogonality_corr", 0.0)),
        var_residual_t=float(dml.residual_diagnostics_.get("var_residual_t", 0.0)),
        fit_time_seconds=float(elapsed),
        extra_info=extra_info or {},
    )
    return dml, res


# ==============================================================================
# 3. EXPERIMENT 1: COMPARE NUISANCE MODELS
# ==============================================================================


def compare_nuisance_models(
    X: Union[pd.DataFrame, np.ndarray],
    T: Union[pd.Series, np.ndarray],
    Y: Union[pd.Series, np.ndarray],
    candidates: Optional[Dict[str, Tuple[Any, Any]]] = None,
    n_folds: int = 5,
    random_state: int = 42,
    alpha: float = 0.05,
) -> RobustnessReport:
    """Compare price effect estimates and out-of-fold nuisance model metrics across estimators.

    Parameters
    ----------
    X, T, Y : matrices / vectors
    candidates : Dict of model name -> (model_y, model_t). Defaults to standard suite.
    n_folds : Cross-fitting folds.
    random_state : Random seed for reproducibility.
    alpha : Significance level for CI.

    Returns
    -------
    RobustnessReport
        Report containing one RobustnessResult per candidate model.
    """
    if candidates is None:
        candidates = get_nuisance_model_candidates(random_state=random_state)

    report = RobustnessReport()
    for name, (my, mt) in candidates.items():
        _, res = fit_and_record_dml(
            X=X,
            T=T,
            Y=Y,
            model_y=my,
            model_t=mt,
            n_folds=n_folds,
            random_state=random_state,
            alpha=alpha,
            experiment_name="nuisance_comparison",
            nuisance_name=name,
        )
        report.add(res)

    return report


# ==============================================================================
# 4. EXPERIMENT 2: SENSITIVITY TO CROSS-FITTING FOLDS AND RANDOM SEEDS
# ==============================================================================


@dataclass
class MultiSplitAggregation:
    """Multi-split median aggregation results across random seeds (Chernozhukov et al., 2018)."""

    n_folds: int
    n_splits: int
    theta_median: float
    theta_mean: float
    theta_std: float
    theta_min: float
    theta_max: float
    std_error_mean: float
    std_error_pooled: float
    ci_lower_pooled: float
    ci_upper_pooled: float
    ci_width_pooled: float

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


def evaluate_cross_fitting_and_seeds(
    X: Union[pd.DataFrame, np.ndarray],
    T: Union[pd.Series, np.ndarray],
    Y: Union[pd.Series, np.ndarray],
    folds_list: Sequence[int] = (2, 3, 5, 8, 10),
    seeds_list: Sequence[int] = (42, 101, 202, 303, 404),
    model_y: Optional[Any] = None,
    model_t: Optional[Any] = None,
    nuisance_name: str = "default_rf",
    alpha: float = 0.05,
) -> Tuple[RobustnessReport, Dict[int, MultiSplitAggregation]]:
    """Evaluate DML sensitivity to the number of cross-fitting folds (K) and random seeds.

    Applies Chernozhukov et al. (2018) multi-split aggregation:
        theta_pooled = median(theta_s)
        sigma_pooled^2 = median(sigma_s^2 + (theta_s - theta_pooled)^2)
        SE_pooled = sqrt(sigma_pooled^2)

    Parameters
    ----------
    X, T, Y : causal data
    folds_list : sequence of K values (e.g. [2, 3, 5, 8, 10])
    seeds_list : sequence of random seeds
    model_y, model_t : nuisance estimators
    nuisance_name : label for nuisance models
    alpha : significance level

    Returns
    -------
    Tuple[RobustnessReport, Dict[int, MultiSplitAggregation]]
        Individual run records and fold-level multi-split aggregations.
    """
    report = RobustnessReport()
    aggregations: Dict[int, MultiSplitAggregation] = {}

    for k in folds_list:
        fold_results: List[RobustnessResult] = []
        thetas: List[float] = []
        std_errors: List[float] = []

        for seed in seeds_list:
            # Clone fresh model instances to avoid state leakage
            my = clone(model_y) if model_y is not None else None
            mt = clone(model_t) if model_t is not None else None

            _, res = fit_and_record_dml(
                X=X,
                T=T,
                Y=Y,
                model_y=my,
                model_t=mt,
                n_folds=k,
                random_state=seed,
                alpha=alpha,
                experiment_name="cross_fitting_sensitivity",
                nuisance_name=nuisance_name,
                extra_info={"seed": seed, "folds": k},
            )
            report.add(res)
            fold_results.append(res)
            thetas.append(res.theta)
            std_errors.append(res.std_error)

        theta_arr = np.array(thetas)
        se_arr = np.array(std_errors)
        theta_med = float(np.median(theta_arr))
        theta_mean = float(np.mean(theta_arr))
        theta_std = float(np.std(theta_arr, ddof=1)) if len(theta_arr) > 1 else 0.0

        # Multi-split pooled variance (Chernozhukov et al. 2018 equation 3.12)
        # Var_pooled = median(Var_s + (theta_s - theta_med)^2)
        var_splits = se_arr**2 + (theta_arr - theta_med) ** 2
        var_pooled = float(np.median(var_splits))
        se_pooled = float(np.sqrt(max(var_pooled, 1e-12)))

        z = float(stats.norm.ppf(1.0 - alpha / 2.0))
        ci_lower_pooled = theta_med - z * se_pooled
        ci_upper_pooled = theta_med + z * se_pooled

        agg = MultiSplitAggregation(
            n_folds=k,
            n_splits=len(seeds_list),
            theta_median=theta_med,
            theta_mean=theta_mean,
            theta_std=theta_std,
            theta_min=float(np.min(theta_arr)),
            theta_max=float(np.max(theta_arr)),
            std_error_mean=float(np.mean(se_arr)),
            std_error_pooled=se_pooled,
            ci_lower_pooled=float(ci_lower_pooled),
            ci_upper_pooled=float(ci_upper_pooled),
            ci_width_pooled=float(ci_upper_pooled - ci_lower_pooled),
        )
        aggregations[k] = agg

    return report, aggregations


# ==============================================================================
# 5. UNCERTAINTY INTERVALS: BOOTSTRAP RESAMPLING
# ==============================================================================


def compute_bootstrap_ci(
    X: Union[pd.DataFrame, np.ndarray],
    T: Union[pd.Series, np.ndarray],
    Y: Union[pd.Series, np.ndarray],
    n_bootstraps: int = 50,
    alpha: float = 0.05,
    n_folds: int = 3,
    model_y: Optional[Any] = None,
    model_t: Optional[Any] = None,
    random_state: int = 42,
) -> Dict[str, Any]:
    """Compute non-parametric percentile and standard-error bootstrap confidence intervals.

    Provides non-asymptotic uncertainty validation to benchmark against
    sandwich variance confidence intervals.
    """
    X_arr = np.asarray(X, dtype=np.float64)
    T_arr = np.asarray(T, dtype=np.float64).ravel()
    Y_arr = np.asarray(Y, dtype=np.float64).ravel()
    n_samples = len(X_arr)

    rng = np.random.RandomState(random_state)
    boot_thetas: List[float] = []

    for b in range(n_bootstraps):
        boot_idx = rng.choice(n_samples, size=n_samples, replace=True)
        X_b, T_b, Y_b = X_arr[boot_idx], T_arr[boot_idx], Y_arr[boot_idx]

        try:
            dml_b = DoubleMachineLearning(
                model_y=clone(model_y) if model_y is not None else None,
                model_t=clone(model_t) if model_t is not None else None,
                n_folds=n_folds,
                random_state=rng.randint(0, 100000),
            )
            dml_b.fit(X_b, T_b, Y_b)
            if not np.isnan(dml_b.theta_) and not np.isinf(dml_b.theta_):
                boot_thetas.append(dml_b.theta_)
        except Exception:
            continue

    if len(boot_thetas) < 5:
        raise RuntimeError("Too few valid bootstrap iterations completed.")

    boot_arr = np.array(boot_thetas)
    theta_boot_mean = float(np.mean(boot_arr))
    theta_boot_std = float(np.std(boot_arr, ddof=1))

    # Empirical percentile CI
    lower_pct = 100.0 * (alpha / 2.0)
    upper_pct = 100.0 * (1.0 - alpha / 2.0)
    ci_percentile = (
        float(np.percentile(boot_arr, lower_pct)),
        float(np.percentile(boot_arr, upper_pct)),
    )

    # Normal approximation bootstrap CI
    z = float(stats.norm.ppf(1.0 - alpha / 2.0))
    ci_normal_approx = (
        float(theta_boot_mean - z * theta_boot_std),
        float(theta_boot_mean + z * theta_boot_std),
    )

    return {
        "n_bootstraps_requested": n_bootstraps,
        "n_bootstraps_successful": len(boot_thetas),
        "theta_boot_mean": theta_boot_mean,
        "theta_boot_std": theta_boot_std,
        "ci_percentile": ci_percentile,
        "ci_percentile_lower": ci_percentile[0],
        "ci_percentile_upper": ci_percentile[1],
        "ci_percentile_width": ci_percentile[1] - ci_percentile[0],
        "ci_normal_approx": ci_normal_approx,
        "alpha": alpha,
        "confidence_level": 1.0 - alpha,
    }


# ==============================================================================
# 6. EXPERIMENT 3: STRESS TEST WITH SMALL SAMPLES
# ==============================================================================


def evaluate_small_sample_behavior(
    X: Union[pd.DataFrame, np.ndarray],
    T: Union[pd.Series, np.ndarray],
    Y: Union[pd.Series, np.ndarray],
    sample_sizes: Sequence[int] = (30, 50, 100, 200, 400),
    n_repeats: int = 5,
    model_y: Optional[Any] = None,
    model_t: Optional[Any] = None,
    nuisance_name: str = "default_rf",
    random_state: int = 42,
    alpha: float = 0.05,
) -> RobustnessReport:
    """Stress test DML behavior under constrained / small sample regimes.

    Analyzes:
    - Finite-sample standard error inflation (asymptotic consistency vs empirical variance).
    - Confidence interval widening.
    - Parameter estimate stability and sign preservation.
    """
    X_arr = np.asarray(X, dtype=np.float64)
    T_arr = np.asarray(T, dtype=np.float64).ravel()
    Y_arr = np.asarray(Y, dtype=np.float64).ravel()
    n_total = len(X_arr)

    report = RobustnessReport()
    rng = np.random.RandomState(random_state)

    valid_sizes = [s for s in sample_sizes if s <= n_total]
    if n_total not in valid_sizes:
        valid_sizes.append(n_total)

    for sz in valid_sizes:
        repeats = 1 if sz == n_total else n_repeats
        for rep in range(repeats):
            if sz == n_total:
                idx = np.arange(n_total)
            else:
                idx = rng.choice(n_total, size=sz, replace=False)

            X_sub = X_arr[idx]
            T_sub = T_arr[idx]
            Y_sub = Y_arr[idx]

            # Adjust folds dynamically for very small samples
            if sz < 40:
                k = 2
            elif sz < 80:
                k = 3
            else:
                k = 5

            try:
                my = clone(model_y) if model_y is not None else None
                mt = clone(model_t) if model_t is not None else None

                _, res = fit_and_record_dml(
                    X=X_sub,
                    T=T_sub,
                    Y=Y_sub,
                    model_y=my,
                    model_t=mt,
                    n_folds=k,
                    random_state=rng.randint(0, 100000),
                    alpha=alpha,
                    experiment_name="small_sample_stress",
                    nuisance_name=nuisance_name,
                    extra_info={"target_sample_size": sz, "repeat_index": rep},
                )
                report.add(res)
            except Exception:
                # In tiny samples, collinearity or zero residual variance might trigger
                pass

    return report


# ==============================================================================
# 7. EXPERIMENT 4: STRESS TEST WITH MISSING VALUES
# ==============================================================================


def handle_missing_confounders(
    X: Union[pd.DataFrame, np.ndarray],
    strategy: str = "median",
    add_indicators: bool = False,
) -> Tuple[np.ndarray, np.ndarray]:
    """Impute missing values in confounder matrix X and optionally add missingness indicators.

    Parameters
    ----------
    X : Confounder matrix with possible NaNs.
    strategy : 'median', 'mean', or 'drop'.
    add_indicators : If True, appends boolean indicator column for missingness.

    Returns
    -------
    Tuple[np.ndarray, np.ndarray]
        (imputed_X, keep_mask) where keep_mask indicates preserved rows.
    """
    X_arr = np.asarray(X, dtype=np.float64).copy()
    n_samples, n_features = X_arr.shape
    nan_mask = np.isnan(X_arr)

    if strategy == "drop":
        keep_mask = ~np.any(nan_mask, axis=1)
        X_clean = X_arr[keep_mask]
        return X_clean, keep_mask

    keep_mask = np.ones(n_samples, dtype=bool)

    # Impute columns
    for col in range(n_features):
        col_mask = nan_mask[:, col]
        if np.any(col_mask):
            valid_vals = X_arr[~col_mask, col]
            if len(valid_vals) == 0:
                imp_val = 0.0
            elif strategy == "mean":
                imp_val = float(np.mean(valid_vals))
            else:  # median
                imp_val = float(np.median(valid_vals))
            X_arr[col_mask, col] = imp_val

    if add_indicators and np.any(nan_mask):
        indicator_cols = nan_mask.astype(np.float64)
        # Drop columns with zero missingness
        active_indicators = indicator_cols[:, np.any(nan_mask, axis=0)]
        if active_indicators.shape[1] > 0:
            X_arr = np.hstack([X_arr, active_indicators])

    return X_arr, keep_mask


def evaluate_missing_value_behavior(
    X: Union[pd.DataFrame, np.ndarray],
    T: Union[pd.Series, np.ndarray],
    Y: Union[pd.Series, np.ndarray],
    missing_rates: Sequence[float] = (0.0, 0.05, 0.10, 0.20),
    strategies: Sequence[str] = ("median", "mean", "indicator", "drop"),
    model_y: Optional[Any] = None,
    model_t: Optional[Any] = None,
    nuisance_name: str = "default_rf",
    random_state: int = 42,
    alpha: float = 0.05,
) -> RobustnessReport:
    """Stress test DML behavior under missing-completely-at-random (MCAR) confounders.

    Tests multiple handling strategies:
    - Median imputation
    - Mean imputation
    - Missingness indicator imputation (impute median + add missing indicator)
    - Complete-case analysis (drop rows)

    Evaluates effect stability and standard error inflation relative to clean baseline.
    """
    X_orig = np.asarray(X, dtype=np.float64).copy()
    T_orig = np.asarray(T, dtype=np.float64).ravel().copy()
    Y_orig = np.asarray(Y, dtype=np.float64).ravel().copy()

    report = RobustnessReport()
    rng = np.random.RandomState(random_state)

    for rate in missing_rates:
        if rate == 0.0:
            # Clean baseline
            my = clone(model_y) if model_y is not None else None
            mt = clone(model_t) if model_t is not None else None
            _, res = fit_and_record_dml(
                X=X_orig,
                T=T_orig,
                Y=Y_orig,
                model_y=my,
                model_t=mt,
                n_folds=5,
                random_state=random_state,
                alpha=alpha,
                experiment_name="missing_values_stress",
                nuisance_name=nuisance_name,
                extra_info={"missing_rate": 0.0, "strategy": "clean_baseline"},
            )
            report.add(res)
            continue

        # Inject MCAR missingness into 30% of features
        X_corrupted = X_orig.copy()
        n_samples, n_feats = X_corrupted.shape
        feats_to_corrupt = rng.choice(n_feats, size=max(1, int(n_feats * 0.4)), replace=False)

        for col in feats_to_corrupt:
            mask = rng.rand(n_samples) < rate
            X_corrupted[mask, col] = np.nan

        for strat in strategies:
            add_ind = (strat == "indicator")
            actual_strat = "median" if strat == "indicator" else strat

            X_imputed, keep_mask = handle_missing_confounders(
                X_corrupted,
                strategy=actual_strat,
                add_indicators=add_ind,
            )
            T_sub = T_orig[keep_mask]
            Y_sub = Y_orig[keep_mask]

            if len(T_sub) < 20:
                continue

            try:
                my = clone(model_y) if model_y is not None else None
                mt = clone(model_t) if model_t is not None else None
                _, res = fit_and_record_dml(
                    X=X_imputed,
                    T=T_sub,
                    Y=Y_sub,
                    model_y=my,
                    model_t=mt,
                    n_folds=5 if len(T_sub) >= 50 else 3,
                    random_state=random_state,
                    alpha=alpha,
                    experiment_name="missing_values_stress",
                    nuisance_name=nuisance_name,
                    extra_info={"missing_rate": rate, "strategy": strat},
                )
                report.add(res)
            except Exception:
                pass

    return report


# ==============================================================================
# 8. EXPERIMENT 5: STRESS TEST WITH WEAK TREATMENT VARIATION
# ==============================================================================


def diagnose_treatment_variation(
    T: Union[pd.Series, np.ndarray],
    pred_t: Union[pd.Series, np.ndarray],
    threshold_evr: float = 0.05,
    min_residual_var: float = 1e-4,
) -> Dict[str, Any]:
    """Diagnose treatment variation strength and identify weak treatment risks.

    In Double Machine Learning, the second-stage estimator variance satisfies:
        Var(theta_hat) ~ 1 / [N * Var(T_tilde)]
    where T_tilde = T - E[T|X].
    Weak treatment variation occurs if:
    1. Treatment has near-zero raw variance.
    2. Confounders explain almost 100% of treatment assignment (R2_T -> 1.0, EVR -> 0).

    Returns
    -------
    Dict[str, Any]
        Health diagnosis with status, effective variation ratio (EVR), and risk tiers.
    """
    T_arr = np.asarray(T, dtype=np.float64).ravel()
    pred_arr = np.asarray(pred_t, dtype=np.float64).ravel()
    res_t = T_arr - pred_arr

    var_total = float(np.var(T_arr))
    var_res = float(np.var(res_t))
    evr = var_res / (var_total + 1e-12)  # Effective Variation Ratio (1 - R2_T)
    norm_res = float(np.linalg.norm(res_t))

    if var_res < min_residual_var or evr < threshold_evr:
        status = "WEAK_IDENTIFICATION"
        severity = "CRITICAL"
        warning = (
            f"Treatment residual variance ({var_res:.2e}) or Effective Variation Ratio ({evr:.2%}) "
            "is dangerously low. Confounders nearly collinearly explain treatment. "
            "Causal effect estimates will suffer massive variance inflation."
        )
    elif evr < threshold_evr * 2:
        status = "MODERATE_RISK"
        severity = "WARNING"
        warning = (
            f"Effective Variation Ratio ({evr:.2%}) indicates modest remaining treatment variation. "
            "Consider pruning highly collinear confounders."
        )
    else:
        status = "HEALTHY"
        severity = "INFO"
        warning = None

    return {
        "status": status,
        "severity": severity,
        "effective_variation_ratio": float(evr),
        "total_treatment_variance": float(var_total),
        "residual_treatment_variance": float(var_res),
        "residual_l2_norm": float(norm_res),
        "warning_message": warning,
    }


def evaluate_weak_treatment_variation(
    X: Union[pd.DataFrame, np.ndarray],
    T: Union[pd.Series, np.ndarray],
    Y: Union[pd.Series, np.ndarray],
    variance_scales: Sequence[float] = (1.0, 0.5, 0.2, 0.1, 0.05, 0.01),
    model_y: Optional[Any] = None,
    model_t: Optional[Any] = None,
    nuisance_name: str = "Ridge",
    random_state: int = 42,
    alpha: float = 0.05,
) -> Tuple[RobustnessReport, List[Dict[str, Any]]]:
    """Stress test DML behavior when treatment variation shrinks towards zero.

    Scales treatment variance:
        T_scaled = mean(T) + scale * (T - mean(T))
    and traces:
    - Standard error explosion: SE(theta) ~ 1 / scale
    - Effective variation ratio collapse
    - Second stage conditioning diagnostics
    """
    T_arr = np.asarray(T, dtype=np.float64).ravel().copy()
    T_mean = np.mean(T_arr)

    report = RobustnessReport()
    diagnostics_list: List[Dict[str, Any]] = []

    for scale in variance_scales:
        T_scaled = T_mean + scale * (T_arr - T_mean)

        try:
            my = clone(model_y) if model_y is not None else Ridge(alpha=1.0)
            mt = clone(model_t) if model_t is not None else Ridge(alpha=1.0)

            dml, res = fit_and_record_dml(
                X=X,
                T=T_scaled,
                Y=Y,
                model_y=my,
                model_t=mt,
                n_folds=5,
                random_state=random_state,
                alpha=alpha,
                experiment_name="weak_treatment_variation",
                nuisance_name=nuisance_name,
                extra_info={"variance_scale": scale},
            )
            report.add(res)

            diag = diagnose_treatment_variation(T_scaled, dml.pred_t_)
            diag["scale"] = scale
            diag["theta"] = res.theta
            diag["std_error"] = res.std_error
            diagnostics_list.append(diag)
        except ValueError as e:
            # Handle extreme zero-variance denominator
            diagnostics_list.append({
                "scale": scale,
                "status": "FAILED_ZERO_VARIANCE",
                "severity": "CRITICAL",
                "warning_message": str(e),
            })

    return report, diagnostics_list


# ==============================================================================
# 9. INTEGRATED EVALUATION SUITE ORCHESTRATOR
# ==============================================================================


class DMLRobustnessEvaluator:
    """End-to-end evaluation orchestrator for DML robustness benchmarks."""

    def __init__(
        self,
        X: Union[pd.DataFrame, np.ndarray],
        T: Union[pd.Series, np.ndarray],
        Y: Union[pd.Series, np.ndarray],
        alpha: float = 0.05,
        random_state: int = 42,
    ):
        self.X = X
        self.T = T
        self.Y = Y
        self.alpha = alpha
        self.random_state = random_state
        self.report = RobustnessReport()
        self.cross_fit_aggregations: Dict[int, MultiSplitAggregation] = {}
        self.weak_variation_diagnostics: List[Dict[str, Any]] = []
        self.bootstrap_ci_results: Dict[str, Any] = {}

    @classmethod
    def from_dataframe(
        cls,
        df: pd.DataFrame,
        treatment_col: str = DEFAULT_TREATMENT_COL,
        outcome_col: str = DEFAULT_OUTCOME_COL,
        confounder_cols: Optional[List[str]] = None,
        categorical_confounders: Optional[List[str]] = None,
        log_transform: bool = True,
        test_size: float = 0.2,
        random_state: int = 42,
    ) -> "DMLRobustnessEvaluator":
        """Factory constructor initializing evaluator from raw pandas DataFrame."""
        splits = split_causal_data(
            df=df,
            test_size=test_size,
            random_state=random_state,
            treatment_col=treatment_col,
            outcome_col=outcome_col,
            confounder_cols=confounder_cols,
            categorical_confounders=categorical_confounders,
            log_transform_treatment=log_transform,
            log_transform_outcome=log_transform,
        )
        return cls(
            X=splits["X_train"],
            T=splits["T_train"],
            Y=splits["Y_train"],
            random_state=random_state,
        )

    def run_nuisance_comparison(
        self,
        candidates: Optional[Dict[str, Tuple[Any, Any]]] = None,
        n_folds: int = 5,
    ) -> RobustnessReport:
        """Run nuisance model comparison experiment."""
        sub_report = compare_nuisance_models(
            X=self.X,
            T=self.T,
            Y=self.Y,
            candidates=candidates,
            n_folds=n_folds,
            random_state=self.random_state,
            alpha=self.alpha,
        )
        self.report.extend(sub_report.records)
        return sub_report

    def run_cross_fitting_evaluation(
        self,
        folds_list: Sequence[int] = (2, 3, 5, 8, 10),
        seeds_list: Sequence[int] = (42, 101, 202, 303, 404),
        model_y: Optional[Any] = None,
        model_t: Optional[Any] = None,
        nuisance_name: str = "default_rf",
    ) -> Tuple[RobustnessReport, Dict[int, MultiSplitAggregation]]:
        """Run cross-fitting and random seed sensitivity evaluation."""
        sub_report, aggs = evaluate_cross_fitting_and_seeds(
            X=self.X,
            T=self.T,
            Y=self.Y,
            folds_list=folds_list,
            seeds_list=seeds_list,
            model_y=model_y,
            model_t=model_t,
            nuisance_name=nuisance_name,
            alpha=self.alpha,
        )
        self.report.extend(sub_report.records)
        self.cross_fit_aggregations.update(aggs)
        return sub_report, aggs

    def run_bootstrap_uncertainty(
        self,
        n_bootstraps: int = 30,
        n_folds: int = 3,
        model_y: Optional[Any] = None,
        model_t: Optional[Any] = None,
    ) -> Dict[str, Any]:
        """Compute bootstrap confidence intervals."""
        self.bootstrap_ci_results = compute_bootstrap_ci(
            X=self.X,
            T=self.T,
            Y=self.Y,
            n_bootstraps=n_bootstraps,
            alpha=self.alpha,
            n_folds=n_folds,
            model_y=model_y,
            model_t=model_t,
            random_state=self.random_state,
        )
        return self.bootstrap_ci_results

    def run_small_sample_stress_test(
        self,
        sample_sizes: Sequence[int] = (30, 50, 100, 200, 400),
        n_repeats: int = 3,
        model_y: Optional[Any] = None,
        model_t: Optional[Any] = None,
        nuisance_name: str = "default_rf",
    ) -> RobustnessReport:
        """Run small sample stress test experiment."""
        sub_report = evaluate_small_sample_behavior(
            X=self.X,
            T=self.T,
            Y=self.Y,
            sample_sizes=sample_sizes,
            n_repeats=n_repeats,
            model_y=model_y,
            model_t=model_t,
            nuisance_name=nuisance_name,
            random_state=self.random_state,
            alpha=self.alpha,
        )
        self.report.extend(sub_report.records)
        return sub_report

    def run_missing_values_stress_test(
        self,
        missing_rates: Sequence[float] = (0.0, 0.05, 0.10, 0.20),
        strategies: Sequence[str] = ("median", "mean", "indicator", "drop"),
        model_y: Optional[Any] = None,
        model_t: Optional[Any] = None,
        nuisance_name: str = "default_rf",
    ) -> RobustnessReport:
        """Run missing values stress test experiment."""
        sub_report = evaluate_missing_value_behavior(
            X=self.X,
            T=self.T,
            Y=self.Y,
            missing_rates=missing_rates,
            strategies=strategies,
            model_y=model_y,
            model_t=model_t,
            nuisance_name=nuisance_name,
            random_state=self.random_state,
            alpha=self.alpha,
        )
        self.report.extend(sub_report.records)
        return sub_report

    def run_weak_treatment_variation_stress_test(
        self,
        variance_scales: Sequence[float] = (1.0, 0.5, 0.2, 0.1, 0.05, 0.01),
        model_y: Optional[Any] = None,
        model_t: Optional[Any] = None,
        nuisance_name: str = "Ridge",
    ) -> Tuple[RobustnessReport, List[Dict[str, Any]]]:
        """Run weak treatment variation stress test experiment."""
        sub_report, diags = evaluate_weak_treatment_variation(
            X=self.X,
            T=self.T,
            Y=self.Y,
            variance_scales=variance_scales,
            model_y=model_y,
            model_t=model_t,
            nuisance_name=nuisance_name,
            random_state=self.random_state,
            alpha=self.alpha,
        )
        self.report.extend(sub_report.records)
        self.weak_variation_diagnostics = diags
        return sub_report, diags

    def run_full_suite(
        self,
        include_bootstrap: bool = True,
        bootstrap_samples: int = 30,
    ) -> RobustnessReport:
        """Execute all robustness and stress-testing experiments."""
        # 1. Nuisance comparison
        self.run_nuisance_comparison()

        # 2. Cross-fitting & Seed sensitivity
        self.run_cross_fitting_evaluation(
            folds_list=(2, 3, 5, 8, 10),
            seeds_list=(42, 101, 202, 303, 404),
        )

        # 3. Small sample stress test
        self.run_small_sample_stress_test(
            sample_sizes=(30, 50, 100, 200, 400),
            n_repeats=3,
        )

        # 4. Missing value stress test
        self.run_missing_values_stress_test(
            missing_rates=(0.0, 0.05, 0.10, 0.20),
            strategies=("median", "mean", "indicator", "drop"),
        )

        # 5. Weak treatment variation stress test
        self.run_weak_treatment_variation_stress_test(
            variance_scales=(1.0, 0.5, 0.2, 0.1, 0.05, 0.01),
        )

        # 6. Bootstrap uncertainty validation
        if include_bootstrap:
            self.run_bootstrap_uncertainty(n_bootstraps=bootstrap_samples)

        return self.report

    def format_full_report(self) -> str:
        """Format consolidated ASCII summary of all robustness experiments."""
        lines = [
            "=" * 78,
            "       DOUBLE MACHINE LEARNING (DML) ROBUSTNESS & CROSS-FITTING REPORT",
            "=" * 78,
        ]

        # Section 1: Nuisance Models
        df_all = self.report.to_dataframe()
        if not df_all.empty and "nuisance_comparison" in df_all["experiment"].values:
            lines.append("\n[1. NUISANCE MODEL COMPARISON]")
            sub = df_all[df_all["experiment"] == "nuisance_comparison"]
            cols = ["nuisance_name", "theta", "std_error", "ci_lower", "ci_upper", "r2_y", "r2_t"]
            sub_disp = sub[[c for c in cols if c in sub.columns]].copy()
            lines.append(sub_disp.to_string(index=False))

        # Section 2: Multi-Split Cross-Fitting Aggregation
        if self.cross_fit_aggregations:
            lines.append("\n[2. CROSS-FITTING MULTI-SPLIT AGGREGATIONS (Chernozhukov et al. 2018)]")
            agg_rows = []
            for k, agg in self.cross_fit_aggregations.items():
                agg_rows.append({
                    "K_folds": k,
                    "theta_med": f"{agg.theta_median:.4f}",
                    "theta_std": f"{agg.theta_std:.4f}",
                    "SE_pooled": f"{agg.std_error_pooled:.4f}",
                    "95% CI Pooled": f"[{agg.ci_lower_pooled:.4f}, {agg.ci_upper_pooled:.4f}]",
                })
            lines.append(pd.DataFrame(agg_rows).to_string(index=False))

        # Section 3: Bootstrap Uncertainty Intervals
        if self.bootstrap_ci_results:
            lines.append("\n[3. NON-PARAMETRIC BOOTSTRAP UNCERTAINTY INTERVALS]")
            b = self.bootstrap_ci_results
            lines.append(f"  - Successful resamples:   {b.get('n_bootstraps_successful')}/{b.get('n_bootstraps_requested')}")
            lines.append(f"  - Bootstrap Mean Theta:   {b.get('theta_boot_mean', 0.0):.4f}")
            lines.append(f"  - Bootstrap Std Error:    {b.get('theta_boot_std', 0.0):.4f}")
            lines.append(f"  - 95% Percentile CI:      [{b.get('ci_percentile_lower', 0.0):.4f}, {b.get('ci_percentile_upper', 0.0):.4f}] (width: {b.get('ci_percentile_width', 0.0):.4f})")

        # Section 4: Weak Treatment Variation Diagnostics
        if self.weak_variation_diagnostics:
            lines.append("\n[4. WEAK TREATMENT VARIATION STRESS TEST]")
            diag_rows = []
            for d in self.weak_variation_diagnostics:
                diag_rows.append({
                    "Scale": d.get("scale"),
                    "Status": d.get("status"),
                    "EVR": f"{d.get('effective_variation_ratio', 0.0):.2%}" if "effective_variation_ratio" in d else "N/A",
                    "SE(theta)": f"{d.get('std_error', 0.0):.4f}" if "std_error" in d else "N/A",
                })
            lines.append(pd.DataFrame(diag_rows).to_string(index=False))

        # Section 5: Small-Sample Stress Test
        if not df_all.empty and "small_sample_stress" in df_all["experiment"].values:
            lines.append("\n[5. SMALL-SAMPLE STRESS TEST (Finite-Sample SE Scaling)]")
            sub_ss = df_all[df_all["experiment"] == "small_sample_stress"].copy()
            ss_summary = (
                sub_ss.groupby("sample_size")
                .agg(
                    mean_theta=("theta", "mean"),
                    std_theta=("theta", lambda x: float(np.std(x, ddof=1)) if len(x) > 1 else 0.0),
                    mean_se=("std_error", "mean"),
                    mean_ci_width=("ci_width", "mean"),
                    n_runs=("theta", "count"),
                )
                .reset_index()
            )
            format_map = {
                "mean_theta": "{:.4f}".format,
                "std_theta": "{:.4f}".format,
                "mean_se": "{:.4f}".format,
                "mean_ci_width": "{:.4f}".format,
            }
            for col, fmt in format_map.items():
                ss_summary[col] = ss_summary[col].apply(fmt)
            lines.append(ss_summary.to_string(index=False))

        # Section 6: Missing-Values Stress Test
        if not df_all.empty and "missing_values_stress" in df_all["experiment"].values:
            lines.append("\n[6. MISSING-VALUES STRESS TEST (MCAR Confounder Imputation)]")
            sub_mv = df_all[df_all["experiment"] == "missing_values_stress"].copy()
            cols_mv = ["extra_strategy", "extra_missing_rate", "theta", "std_error", "ci_lower", "ci_upper", "r2_y"]
            avail_cols = [c for c in cols_mv if c in sub_mv.columns]
            disp_mv = sub_mv[avail_cols].copy()
            disp_mv.rename(columns={"extra_strategy": "strategy", "extra_missing_rate": "missing_rate"}, inplace=True)
            for c in ["theta", "std_error", "ci_lower", "ci_upper", "r2_y"]:
                if c in disp_mv.columns:
                    disp_mv[c] = disp_mv[c].apply(lambda x: f"{x:.4f}")
            lines.append(disp_mv.to_string(index=False))

        lines.append("\n" + "=" * 78)
        return "\n".join(lines)


# ==============================================================================
# 10. CLI SCRIPT ENTRY POINT
# ==============================================================================


def main() -> int:
    """CLI entrypoint for running DML robustness evaluation."""
    parser = argparse.ArgumentParser(
        description="EconoCausal DML Robustness & Cross-Fitting Evaluation Suite",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--data-path",
        type=str,
        default="data/raw/retail_price.csv",
        help="Path to retail dataset CSV",
    )
    parser.add_argument(
        "--experiment",
        type=str,
        default="all",
        choices=["all", "nuisance", "cross_fitting", "small_sample", "missing", "weak_variation"],
        help="Specific experiment to execute",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="experiments/results",
        help="Directory to save CSV/JSON reports",
    )
    parser.add_argument(
        "--bootstrap-samples",
        type=int,
        default=25,
        help="Number of bootstrap resamples for uncertainty evaluation",
    )
    parser.add_argument(
        "--random-state",
        type=int,
        default=42,
        help="Random seed for reproducibility",
    )

    args = parser.parse_args()

    print(f"Loading retail price dataset from {args.data_path}...")
    if not os.path.exists(args.data_path):
        print(f"Error: dataset path '{args.data_path}' not found.", file=sys.stderr)
        return 1

    df = load_retail_price_dataset(args.data_path)
    print(f"Loaded {len(df)} transactions. Initializing DML robustness evaluator...")

    evaluator = DMLRobustnessEvaluator.from_dataframe(
        df=df,
        random_state=args.random_state,
    )

    if args.experiment == "all":
        print("Running full DML robustness and stress-testing suite...")
        evaluator.run_full_suite(
            include_bootstrap=True,
            bootstrap_samples=args.bootstrap_samples,
        )
    elif args.experiment == "nuisance":
        print("Running nuisance model comparison...")
        evaluator.run_nuisance_comparison()
    elif args.experiment == "cross_fitting":
        print("Running cross-fitting & seed sensitivity evaluation...")
        evaluator.run_cross_fitting_evaluation()
    elif args.experiment == "small_sample":
        print("Running small-sample stress test...")
        evaluator.run_small_sample_stress_test()
    elif args.experiment == "missing":
        print("Running missing-values stress test...")
        evaluator.run_missing_values_stress_test()
    elif args.experiment == "weak_variation":
        print("Running weak treatment variation stress test...")
        evaluator.run_weak_treatment_variation_stress_test()

    report_text = evaluator.format_full_report()
    print("\n" + report_text)

    # Save artifacts
    csv_path = os.path.join(args.output_dir, "dml_robustness_results.csv")
    json_path = os.path.join(args.output_dir, "dml_robustness_results.json")
    evaluator.report.save_csv(csv_path)
    evaluator.report.save_json(json_path)
    print(f"\nArtifacts persisted:\n  - CSV:  {csv_path}\n  - JSON: {json_path}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
