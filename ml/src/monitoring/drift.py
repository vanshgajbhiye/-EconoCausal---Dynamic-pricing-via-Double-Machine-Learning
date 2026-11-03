"""Continuous Causal MLOps, Confounder Drift Detection & Automated Refutation Re-Auditing.

Provides statistical and causal distribution monitoring for production dynamic pricing:
1. Confounder Population Stability Index (PSI), Kolmogorov-Smirnov (KS) tests,
   and Wasserstein distance metrics across high-dimensional covariates.
2. Treatment (Price) and Outcome (Demand) distribution drift tracking.
3. Heterogeneous Price Elasticity (CATE) shift tracking and microeconomic regime checks.
4. Continuous automated causal refutation re-audits (Placebo, Random Common Cause, Subsampling).
5. Multi-criteria automated retraining trigger decision engine (HEALTHY, WARNING,
   RETRAIN_RECOMMENDED, CRITICAL).
"""

import datetime
import json
import os
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
from scipy import stats

from src.data.load_data import (
    DEFAULT_CONFOUNDER_COLS,
    DEFAULT_OUTCOME_COL,
    DEFAULT_TREATMENT_COL,
)
from src.models.refutation import CausalRefuter


# Standard industry PSI thresholds
PSI_THRESHOLD_STABLE = 0.10
PSI_THRESHOLD_ALERT = 0.25


def calculate_population_stability_index(
    baseline: Union[np.ndarray, pd.Series, List[float]],
    target: Union[np.ndarray, pd.Series, List[float]],
    num_bins: int = 10,
    is_categorical: bool = False,
    epsilon: float = 1e-4,
) -> float:
    """Calculate Population Stability Index (PSI) between baseline and target distributions.

    PSI measures the shift between two probability distributions:
        PSI = sum_{b=1}^B (P_target,b - P_baseline,b) * ln(P_target,b / P_baseline,b)

    Interpretation:
        - PSI < 0.10: No significant shift (Stable)
        - 0.10 <= PSI < 0.25: Moderate shift (Warning)
        - PSI >= 0.25: Significant shift (Action/Retraining required)

    Parameters
    ----------
    baseline : Union[np.ndarray, pd.Series, List[float]]
        Reference or training distribution.
    target : Union[np.ndarray, pd.Series, List[float]]
        Incoming production or evaluation distribution.
    num_bins : int, default=10
        Number of quantile-based buckets for continuous features.
    is_categorical : bool, default=False
        Whether the feature is categorical (frequency based) or continuous (binned).
    epsilon : float, default=1e-4
        Small regularization constant to avoid division by zero or ln(0).

    Returns
    -------
    float
        Computed Population Stability Index (>= 0.0).
    """
    base_arr = pd.Series(baseline).dropna().to_numpy()
    target_arr = pd.Series(target).dropna().to_numpy()

    if len(base_arr) == 0 or len(target_arr) == 0:
        return 0.0

    # Fast path: identical distributions
    if len(base_arr) == len(target_arr) and np.array_equal(base_arr, target_arr):
        return 0.0

    if is_categorical:
        # Categorical frequency binning
        all_categories = sorted(list(set(base_arr) | set(target_arr)))
        base_counts = pd.Series(base_arr).value_counts(normalize=False).reindex(all_categories, fill_value=0)
        target_counts = pd.Series(target_arr).value_counts(normalize=False).reindex(all_categories, fill_value=0)

        p_base = (base_counts.to_numpy(dtype=float) + epsilon) / (len(base_arr) + len(all_categories) * epsilon)
        p_target = (target_counts.to_numpy(dtype=float) + epsilon) / (len(target_arr) + len(all_categories) * epsilon)

    else:
        # Continuous quantile binning based on baseline percentiles
        percentiles = np.linspace(0, 100, num_bins + 1)
        bin_edges = np.percentile(base_arr, percentiles)
        bin_edges = np.unique(bin_edges)

        # If baseline has too few unique values, fallback to min-max equal bins
        if len(bin_edges) < 2:
            min_val = min(base_arr.min(), target_arr.min())
            max_val = max(base_arr.max(), target_arr.max())
            if min_val == max_val:
                return 0.0
            bin_edges = np.linspace(min_val, max_val, num_bins + 1)

        # Expand outer edges slightly to include min and max points
        bin_edges[0] -= 1e-5
        bin_edges[-1] += 1e-5

        base_counts, _ = np.histogram(base_arr, bins=bin_edges)
        target_counts, _ = np.histogram(target_arr, bins=bin_edges)

        n_bins = len(base_counts)
        p_base = (base_counts.astype(float) + epsilon) / (len(base_arr) + n_bins * epsilon)
        p_target = (target_counts.astype(float) + epsilon) / (len(target_arr) + n_bins * epsilon)

    # Compute PSI summation
    psi_value = np.sum((p_target - p_base) * np.log(p_target / p_base))
    return float(max(0.0, psi_value))


class FeatureDriftResult:
    """Detailed drift diagnostics for a single feature or variable."""

    def __init__(
        self,
        feature_name: str,
        psi: float,
        drift_level: str,
        drift_detected: bool,
        is_categorical: bool = False,
        ks_stat: Optional[float] = None,
        p_value: Optional[float] = None,
        wasserstein_dist: Optional[float] = None,
        baseline_mean: Optional[float] = None,
        target_mean: Optional[float] = None,
        mean_shift_pct: Optional[float] = None,
    ):
        self.feature_name = feature_name
        self.psi = round(psi, 5)
        self.drift_level = drift_level
        self.drift_detected = drift_detected
        self.is_categorical = is_categorical
        self.ks_stat = round(ks_stat, 5) if ks_stat is not None else None
        self.p_value = round(p_value, 5) if p_value is not None else None
        self.wasserstein_dist = round(wasserstein_dist, 5) if wasserstein_dist is not None else None
        self.baseline_mean = round(baseline_mean, 4) if baseline_mean is not None else None
        self.target_mean = round(target_mean, 4) if target_mean is not None else None
        self.mean_shift_pct = round(mean_shift_pct, 2) if mean_shift_pct is not None else None

    def to_dict(self) -> Dict[str, Any]:
        """Convert result to dictionary representation."""
        return {
            "feature_name": self.feature_name,
            "psi": self.psi,
            "drift_level": self.drift_level,
            "drift_detected": self.drift_detected,
            "is_categorical": self.is_categorical,
            "ks_stat": self.ks_stat,
            "p_value": self.p_value,
            "wasserstein_distance": self.wasserstein_dist,
            "baseline_mean": self.baseline_mean,
            "target_mean": self.target_mean,
            "mean_shift_pct": self.mean_shift_pct,
        }


class CausalDriftReport:
    """Comprehensive Causal MLOps Drift and Health Audit Report."""

    def __init__(
        self,
        timestamp: str,
        dataset_baseline_rows: int,
        dataset_target_rows: int,
        overall_status: str,
        recommendation: str,
        confounder_drift: Dict[str, FeatureDriftResult],
        max_confounder_psi: float,
        mean_confounder_psi: float,
        num_drifting_confounders: int,
        treatment_drift: Optional[FeatureDriftResult] = None,
        outcome_drift: Optional[FeatureDriftResult] = None,
        elasticity_drift: Optional[Dict[str, Any]] = None,
        refutation_audit: Optional[Dict[str, Any]] = None,
    ):
        self.timestamp = timestamp
        self.dataset_baseline_rows = dataset_baseline_rows
        self.dataset_target_rows = dataset_target_rows
        self.overall_status = overall_status
        self.recommendation = recommendation
        self.confounder_drift = confounder_drift
        self.max_confounder_psi = round(max_confounder_psi, 4)
        self.mean_confounder_psi = round(mean_confounder_psi, 4)
        self.num_drifting_confounders = num_drifting_confounders
        self.treatment_drift = treatment_drift
        self.outcome_drift = outcome_drift
        self.elasticity_drift = elasticity_drift
        self.refutation_audit = refutation_audit

    def to_dict(self) -> Dict[str, Any]:
        """Serialize full report to structured nested dictionary."""
        return {
            "timestamp": self.timestamp,
            "dataset_baseline_rows": self.dataset_baseline_rows,
            "dataset_target_rows": self.dataset_target_rows,
            "overall_status": self.overall_status,
            "recommendation": self.recommendation,
            "max_confounder_psi": self.max_confounder_psi,
            "mean_confounder_psi": self.mean_confounder_psi,
            "num_drifting_confounders": self.num_drifting_confounders,
            "confounders": {k: v.to_dict() for k, v in self.confounder_drift.items()},
            "treatment_drift": self.treatment_drift.to_dict() if self.treatment_drift else None,
            "outcome_drift": self.outcome_drift.to_dict() if self.outcome_drift else None,
            "elasticity_drift": self.elasticity_drift,
            "refutation_audit": self.refutation_audit,
        }

    def summary(self) -> str:
        """Generate human-readable terminal audit summary."""
        lines = [
            "=" * 70,
            "[DRIFT REPORT] ECONOCAUSAL CAUSAL MLOPS DRIFT & GOVERNANCE REPORT",
            "=" * 70,
            f"Timestamp:                 {self.timestamp}",
            f"Baseline Samples:          {self.dataset_baseline_rows:,}",
            f"Target Production Samples: {self.dataset_target_rows:,}",
            f"Overall Status:            {self.overall_status}",
            f"Recommendation:            {self.recommendation}",
            "-" * 70,
            f"Max Confounder PSI:        {self.max_confounder_psi:.4f}",
            f"Mean Confounder PSI:       {self.mean_confounder_psi:.4f}",
            f"Drifting Confounders:      {self.num_drifting_confounders} / {len(self.confounder_drift)}",
        ]

        if self.treatment_drift:
            lines.append(
                f"Treatment (Price) PSI:     {self.treatment_drift.psi:.4f} [{self.treatment_drift.drift_level}]"
            )
        if self.outcome_drift:
            lines.append(
                f"Outcome (Demand) PSI:      {self.outcome_drift.psi:.4f} [{self.outcome_drift.drift_level}]"
            )

        if self.elasticity_drift:
            lines.append(
                f"CATE Elasticity PSI:       {self.elasticity_drift.get('psi', 0.0):.4f} [{self.elasticity_drift.get('drift_level', 'UNKNOWN')}]"
            )
            delta = self.elasticity_drift.get('delta_mean', 0.0)
            lines.append(
                f"Mean Elasticity Shift:     {self.elasticity_drift.get('baseline_mean', 0.0):.3f} -> {self.elasticity_drift.get('target_mean', 0.0):.3f} (Delta {delta:+.3f})"
            )

        if self.refutation_audit:
            pass_rate = self.refutation_audit.get("pass_rate", 1.0) * 100
            lines.append(f"Production Refutations:    {pass_rate:.1f}% Passed")

        lines.append("-" * 70)
        lines.append(f"{'Feature Name':<28} {'PSI':<10} {'Level':<18} {'KS Stat':<10} {'p-val'}")
        lines.append("-" * 70)

        # Sort features by PSI descending
        sorted_features = sorted(self.confounder_drift.values(), key=lambda f: f.psi, reverse=True)
        for feat in sorted_features:
            ks_str = f"{feat.ks_stat:.4f}" if feat.ks_stat is not None else "N/A"
            p_str = f"{feat.p_value:.4f}" if feat.p_value is not None else "N/A"
            lines.append(f"{feat.feature_name:<28} {feat.psi:<10.4f} {feat.drift_level:<18} {ks_str:<10} {p_str}")

        lines.append("=" * 70)
        return "\n".join(lines)

    def save_json(self, output_path: str) -> None:
        """Save report to a JSON file."""
        os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(self.to_dict(), f, indent=2)


class CausalDriftDetector:
    """Engine for continuous confounder drift detection, elasticity shift, and refutation re-auditing.

    Parameters
    ----------
    psi_warning_threshold : float, default=0.10
        Threshold for moderate drift warning.
    psi_alert_threshold : float, default=0.25
        Threshold for significant drift alert triggering retraining.
    num_bins : int, default=10
        Number of quantile bins for continuous variable PSI calculation.
    random_state : int, default=42
        Random seed for refutation re-audits.
    """

    def __init__(
        self,
        psi_warning_threshold: float = PSI_THRESHOLD_STABLE,
        psi_alert_threshold: float = PSI_THRESHOLD_ALERT,
        num_bins: int = 10,
        random_state: int = 42,
    ):
        self.psi_warning_threshold = psi_warning_threshold
        self.psi_alert_threshold = psi_alert_threshold
        self.num_bins = num_bins
        self.random_state = random_state

    def detect_feature_drift(
        self,
        baseline: pd.Series,
        target: pd.Series,
        feature_name: Optional[str] = None,
    ) -> FeatureDriftResult:
        """Analyze drift for a single covariate or metric.

        Computes PSI, two-sample Kolmogorov-Smirnov test (for continuous data),
        Wasserstein distance, and mean percentage shift.

        Parameters
        ----------
        baseline : pd.Series
            Baseline / reference series.
        target : pd.Series
            Target / production series.
        feature_name : Optional[str]
            Name of feature. If None, inferred from Series name.

        Returns
        -------
        FeatureDriftResult
            Detailed feature drift result.
        """
        name = feature_name or str(baseline.name or "feature")
        is_categorical = baseline.dtype == "object" or isinstance(baseline.dtype, pd.CategoricalDtype)

        psi = calculate_population_stability_index(
            baseline=baseline,
            target=target,
            num_bins=self.num_bins,
            is_categorical=is_categorical,
        )

        # Assign drift level based on PSI
        if psi >= self.psi_alert_threshold:
            drift_level = "SIGNIFICANT_DRIFT"
            drift_detected = True
        elif psi >= self.psi_warning_threshold:
            drift_level = "MODERATE_DRIFT"
            drift_detected = True
        else:
            drift_level = "STABLE"
            drift_detected = False

        ks_stat: Optional[float] = None
        p_val: Optional[float] = None
        wass_dist: Optional[float] = None
        base_mean: Optional[float] = None
        targ_mean: Optional[float] = None
        shift_pct: Optional[float] = None

        if not is_categorical:
            b_clean = baseline.dropna().to_numpy(dtype=float)
            t_clean = target.dropna().to_numpy(dtype=float)

            if len(b_clean) > 0 and len(t_clean) > 0:
                # Two-sample KS test
                ks_res = stats.ks_2samp(b_clean, t_clean)
                ks_stat = float(ks_res.statistic)
                p_val = float(ks_res.pvalue)

                # Wasserstein distance (Earth Mover's Distance)
                try:
                    wass_dist = float(stats.wasserstein_distance(b_clean, t_clean))
                except Exception:
                    wass_dist = None

                base_mean = float(np.mean(b_clean))
                targ_mean = float(np.mean(t_clean))

                if abs(base_mean) > 1e-6:
                    shift_pct = float(((targ_mean - base_mean) / abs(base_mean)) * 100.0)
                else:
                    shift_pct = 0.0

        return FeatureDriftResult(
            feature_name=name,
            psi=psi,
            drift_level=drift_level,
            drift_detected=drift_detected,
            is_categorical=is_categorical,
            ks_stat=ks_stat,
            p_value=p_val,
            wasserstein_dist=wass_dist,
            baseline_mean=base_mean,
            target_mean=targ_mean,
            mean_shift_pct=shift_pct,
        )

    def audit_confounders(
        self,
        baseline_df: pd.DataFrame,
        target_df: pd.DataFrame,
        confounder_cols: Optional[List[str]] = None,
    ) -> Dict[str, FeatureDriftResult]:
        """Audit confounder feature distributions for covariate shift.

        Parameters
        ----------
        baseline_df : pd.DataFrame
            Reference training dataset.
        target_df : pd.DataFrame
            Production evaluation dataset.
        confounder_cols : Optional[List[str]]
            Columns to audit. If None, defaults to columns present in both datasets.

        Returns
        -------
        Dict[str, FeatureDriftResult]
            Dictionary of feature drift results keyed by feature name.
        """
        if confounder_cols is None:
            candidate_cols = [
                c
                for c in DEFAULT_CONFOUNDER_COLS
                if c in baseline_df.columns and c in target_df.columns
            ]
            if not candidate_cols:
                # Fallback to shared columns excluding treatment and outcome
                candidate_cols = [
                    c
                    for c in baseline_df.columns
                    if c in target_df.columns
                    and c not in [DEFAULT_TREATMENT_COL, DEFAULT_OUTCOME_COL, "product_id"]
                ]
            confounder_cols = candidate_cols

        results: Dict[str, FeatureDriftResult] = {}
        for col in confounder_cols:
            if col in baseline_df.columns and col in target_df.columns:
                results[col] = self.detect_feature_drift(
                    baseline=baseline_df[col],
                    target=target_df[col],
                    feature_name=col,
                )

        return results

    def audit_treatment_outcome(
        self,
        baseline_df: pd.DataFrame,
        target_df: pd.DataFrame,
        treatment_col: str = DEFAULT_TREATMENT_COL,
        outcome_col: str = DEFAULT_OUTCOME_COL,
    ) -> Tuple[Optional[FeatureDriftResult], Optional[FeatureDriftResult]]:
        """Audit treatment (price) and outcome (sales demand) distribution drift."""
        t_drift: Optional[FeatureDriftResult] = None
        y_drift: Optional[FeatureDriftResult] = None

        if treatment_col in baseline_df.columns and treatment_col in target_df.columns:
            t_drift = self.detect_feature_drift(
                baseline=baseline_df[treatment_col],
                target=target_df[treatment_col],
                feature_name=f"{treatment_col} (Treatment)",
            )

        if outcome_col in baseline_df.columns and outcome_col in target_df.columns:
            y_drift = self.detect_feature_drift(
                baseline=baseline_df[outcome_col],
                target=target_df[outcome_col],
                feature_name=f"{outcome_col} (Outcome)",
            )

        return t_drift, y_drift

    def audit_elasticity_drift(
        self,
        pipeline: Any,
        baseline_df: pd.DataFrame,
        target_df: pd.DataFrame,
    ) -> Dict[str, Any]:
        """Audit heterogeneous causal price elasticity (CATE) distribution drift.

        Evaluates whether customer price sensitivity has structurally shifted
        between baseline and target observations.

        Parameters
        ----------
        pipeline : EconoCausalPipeline
            Fitted EconoCausal pipeline.
        baseline_df : pd.DataFrame
            Reference dataset.
        target_df : pd.DataFrame
            Target dataset.

        Returns
        -------
        Dict[str, Any]
            Elasticity drift diagnostics dictionary.
        """
        # Predict heterogeneous elasticities
        theta_base = pipeline.predict_elasticity(baseline_df)
        theta_target = pipeline.predict_elasticity(target_df)

        psi = calculate_population_stability_index(
            baseline=theta_base,
            target=theta_target,
            num_bins=self.num_bins,
            is_categorical=False,
        )

        base_mean = float(np.mean(theta_base))
        target_mean = float(np.mean(theta_target))
        delta_mean = float(target_mean - base_mean)

        # KS test on elasticity distributions
        ks_res = stats.ks_2samp(theta_base, theta_target)

        # Check for category-level microeconomic regime changes (elastic < -1 vs inelastic >= -1)
        category_col = "product_category_name"
        regime_changes: List[Dict[str, Any]] = []

        if category_col in baseline_df.columns and category_col in target_df.columns:
            base_cats = baseline_df[category_col].unique()
            target_cats = target_df[category_col].unique()
            common_cats = set(base_cats) & set(target_cats)

            for cat in sorted(common_cats):
                idx_b = baseline_df[category_col] == cat
                idx_t = target_df[category_col] == cat

                cat_theta_b = float(np.mean(theta_base[idx_b]))
                cat_theta_t = float(np.mean(theta_target[idx_t]))

                b_is_elastic = cat_theta_b < -1.0
                t_is_elastic = cat_theta_t < -1.0

                if b_is_elastic != t_is_elastic:
                    regime_changes.append(
                        {
                            "category": cat,
                            "baseline_elasticity": round(cat_theta_b, 4),
                            "target_elasticity": round(cat_theta_t, 4),
                            "baseline_regime": "ELASTIC" if b_is_elastic else "INELASTIC",
                            "target_regime": "ELASTIC" if t_is_elastic else "INELASTIC",
                        }
                    )

        if psi >= self.psi_alert_threshold:
            drift_level = "SIGNIFICANT_DRIFT"
        elif psi >= self.psi_warning_threshold:
            drift_level = "MODERATE_DRIFT"
        else:
            drift_level = "STABLE"

        return {
            "psi": round(psi, 5),
            "drift_level": drift_level,
            "baseline_mean": round(base_mean, 4),
            "target_mean": round(target_mean, 4),
            "delta_mean": round(delta_mean, 4),
            "ks_stat": round(float(ks_res.statistic), 5),
            "p_value": round(float(ks_res.pvalue), 5),
            "regime_changes": regime_changes,
            "num_regime_changes": len(regime_changes),
        }

    def audit_refutations(
        self,
        pipeline: Any,
        target_df: pd.DataFrame,
        n_repeats: int = 2,
    ) -> Dict[str, Any]:
        """Perform continuous causal refutation re-auditing on target production data.

        Verifies if causal identification holds on fresh production observations:
        1. Placebo Treatment: Shuffled price must collapse effect to zero (p > 0.05).
        2. Random Common Cause: Injecting noise must not shift estimate > 1 SE.
        3. Data Subsampling: 80% subsample must preserve negative elasticity.

        Parameters
        ----------
        pipeline : EconoCausalPipeline
            Fitted EconoCausal pipeline.
        target_df : pd.DataFrame
            New production evaluation data.
        n_repeats : int, default=2
            Repetitions for refuter simulation tests.

        Returns
        -------
        Dict[str, Any]
            Refutation audit dictionary.
        """
        # Preprocess target dataframe using the pipeline's fitted preprocessor
        X_df, T_series, Y_series = pipeline.preprocessor.transform(target_df)

        # Initialize refuter on the pipeline's fitted DML model
        refuter = CausalRefuter(model=pipeline.dml_model, random_state=self.random_state)

        placebo = refuter.refute_placebo_treatment(
            X=X_df, T=T_series, Y=Y_series, n_repeats=n_repeats
        )
        common_cause = refuter.refute_random_common_cause(
            X=X_df, T=T_series, Y=Y_series, n_repeats=n_repeats
        )
        subset = refuter.refute_data_subset(
            X=X_df, T=T_series, Y=Y_series, n_repeats=n_repeats
        )

        passed_count = sum(
            [
                placebo.get("passed", False),
                common_cause.get("passed", False),
                subset.get("passed", False),
            ]
        )
        total_count = 3
        pass_rate = passed_count / total_count

        return {
            "pass_rate": round(pass_rate, 2),
            "tests_passed": passed_count,
            "total_tests": total_count,
            "placebo_treatment": {
                "passed": placebo.get("passed", False),
                "placebo_theta": round(placebo.get("mean_placebo_theta", 0.0), 4),
                "p_value": round(placebo.get("mean_p_value", 1.0), 4),
            },
            "random_common_cause": {
                "passed": common_cause.get("passed", False),
                "new_theta": round(common_cause.get("mean_new_theta", 0.0), 4),
                "effect_shift": round(common_cause.get("mean_effect_shift", 0.0), 4),
            },
            "data_subset": {
                "passed": subset.get("passed", False),
                "subset_theta": round(subset.get("mean_subset_theta", 0.0), 4),
            },
        }

    def evaluate_health(
        self,
        confounder_results: Dict[str, FeatureDriftResult],
        elasticity_results: Optional[Dict[str, Any]] = None,
        refutation_results: Optional[Dict[str, Any]] = None,
    ) -> Tuple[str, str]:
        """Automated multi-criteria decision engine for Causal MLOps retraining triggers.

        Returns
        -------
        Tuple[str, str]
            (overall_status, recommendation)
            Status can be HEALTHY, WARNING, RETRAIN_RECOMMENDED, or CRITICAL.
        """
        # Criteria 1: Check refutation audit
        if refutation_results is not None:
            pass_rate = refutation_results.get("pass_rate", 1.0)
            if pass_rate < 0.67:  # Failed more than 1 refuter test
                return (
                    "CRITICAL",
                    "Causal identification refutation failed on production data. Unobserved confounding detected; "
                    "pause automated pricing and fallback to rule-based guardrails immediately.",
                )

        # Criteria 2: Check confounder PSI
        psi_values = [res.psi for res in confounder_results.values()]
        max_psi = max(psi_values) if psi_values else 0.0
        sig_drift_count = sum(1 for res in confounder_results.values() if res.psi >= self.psi_alert_threshold)
        mod_drift_count = sum(1 for res in confounder_results.values() if res.psi >= self.psi_warning_threshold)

        # Criteria 3: Check CATE elasticity shift
        elasticity_psi = elasticity_results.get("psi", 0.0) if elasticity_results else 0.0
        num_regime_flips = elasticity_results.get("num_regime_changes", 0) if elasticity_results else 0

        # Decision rules
        if max_psi >= self.psi_alert_threshold or sig_drift_count >= 2 or elasticity_psi >= self.psi_alert_threshold:
            return (
                "RETRAIN_RECOMMENDED",
                "Significant distribution shift detected in critical confounders or price elasticity. "
                "Trigger automated DML pipeline retraining and recalibrate CATE nuisance models.",
            )

        if (
            max_psi >= self.psi_warning_threshold
            or mod_drift_count >= 2
            or elasticity_psi >= self.psi_warning_threshold
            or num_regime_flips > 0
        ):
            return (
                "WARNING",
                "Moderate confounder or elasticity drift observed. Monitor pricing KPIs closely; "
                "schedule scheduled retraining within standard weekly cycle.",
            )

        return (
            "HEALTHY",
            "Confounder distributions and causal identification are stable. "
            "Continuous pricing inference is operating normally within specification.",
        )

    def generate_report(
        self,
        baseline_df: pd.DataFrame,
        target_df: pd.DataFrame,
        pipeline: Optional[Any] = None,
        run_refutations: bool = False,
        confounder_cols: Optional[List[str]] = None,
        treatment_col: str = DEFAULT_TREATMENT_COL,
        outcome_col: str = DEFAULT_OUTCOME_COL,
    ) -> CausalDriftReport:
        """Execute end-to-end drift audit and construct a full CausalDriftReport."""
        timestamp = datetime.datetime.now(datetime.timezone.utc).isoformat()

        # 1. Audit Confounders
        confounder_results = self.audit_confounders(
            baseline_df=baseline_df,
            target_df=target_df,
            confounder_cols=confounder_cols,
        )

        psi_vals = [res.psi for res in confounder_results.values()]
        max_psi = max(psi_vals) if psi_vals else 0.0
        mean_psi = float(np.mean(psi_vals)) if psi_vals else 0.0
        drifting_count = sum(1 for res in confounder_results.values() if res.drift_detected)

        # 2. Audit Treatment and Outcome
        t_drift, y_drift = self.audit_treatment_outcome(
            baseline_df=baseline_df,
            target_df=target_df,
            treatment_col=treatment_col,
            outcome_col=outcome_col,
        )

        # 3. Audit Elasticity Drift (if pipeline provided)
        elasticity_results: Optional[Dict[str, Any]] = None
        if pipeline is not None and getattr(pipeline, "is_fitted_", False):
            try:
                elasticity_results = self.audit_elasticity_drift(
                    pipeline=pipeline,
                    baseline_df=baseline_df,
                    target_df=target_df,
                )
            except Exception:
                elasticity_results = None

        # 4. Audit Refutations (if requested and pipeline provided)
        refutation_results: Optional[Dict[str, Any]] = None
        if run_refutations and pipeline is not None and getattr(pipeline, "is_fitted_", False):
            try:
                refutation_results = self.audit_refutations(
                    pipeline=pipeline,
                    target_df=target_df,
                    n_repeats=2,
                )
            except Exception:
                refutation_results = None

        # 5. Evaluate Health & Recommendations
        overall_status, recommendation = self.evaluate_health(
            confounder_results=confounder_results,
            elasticity_results=elasticity_results,
            refutation_results=refutation_results,
        )

        return CausalDriftReport(
            timestamp=timestamp,
            dataset_baseline_rows=len(baseline_df),
            dataset_target_rows=len(target_df),
            overall_status=overall_status,
            recommendation=recommendation,
            confounder_drift=confounder_results,
            max_confounder_psi=max_psi,
            mean_confounder_psi=mean_psi,
            num_drifting_confounders=drifting_count,
            treatment_drift=t_drift,
            outcome_drift=y_drift,
            elasticity_drift=elasticity_results,
            refutation_audit=refutation_results,
        )
