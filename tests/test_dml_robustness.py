"""Automated tests for DML Robustness and Cross-Fitting Evaluation Suite.

Tests:
1. Nuisance model candidate registry and factory.
2. Nuisance model comparison and out-of-fold metrics.
3. Cross-fitting folds (K) and random seeds sensitivity with Chernozhukov multi-split pooling.
4. Asymptotic and non-parametric bootstrap uncertainty intervals across alpha levels.
5. Small-sample finite-sample behavior and standard error inflation.
6. Missing-value imputation strategies ('median', 'mean', 'indicator', 'drop') and stress testing.
7. Weak treatment variation diagnostics, effective variation ratio (EVR), and SE explosion.
8. RobustnessReport data models, DataFrame export, and CSV/JSON persistence.
9. End-to-end DMLRobustnessEvaluator on real Kaggle retail_price.csv dataset.
"""

import json
import os
import tempfile
import numpy as np
import pandas as pd
import pytest
from sklearn.linear_model import Lasso, Ridge
from sklearn.ensemble import RandomForestRegressor

from experiments.dml_robustness import (
    DMLRobustnessEvaluator,
    MultiSplitAggregation,
    RobustnessReport,
    RobustnessResult,
    compare_nuisance_models,
    compute_bootstrap_ci,
    diagnose_treatment_variation,
    evaluate_cross_fitting_and_seeds,
    evaluate_missing_value_behavior,
    evaluate_small_sample_behavior,
    evaluate_weak_treatment_variation,
    get_nuisance_model_candidates,
    handle_missing_confounders,
)
from src.data.load_data import load_retail_price_dataset
from src.models.train_dml import DoubleMachineLearning


# ==============================================================================
# FIXTURES
# ==============================================================================


@pytest.fixture
def synthetic_dgp_data():
    """Generate synthetic causal data with known true theta = -2.5.

    Structural Data Generating Process:
        X ~ N(0, I_4)
        T = 0.7 * X_0 - 0.5 * X_1 + eta_T,  eta_T ~ N(0, 1)
        Y = theta * T + 1.2 * X_0 + 0.8 * X_2 + eta_Y,  eta_Y ~ N(0, 1)
    """
    np.random.seed(42)
    n_samples = 600
    n_features = 4
    true_theta = -2.5

    X = np.random.randn(n_samples, n_features)
    X_df = pd.DataFrame(X, columns=[f"feat_{i}" for i in range(n_features)])
    T = 0.7 * X[:, 0] - 0.5 * X[:, 1] + np.random.randn(n_samples)
    Y = true_theta * T + 1.2 * X[:, 0] + 0.8 * X[:, 2] + np.random.randn(n_samples)

    return X_df, pd.Series(T, name="treatment"), pd.Series(Y, name="outcome"), true_theta


@pytest.fixture
def real_retail_data():
    """Load real Kaggle retail price dataset."""
    return load_retail_price_dataset("data/raw/retail_price.csv")


# ==============================================================================
# 1. NUISANCE MODEL CANDIDATES & REGISTRY
# ==============================================================================


def test_nuisance_model_candidates_registry():
    """Verify registry returns expected linear, ensemble, and hybrid candidate models."""
    candidates = get_nuisance_model_candidates(random_state=42)

    expected_models = [
        "Ridge",
        "Lasso",
        "ElasticNet",
        "RandomForest",
        "GradientBoosting",
        "ExtraTrees",
        "HistGradientBoosting",
        "Hybrid_RF_Ridge",
        "Hybrid_GB_Ridge",
    ]
    for model_name in expected_models:
        assert model_name in candidates, f"Missing expected nuisance candidate '{model_name}'"
        model_y, model_t = candidates[model_name]
        assert hasattr(model_y, "fit"), f"model_y for {model_name} must implement fit"
        assert hasattr(model_t, "fit"), f"model_t for {model_name} must implement fit"
        assert hasattr(model_y, "predict"), f"model_y for {model_name} must implement predict"
        assert hasattr(model_t, "predict"), f"model_t for {model_name} must implement predict"


def test_compare_nuisance_models_synthetic(synthetic_dgp_data):
    """Verify nuisance model comparison computes out-of-fold metrics and estimates."""
    X, T, Y, true_theta = synthetic_dgp_data

    # Test with a subset of fast candidates
    fast_candidates = {
        "Ridge": (Ridge(alpha=1.0), Ridge(alpha=1.0)),
        "Lasso": (Lasso(alpha=0.01, random_state=42), Lasso(alpha=0.01, random_state=42)),
        "RF_Fast": (
            RandomForestRegressor(n_estimators=20, max_depth=4, random_state=42),
            RandomForestRegressor(n_estimators=20, max_depth=4, random_state=42),
        ),
    }

    report = compare_nuisance_models(
        X=X,
        T=T,
        Y=Y,
        candidates=fast_candidates,
        n_folds=3,
        random_state=42,
    )

    df = report.to_dataframe()
    assert len(df) == 3
    assert set(df["nuisance_name"]) == {"Ridge", "Lasso", "RF_Fast"}

    for _, row in df.iterrows():
        # Parameter estimates should be within 0.25 of true theta (-2.5)
        assert abs(row["theta"] - true_theta) < 0.25
        assert row["std_error"] > 0
        assert row["ci_lower"] <= row["theta"] <= row["ci_upper"]
        assert row["r2_y"] > 0.0
        assert row["r2_t"] > 0.0
        assert row["rmse_y"] > 0.0
        assert row["rmse_t"] > 0.0
        assert row["fit_time_seconds"] >= 0.0


# ==============================================================================
# 2. CROSS-FITTING AND SEED SENSITIVITY (CHERNOZHUKOV POOLING)
# ==============================================================================


def test_cross_fitting_and_seed_sensitivity(synthetic_dgp_data):
    """Verify cross-fitting sensitivity across K folds and seeds with multi-split pooling."""
    X, T, Y, true_theta = synthetic_dgp_data

    folds_list = [2, 3, 5]
    seeds_list = [42, 101, 202]

    report, aggregations = evaluate_cross_fitting_and_seeds(
        X=X,
        T=T,
        Y=Y,
        folds_list=folds_list,
        seeds_list=seeds_list,
        model_y=Ridge(alpha=1.0),
        model_t=Ridge(alpha=1.0),
        nuisance_name="Ridge",
    )

    # 3 folds * 3 seeds = 9 runs
    assert len(report.records) == 9

    for k in folds_list:
        assert k in aggregations
        agg = aggregations[k]
        assert isinstance(agg, MultiSplitAggregation)
        assert agg.n_folds == k
        assert agg.n_splits == len(seeds_list)

        # Median estimate covers true theta
        assert abs(agg.theta_median - true_theta) < 0.20
        assert agg.ci_lower_pooled <= true_theta <= agg.ci_upper_pooled
        assert agg.std_error_pooled > 0.0
        assert agg.ci_width_pooled > 0.0


# ==============================================================================
# 3. UNCERTAINTY INTERVALS: BOOTSTRAP AND ALPHA LEVELS
# ==============================================================================


def test_dml_confidence_interval_alpha_scaling(synthetic_dgp_data):
    """Verify confidence interval widths scale monotonically with significance level alpha."""
    X, T, Y, _ = synthetic_dgp_data

    dml = DoubleMachineLearning(
        model_y=Ridge(),
        model_t=Ridge(),
        n_folds=3,
        random_state=42,
    )
    dml.fit(X, T, Y)

    ci_90 = dml.confidence_interval(alpha=0.10)
    ci_95 = dml.confidence_interval(alpha=0.05)
    ci_99 = dml.confidence_interval(alpha=0.01)

    width_90 = ci_90[1] - ci_90[0]
    width_95 = ci_95[1] - ci_95[0]
    width_99 = ci_99[1] - ci_99[0]

    assert width_90 < width_95 < width_99, (
        f"Expected width_90 < width_95 < width_99, got {width_90:.4f}, {width_95:.4f}, {width_99:.4f}"
    )

    # Test invalid alpha
    with pytest.raises(ValueError, match="alpha must be between 0 and 1"):
        dml.confidence_interval(alpha=1.5)

    with pytest.raises(ValueError, match="alpha must be between 0 and 1"):
        dml.confidence_interval(alpha=-0.05)


def test_bootstrap_uncertainty_intervals(synthetic_dgp_data):
    """Verify non-parametric bootstrap confidence interval estimation."""
    X, T, Y, true_theta = synthetic_dgp_data

    boot_res = compute_bootstrap_ci(
        X=X,
        T=T,
        Y=Y,
        n_bootstraps=20,
        alpha=0.05,
        n_folds=3,
        model_y=Ridge(alpha=1.0),
        model_t=Ridge(alpha=1.0),
        random_state=42,
    )

    assert boot_res["n_bootstraps_successful"] >= 15
    assert abs(boot_res["theta_boot_mean"] - true_theta) < 0.25
    assert boot_res["theta_boot_std"] > 0.0

    pct_lower = boot_res["ci_percentile_lower"]
    pct_upper = boot_res["ci_percentile_upper"]
    assert pct_lower < pct_upper
    assert pct_lower <= true_theta <= pct_upper


# ==============================================================================
# 4. SMALL SAMPLE STRESS TESTING
# ==============================================================================


def test_small_sample_behavior(synthetic_dgp_data):
    """Verify standard error increases as sample size shrinks (finite-sample SE inflation)."""
    X, T, Y, _ = synthetic_dgp_data

    sample_sizes = [40, 100, 300]
    report = evaluate_small_sample_behavior(
        X=X,
        T=T,
        Y=Y,
        sample_sizes=sample_sizes,
        n_repeats=2,
        model_y=Ridge(),
        model_t=Ridge(),
        nuisance_name="Ridge",
        random_state=42,
    )

    df = report.to_dataframe()
    assert not df.empty

    mean_se_by_size = df.groupby("sample_size")["std_error"].mean()

    # Smaller sample size must have higher standard error than larger sample size
    assert mean_se_by_size[40] > mean_se_by_size[300], (
        f"SE at N=40 ({mean_se_by_size[40]:.4f}) should exceed SE at N=300 ({mean_se_by_size[300]:.4f})"
    )


# ==============================================================================
# 5. MISSING VALUES HANDLING AND STRESS TESTING
# ==============================================================================


def test_handle_missing_confounders_imputation_strategies():
    """Verify missing confounder handlers: median, mean, indicator, and drop."""
    np.random.seed(42)
    X = np.random.randn(100, 4)
    # Inject 10 NaNs into col 1
    X[0:10, 1] = np.nan

    # 1. Median
    X_med, keep_med = handle_missing_confounders(X, strategy="median", add_indicators=False)
    assert not np.isnan(X_med).any()
    assert len(X_med) == 100
    assert keep_med.all()

    # 2. Mean
    X_mean, keep_mean = handle_missing_confounders(X, strategy="mean", add_indicators=False)
    assert not np.isnan(X_mean).any()
    assert len(X_mean) == 100

    # 3. Missing Indicator
    X_ind, _ = handle_missing_confounders(X, strategy="median", add_indicators=True)
    assert not np.isnan(X_ind).any()
    # 4 original features + 1 missingness indicator feature = 5 columns
    assert X_ind.shape[1] == 5

    # 4. Drop
    X_drop, keep_drop = handle_missing_confounders(X, strategy="drop")
    assert not np.isnan(X_drop).any()
    assert len(X_drop) == 90
    assert np.sum(keep_drop) == 90


def test_evaluate_missing_value_behavior(synthetic_dgp_data):
    """Verify that DML remains viable under missing values when using robust imputation."""
    X, T, Y, true_theta = synthetic_dgp_data

    report = evaluate_missing_value_behavior(
        X=X,
        T=T,
        Y=Y,
        missing_rates=[0.0, 0.10],
        strategies=["median", "indicator", "drop"],
        model_y=Ridge(),
        model_t=Ridge(),
        nuisance_name="Ridge",
        random_state=42,
    )

    df = report.to_dataframe()
    assert len(df) >= 3

    # All imputed runs should remain close to true theta
    for _, row in df.iterrows():
        assert abs(row["theta"] - true_theta) < 0.40
        assert row["std_error"] > 0.0


# ==============================================================================
# 6. WEAK TREATMENT VARIATION STRESS TESTING & DIAGNOSTICS
# ==============================================================================


def test_diagnose_treatment_variation():
    """Verify diagnostic identification of healthy vs weak treatment variation."""
    np.random.seed(42)
    n = 200

    # Healthy: T has significant independent residual variance
    T_healthy = np.random.randn(n) * 2.0
    pred_t_healthy = np.random.randn(n) * 0.5
    diag_healthy = diagnose_treatment_variation(T_healthy, pred_t_healthy)
    assert diag_healthy["status"] == "HEALTHY"
    assert diag_healthy["severity"] == "INFO"
    assert diag_healthy["warning_message"] is None

    # Weak: Pred_t almost exactly equals T (EVR ~ 0)
    T_weak = np.random.randn(n)
    pred_t_weak = T_weak + np.random.randn(n) * 0.001
    diag_weak = diagnose_treatment_variation(T_weak, pred_t_weak)
    assert diag_weak["status"] == "WEAK_IDENTIFICATION"
    assert diag_weak["severity"] == "CRITICAL"
    assert "dangerously low" in diag_weak["warning_message"]


def test_evaluate_weak_treatment_variation_scaling(synthetic_dgp_data):
    """Verify standard error explodes inversely as treatment variance collapses towards zero."""
    X, T, Y, _ = synthetic_dgp_data

    scales = [1.0, 0.5, 0.1]
    report, diagnostics = evaluate_weak_treatment_variation(
        X=X,
        T=T,
        Y=Y,
        variance_scales=scales,
        model_y=Ridge(),
        model_t=Ridge(),
        random_state=42,
    )

    assert len(diagnostics) == 3
    se_by_scale = {d["scale"]: d["std_error"] for d in diagnostics if "std_error" in d}

    # SE at scale 0.1 must be approximately 10x larger than SE at scale 1.0
    assert se_by_scale[0.1] > se_by_scale[1.0] * 5.0


# ==============================================================================
# 7. ROBUSTNESS REPORT PERSISTENCE AND FORMATTING
# ==============================================================================


def test_robustness_report_serialization_and_summary():
    """Verify RobustnessReport filtering, summary table formatting, and CSV/JSON export."""
    report = RobustnessReport()
    res1 = RobustnessResult(
        experiment="nuisance_comparison",
        nuisance_name="Model_A",
        n_folds=5,
        random_state=42,
        sample_size=500,
        theta=-1.5,
        std_error=0.2,
        t_stat=-7.5,
        p_value=1e-12,
        ci_lower=-1.892,
        ci_upper=-1.108,
        ci_width=0.784,
        ci_level=0.95,
        r2_y=0.45,
        rmse_y=0.8,
        mae_y=0.6,
        r2_t=0.55,
        rmse_t=0.7,
        mae_t=0.5,
        orthogonality_score=1e-12,
        orthogonality_corr=1e-12,
        var_residual_t=0.5,
        fit_time_seconds=0.12,
        extra_info={"note": "test"},
    )
    report.add(res1)

    # Test table formatting
    table = report.get_summary_table()
    assert "Model_A" in table
    assert "-1.5000" in table

    # Test file persistence
    with tempfile.TemporaryDirectory() as tmpdir:
        csv_path = os.path.join(tmpdir, "test_report.csv")
        json_path = os.path.join(tmpdir, "test_report.json")

        report.save_csv(csv_path)
        report.save_json(json_path)

        assert os.path.exists(csv_path)
        assert os.path.exists(json_path)

        # Validate JSON content
        with open(json_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        assert len(data) == 1
        assert data[0]["nuisance_name"] == "Model_A"
        assert data[0]["extra_note"] == "test"


# ==============================================================================
# 8. INTEGRATED DML ROBUSTNESS EVALUATOR ON REAL DATASET
# ==============================================================================


def test_evaluator_from_dataframe_real_retail_price(real_retail_data):
    """Verify DMLRobustnessEvaluator runs on real retail_price.csv data."""
    evaluator = DMLRobustnessEvaluator.from_dataframe(
        df=real_retail_data,
        random_state=42,
    )

    assert len(evaluator.X) > 400
    assert len(evaluator.T) == len(evaluator.X)
    assert len(evaluator.Y) == len(evaluator.X)

    # Run quick nuisance comparison with fast models
    fast_models = {
        "Ridge": (Ridge(alpha=1.0), Ridge(alpha=1.0)),
        "RF_Small": (
            RandomForestRegressor(n_estimators=15, max_depth=4, random_state=42),
            RandomForestRegressor(n_estimators=15, max_depth=4, random_state=42),
        ),
    }
    report_nuisance = evaluator.run_nuisance_comparison(candidates=fast_models, n_folds=3)
    assert len(report_nuisance.records) == 2

    # Check full report string formatting
    text_report = evaluator.format_full_report()
    assert "DOUBLE MACHINE LEARNING (DML) ROBUSTNESS" in text_report
    assert "NUISANCE MODEL COMPARISON" in text_report
