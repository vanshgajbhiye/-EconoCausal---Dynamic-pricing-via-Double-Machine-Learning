"""Unit and integration tests for Continuous Causal MLOps and Drift Monitoring (Day 10).

Validates:
- Population Stability Index (PSI) calculation on continuous and categorical variables
- Two-sample Kolmogorov-Smirnov test and Wasserstein distance computation
- Confounder and elasticity distribution drift detection on real retail data
- Automated refutation re-auditing on production data
- Retraining trigger decision tiers (HEALTHY, WARNING, RETRAIN_RECOMMENDED, CRITICAL)
- REST API microservice /monitoring/drift endpoint
- Prometheus telemetry drift gauges
- CLI audit-drift subcommand
- Plotly drift dashboard visualizers
"""

import json
import os
import tempfile
import numpy as np
import pandas as pd
import pytest
from starlette.testclient import TestClient

from src.api.main import create_app
from src.api.metrics import (
    CONFOUNDER_MAX_PSI,
    CONFOUNDER_MEAN_PSI,
    DRIFT_HEALTH_STATUS,
    export_metrics,
    record_drift_audit,
)
from src.api.service import CausalPricingService
from src.cli import main as cli_main
from src.dashboard.components import (
    build_confounder_psi_chart,
    build_distribution_comparison_chart,
    build_elasticity_shift_chart,
)
from src.data.load_data import load_retail_price_dataset
from src.monitoring.drift import (
    CausalDriftDetector,
    CausalDriftReport,
    FeatureDriftResult,
    calculate_population_stability_index,
)
from src.pipeline import EconoCausalPipeline


@pytest.fixture(scope="module")
def retail_data():
    """Load baseline retail price optimization dataset."""
    data_path = os.path.join("data", "raw", "retail_price.csv")
    assert os.path.exists(data_path), f"Dataset not found at {data_path}"
    return load_retail_price_dataset(data_path)


@pytest.fixture(scope="module")
def fitted_pipeline(retail_data):
    """Load or train pipeline model fixture."""
    model_path = os.path.join("models", "econocausal_pipeline.joblib")
    if os.path.exists(model_path):
        return EconoCausalPipeline.load(model_path)
    pipeline = EconoCausalPipeline(n_splits=3, random_state=42)
    pipeline.fit(retail_data, test_size=0.20, run_refutations=False)
    return pipeline


@pytest.fixture
def api_client():
    """FastAPI TestClient fixture."""
    CausalPricingService.reset_instance()
    app = create_app()
    return TestClient(app)


# ==============================================================================
# 1. MATHEMATICAL PSI & DRIFT STATISTIC TESTS
# ==============================================================================


def test_psi_identical_distributions():
    """Identical distributions must yield exactly 0.0 PSI."""
    arr = np.random.RandomState(42).normal(loc=50.0, scale=10.0, size=500)
    psi = calculate_population_stability_index(arr, arr)
    assert psi == 0.0, f"Expected 0.0 for identical data, got {psi}"


def test_psi_continuous_monotonic_shift():
    """Increasing distributional divergence must monotonically increase PSI."""
    rng = np.random.RandomState(42)
    base = rng.normal(loc=100.0, scale=15.0, size=1000)
    mild_shift = rng.normal(loc=103.0, scale=15.0, size=1000)
    moderate_shift = rng.normal(loc=108.0, scale=15.0, size=1000)
    severe_shift = rng.normal(loc=125.0, scale=15.0, size=1000)

    psi_mild = calculate_population_stability_index(base, mild_shift)
    psi_mod = calculate_population_stability_index(base, moderate_shift)
    psi_severe = calculate_population_stability_index(base, severe_shift)

    assert 0.0 <= psi_mild < psi_mod < psi_severe
    assert psi_mild < 0.10, f"Mild shift should be < 0.10, got {psi_mild}"
    assert psi_severe >= 0.25, f"Severe shift should be >= 0.25, got {psi_severe}"


def test_psi_categorical_frequencies():
    """Categorical PSI tracks category proportion changes and new categories."""
    base_cats = ["electronics"] * 60 + ["fashion"] * 30 + ["home"] * 10
    stable_cats = ["electronics"] * 58 + ["fashion"] * 31 + ["home"] * 11
    shifted_cats = ["electronics"] * 10 + ["fashion"] * 20 + ["home"] * 70 + ["toys"] * 20

    psi_stable = calculate_population_stability_index(base_cats, stable_cats, is_categorical=True)
    psi_shifted = calculate_population_stability_index(base_cats, shifted_cats, is_categorical=True)

    assert psi_stable < 0.05
    assert psi_shifted > 0.25


def test_psi_edge_cases():
    """Robustness to edge cases: constant values, empty arrays, NaNs."""
    # Empty
    assert calculate_population_stability_index([], []) == 0.0

    # Constant arrays
    const_arr = np.full(100, 42.0)
    assert calculate_population_stability_index(const_arr, const_arr) == 0.0

    # Arrays with NaNs
    arr_with_nan1 = np.array([1.0, 2.0, np.nan, 4.0, 5.0])
    arr_with_nan2 = np.array([1.0, 2.0, 3.0, 4.0, np.nan])
    psi = calculate_population_stability_index(arr_with_nan1, arr_with_nan2)
    assert psi >= 0.0


def test_detect_feature_drift_metrics():
    """Verify KS statistic, p-value, Wasserstein distance, and mean shift calculation."""
    detector = CausalDriftDetector(psi_warning_threshold=0.10, psi_alert_threshold=0.25)
    s_base = pd.Series([10.0, 12.0, 14.0, 16.0, 18.0] * 20, name="price")
    s_target = pd.Series([15.0, 17.0, 19.0, 21.0, 23.0] * 20, name="price")

    result = detector.detect_feature_drift(s_base, s_target)

    assert isinstance(result, FeatureDriftResult)
    assert result.feature_name == "price"
    assert result.psi > 0.0
    assert result.ks_stat is not None and result.ks_stat > 0.0
    assert result.p_value is not None
    assert result.wasserstein_dist is not None and result.wasserstein_dist > 0.0
    assert result.baseline_mean is not None
    assert result.target_mean is not None
    assert result.mean_shift_pct > 0.0


# ==============================================================================
# 2. CONFOUNDER & ELASTICITY DRIFT DETECTOR TESTS
# ==============================================================================


def test_confounder_audit_baseline_vs_baseline(retail_data):
    """Auditing retail dataset against itself must evaluate as HEALTHY with 0 PSI."""
    detector = CausalDriftDetector()
    results = detector.audit_confounders(retail_data, retail_data)

    assert len(results) > 0
    for name, res in results.items():
        assert res.psi == 0.0, f"Feature {name} had non-zero PSI against itself: {res.psi}"
        assert res.drift_level == "STABLE"
        assert not res.drift_detected


def test_confounder_audit_synthetic_shock(retail_data):
    """Competitor price cut of 35% must trigger significant drift in competitor features."""
    detector = CausalDriftDetector()
    shocked_data = retail_data.copy()
    shocked_data["comp_1"] = shocked_data["comp_1"] * 0.65
    shocked_data["comp_2"] = shocked_data["comp_2"] * 0.65

    results = detector.audit_confounders(retail_data, shocked_data)

    assert results["comp_1"].psi > 0.10
    assert results["comp_1"].drift_detected
    assert results["comp_2"].drift_detected
    # Untouched feature should remain stable
    assert results["product_score"].psi == 0.0


def test_elasticity_drift_detection(retail_data, fitted_pipeline):
    """Verify CATE elasticity shift detection under price shocks."""
    detector = CausalDriftDetector()
    shocked_data = retail_data.copy()
    shocked_data["comp_1"] = shocked_data["comp_1"] * 0.60

    elast_drift = detector.audit_elasticity_drift(
        pipeline=fitted_pipeline,
        baseline_df=retail_data,
        target_df=shocked_data,
    )

    assert "psi" in elast_drift
    assert "baseline_mean" in elast_drift
    assert "target_mean" in elast_drift
    assert "delta_mean" in elast_drift
    assert "ks_stat" in elast_drift
    assert "p_value" in elast_drift
    assert isinstance(elast_drift["regime_changes"], list)


def test_evaluate_health_decision_tiers():
    """Verify all 4 MLOps health decision tiers."""
    detector = CausalDriftDetector()

    # Tier 1: HEALTHY
    stable_results = {
        "comp_1": FeatureDriftResult("comp_1", psi=0.02, drift_level="STABLE", drift_detected=False),
        "fp1": FeatureDriftResult("fp1", psi=0.03, drift_level="STABLE", drift_detected=False),
    }
    status, rec = detector.evaluate_health(stable_results)
    assert status == "HEALTHY"

    # Tier 2: WARNING
    warning_results = {
        "comp_1": FeatureDriftResult("comp_1", psi=0.14, drift_level="MODERATE_DRIFT", drift_detected=True),
        "fp1": FeatureDriftResult("fp1", psi=0.03, drift_level="STABLE", drift_detected=False),
    }
    status, rec = detector.evaluate_health(warning_results)
    assert status == "WARNING"

    # Tier 3: RETRAIN_RECOMMENDED
    retrain_results = {
        "comp_1": FeatureDriftResult("comp_1", psi=0.35, drift_level="SIGNIFICANT_DRIFT", drift_detected=True),
        "fp1": FeatureDriftResult("fp1", psi=0.28, drift_level="SIGNIFICANT_DRIFT", drift_detected=True),
    }
    status, rec = detector.evaluate_health(retrain_results)
    assert status == "RETRAIN_RECOMMENDED"

    # Tier 4: CRITICAL
    refutation_fail = {"pass_rate": 0.33, "tests_passed": 1, "total_tests": 3}
    status, rec = detector.evaluate_health(stable_results, refutation_results=refutation_fail)
    assert status == "CRITICAL"


def test_full_report_generation_and_export(retail_data, fitted_pipeline):
    """Verify end-to-end report generation and JSON persistence."""
    detector = CausalDriftDetector()
    report = detector.generate_report(
        baseline_df=retail_data,
        target_df=retail_data,
        pipeline=fitted_pipeline,
        run_refutations=False,
    )

    assert isinstance(report, CausalDriftReport)
    assert report.overall_status == "HEALTHY"
    assert report.max_confounder_psi == 0.0

    summary_str = report.summary()
    assert "[DRIFT REPORT]" in summary_str
    assert "HEALTHY" in summary_str

    report_dict = report.to_dict()
    assert report_dict["overall_status"] == "HEALTHY"
    assert "comp_1" in report_dict["confounders"]

    # Test file saving
    with tempfile.TemporaryDirectory() as tmp_dir:
        json_path = os.path.join(tmp_dir, "test_report.json")
        report.save_json(json_path)
        assert os.path.exists(json_path)

        with open(json_path, "r", encoding="utf-8") as f:
            loaded_data = json.load(f)
        assert loaded_data["overall_status"] == "HEALTHY"


# ==============================================================================
# 3. FASTAPI REST API ENDPOINT TESTS
# ==============================================================================


def test_api_drift_endpoint_healthy(api_client, retail_data):
    """POST /monitoring/drift returns HEALTHY report for representative unshifted batch."""
    sample_items = retail_data.sample(frac=0.75, random_state=42).to_dict(orient="records")
    payload = {
        "items": sample_items,
        "run_refutations": False,
        "psi_warning_threshold": 0.10,
        "psi_alert_threshold": 0.25,
    }

    response = api_client.post("/monitoring/drift", json=payload)
    assert response.status_code == 200

    data = response.json()
    assert data["overall_status"] in ["HEALTHY", "WARNING"]
    assert "max_confounder_psi" in data
    assert "mean_confounder_psi" in data
    assert "confounders" in data
    assert "audit_latency_ms" in data
    assert data["audit_latency_ms"] > 0.0


def test_api_drift_endpoint_shock_retrain(api_client, retail_data):
    """POST /monitoring/drift detects synthetic covariate shock."""
    shocked_items = retail_data.head(50).copy()
    shocked_items["comp_1"] = shocked_items["comp_1"] * 0.50
    shocked_items["comp_2"] = shocked_items["comp_2"] * 0.50
    shocked_items["comp_3"] = shocked_items["comp_3"] * 0.50

    payload = {
        "items": shocked_items.to_dict(orient="records"),
        "run_refutations": False,
        "psi_warning_threshold": 0.10,
        "psi_alert_threshold": 0.25,
    }

    response = api_client.post("/monitoring/drift", json=payload)
    assert response.status_code == 200

    data = response.json()
    assert data["overall_status"] in ["WARNING", "RETRAIN_RECOMMENDED"]
    assert data["max_confounder_psi"] > 0.10


# ==============================================================================
# 4. PROMETHEUS TELEMETRY TESTS
# ==============================================================================


def test_prometheus_drift_telemetry():
    """Verify Prometheus metrics recording and exposition for drift."""
    record_drift_audit(
        max_psi=0.28,
        mean_psi=0.12,
        status_str="RETRAIN_RECOMMENDED",
        refutation_pass_rate=1.0,
    )

    resp = export_metrics()
    assert resp.status_code == 200
    metrics_text = resp.body.decode("utf-8")

    assert "econocausal_confounder_max_psi" in metrics_text
    assert "econocausal_confounder_mean_psi" in metrics_text
    assert "econocausal_drift_health_status" in metrics_text


# ==============================================================================
# 5. CLI AUDIT-DRIFT SUBCOMMAND TESTS
# ==============================================================================


def test_cli_audit_drift_command():
    """Verify CLI audit-drift subcommand execution."""
    data_path = os.path.join("data", "raw", "retail_price.csv")
    model_path = os.path.join("models", "econocausal_pipeline.joblib")

    with tempfile.TemporaryDirectory() as tmp_dir:
        report_out = os.path.join(tmp_dir, "drift.json")
        exit_code = cli_main(
            [
                "audit-drift",
                "--baseline",
                data_path,
                "--target",
                data_path,
                "--model",
                model_path,
                "--output",
                report_out,
            ]
        )

        assert exit_code == 0
        assert os.path.exists(report_out)


# ==============================================================================
# 6. DASHBOARD PLOTLY VISUALIZERS TESTS
# ==============================================================================


def test_dashboard_drift_charts(retail_data):
    """Verify Plotly chart generation for drift monitor."""
    confounders_sample = {
        "comp_1": {"psi": 0.05, "drift_level": "STABLE", "ks_stat": 0.02, "p_value": 0.9},
        "fp1": {"psi": 0.18, "drift_level": "MODERATE_DRIFT", "ks_stat": 0.15, "p_value": 0.03},
        "customers": {"psi": 0.32, "drift_level": "SIGNIFICANT_DRIFT", "ks_stat": 0.28, "p_value": 0.001},
    }

    # 1. PSI bar chart
    fig_psi = build_confounder_psi_chart(confounders_sample)
    assert fig_psi is not None
    assert len(fig_psi.data) == 1
    assert len(fig_psi.data[0].x) == 3

    # 2. Distribution comparison chart
    s1 = retail_data["comp_1"]
    s2 = retail_data["comp_1"] * 0.8
    fig_dist = build_distribution_comparison_chart(s1, s2, "comp_1")
    assert fig_dist is not None
    assert len(fig_dist.data) == 2

    # 3. Elasticity shift chart
    e1 = np.random.normal(-1.25, 0.2, 100)
    e2 = np.random.normal(-1.45, 0.2, 100)
    fig_elast = build_elasticity_shift_chart(e1, e2)
    assert fig_elast is not None
    assert len(fig_elast.data) == 2
