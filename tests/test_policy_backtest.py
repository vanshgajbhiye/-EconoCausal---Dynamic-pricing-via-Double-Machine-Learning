"""Automated unit and integration tests for Offline Pricing Policy Evaluation.

Milestone: Day 17 — Offline pricing policy evaluation.
Covers:
- Policy pricing generators (Current, Fixed Discount, Fixed Markup, Competitor Match, Non-Causal OLS, Causal)
- Direct Method counterfactual evaluation & factual identity
- Continuous kernel-smoothed IPW and Doubly Robust estimators
- Positivity and overlap diagnostics (ESS calculation, weight trimming)
- Vectorized bootstrap uncertainty engine (SEs, CIs, p-values)
- Realization friction and haircut scenario analysis
- Data partitioning (Random 80/20 and Temporal 2017/2018)
- End-to-end backtest orchestrator and CLI entrypoint
"""

import os
import tempfile
import numpy as np
import pytest

from experiments.policy_backtest import (
    CausalOptimizerPolicy,
    CompetitorMatchPolicy,
    CurrentPricesPolicy,
    FixedDiscountPolicy,
    FixedMarkupPolicy,
    NonCausalRegressionPolicy,
    OffPolicyEvaluator,
    OverlapDiagnostics,
    PolicyBacktester,
    PolicyEvaluationMetrics,
    PolicyUncertaintyEngine,
    RealizationFrictionEngine,
    main,
)
from src.data.load_data import generate_synthetic_retail_data
from src.pipeline import EconoCausalPipeline


@pytest.fixture(scope="module")
def sample_retail_df():
    """Create a realistic sample DataFrame for fast deterministic testing."""
    df = generate_synthetic_retail_data(n_samples=120, random_seed=42)
    # Ensure year column is present
    df["year"] = [2017] * 60 + [2018] * 60
    return df


@pytest.fixture(scope="module")
def fitted_pipeline(sample_retail_df):
    """Fit a small EconoCausalPipeline on sample data."""
    pipe = EconoCausalPipeline(
        n_splits=3,
        default_margin=0.30,
        min_margin=0.05,
        random_state=42,
    )
    pipe.fit(sample_retail_df, test_size=0.20, run_refutations=False)
    return pipe


# ==============================================================================
# 1. POLICY PRICING GENERATION TESTS
# ==============================================================================


def test_current_prices_policy(sample_retail_df):
    """CurrentPricesPolicy must return exact observed prices."""
    pol = CurrentPricesPolicy(price_col="unit_price")
    prices = pol.recommend_prices(sample_retail_df)
    assert len(prices) == len(sample_retail_df)
    np.testing.assert_allclose(prices.values, sample_retail_df["unit_price"].values)


def test_fixed_discount_policy_margin_protection(sample_retail_df):
    """FixedDiscountPolicy must discount prices by discount_pct but protect cost floors."""
    pol = FixedDiscountPolicy(discount_pct=0.10, default_margin=0.30, min_margin=0.05)
    prices = pol.recommend_prices(sample_retail_df)

    p0 = sample_retail_df["unit_price"].values
    unit_cost = p0 * 0.70
    cost_floor = unit_cost / 0.95

    # Prices must be at least cost floor
    assert np.all(prices.values >= cost_floor - 1e-4)
    # All prices must be less than or equal to baseline price
    assert np.all(prices.values <= p0 + 1e-4)


def test_fixed_markup_policy_bounds(sample_retail_df):
    """FixedMarkupPolicy must markup prices subject to max_markup_pct boundary."""
    pol = FixedMarkupPolicy(markup_pct=0.10, max_markup_pct=0.20)
    prices = pol.recommend_prices(sample_retail_df)

    p0 = sample_retail_df["unit_price"].values
    expected = p0 * 1.10
    np.testing.assert_allclose(prices.values, expected)

    # Test clamping when markup exceeds max_markup_pct
    pol_excess = FixedMarkupPolicy(markup_pct=0.35, max_markup_pct=0.20)
    prices_excess = pol_excess.recommend_prices(sample_retail_df)
    np.testing.assert_allclose(prices_excess.values, p0 * 1.20)


def test_competitor_match_policy(sample_retail_df):
    """CompetitorMatchPolicy must match competitor price or fall back to baseline with cost floor."""
    pol = CompetitorMatchPolicy(comp_col="comp_1", default_margin=0.30, min_margin=0.05)
    prices = pol.recommend_prices(sample_retail_df)

    p0 = sample_retail_df["unit_price"].values
    comp = sample_retail_df["comp_1"].values
    unit_cost = p0 * 0.70
    cost_floor = unit_cost / 0.95

    expected = np.maximum(comp, cost_floor)
    np.testing.assert_allclose(prices.values, expected)


def test_non_causal_regression_policy_fit_and_recommend(sample_retail_df):
    """NonCausalRegressionPolicy must fit naive regression and recommend markup when elasticity is positive or inelastic."""
    pol = NonCausalRegressionPolicy(max_markup_pct=0.20)
    assert not pol.is_fitted_

    confounders = ["comp_1", "product_score", "customers", "s"]
    pol.fit(sample_retail_df, confounder_cols=confounders)

    assert pol.is_fitted_
    assert isinstance(pol.naive_elasticity_, float)

    prices = pol.recommend_prices(sample_retail_df)
    assert len(prices) == len(sample_retail_df)
    assert np.all(prices.values > 0)


def test_causal_optimizer_policy(sample_retail_df, fitted_pipeline):
    """CausalOptimizerPolicy must use CATE elasticity and Amoroso-Robinson optimization."""
    pol = CausalOptimizerPolicy(pipeline=fitted_pipeline, objective="profit")
    prices = pol.recommend_prices(sample_retail_df)

    assert len(prices) == len(sample_retail_df)
    p0 = sample_retail_df["unit_price"].values
    # Prices must stay within allowable guardrail window [0.70x, 1.20x]
    assert np.all(prices.values >= p0 * 0.69)
    assert np.all(prices.values <= p0 * 1.21)


# ==============================================================================
# 2. OFF-POLICY EVALUATION (OPE) ENGINE TESTS
# ==============================================================================


def test_direct_method_logging_policy_identity():
    """Direct Method evaluating logging policy at base prices must reproduce factual outcomes with 0% uplift."""
    evaluator = OffPolicyEvaluator(default_margin=0.30)
    p0 = np.array([50.0, 100.0, 150.0])
    q0 = np.array([10.0, 20.0, 30.0])
    theta = np.array([-1.2, -1.5, -1.0])

    res = evaluator.evaluate_direct_method(
        policy_prices=p0,
        base_prices=p0,
        base_qtys=q0,
        elasticities=theta,
        policy_name="Status Quo",
    )

    expected_rev = 50.0 * 10 + 100.0 * 20 + 150.0 * 30  # 500 + 2000 + 4500 = 7000
    expected_prof = expected_rev * 0.30  # 2100

    assert pytest.approx(res.total_revenue, rel=1e-4) == expected_rev
    assert pytest.approx(res.total_profit, rel=1e-4) == expected_prof
    assert pytest.approx(res.revenue_uplift_pct, abs=1e-4) == 0.0
    assert pytest.approx(res.profit_uplift_pct, abs=1e-4) == 0.0
    assert pytest.approx(res.gross_margin, rel=1e-4) == 0.30


def test_direct_method_demand_decay_on_elastic_markup():
    """Raising price on elastic good must decrease demand volume."""
    evaluator = OffPolicyEvaluator(default_margin=0.30)
    p0 = np.array([100.0])
    q0 = np.array([100.0])
    p1 = np.array([120.0])  # +20% price hike
    theta = np.array([-1.5])  # Elastic demand

    res = evaluator.evaluate_direct_method(
        policy_prices=p1,
        base_prices=p0,
        base_qtys=q0,
        elasticities=theta,
    )

    # Q(120) = 100 * (1.2)^(-1.5) = 100 / 1.3145 = ~76.07
    expected_q1 = 100.0 * (1.2 ** -1.5)
    assert pytest.approx(res.total_units, rel=1e-3) == expected_q1
    assert res.units_uplift_pct < 0.0


def test_continuous_weights_logging_policy_uniformity():
    """For logging policy, continuous IPW normalized weights must be identically uniform with ESS ratio = 1.0."""
    evaluator = OffPolicyEvaluator()
    p0 = np.array([25.0, 50.0, 75.0, 100.0])
    prop_locs = np.log(p0)
    prop_scale = 0.20

    raw_w, norm_w, _ = evaluator.compute_continuous_weights(
        policy_prices=p0,
        base_prices=p0,
        propensity_locs=prop_locs,
        propensity_scale=prop_scale,
    )

    np.testing.assert_allclose(norm_w, 0.25)
    diag = evaluator.compute_overlap_diagnostics(
        policy_prices=p0,
        base_prices=p0,
        normalized_weights=norm_w,
        policy_name="Current",
    )
    assert pytest.approx(diag.ess_ratio, rel=1e-4) == 1.0
    assert diag.positivity_status == "SATISFIED"
    assert diag.extrapolation_risk == "LOW"


def test_continuous_ipw_and_doubly_robust_execution():
    """Continuous IPW and Doubly Robust estimators must return valid bounded financial metrics."""
    evaluator = OffPolicyEvaluator(default_margin=0.30)
    p0 = np.array([40.0, 80.0, 120.0, 160.0])
    q0 = np.array([15.0, 25.0, 10.0, 5.0])
    p1 = p0 * 1.05  # +5% policy
    theta = np.array([-1.2, -1.4, -1.1, -1.3])
    prop_locs = np.log(p0)
    prop_scale = 0.15

    ipw_res, diag = evaluator.evaluate_continuous_ipw(
        policy_prices=p1,
        base_prices=p0,
        base_qtys=q0,
        propensity_locs=prop_locs,
        propensity_scale=prop_scale,
        policy_name="IPW Test",
    )

    assert isinstance(ipw_res, PolicyEvaluationMetrics)
    assert ipw_res.total_revenue > 0
    assert ipw_res.total_profit > 0
    assert 0.0 <= ipw_res.gross_margin <= 1.0
    assert isinstance(diag, OverlapDiagnostics)

    dr_res = evaluator.evaluate_doubly_robust(
        policy_prices=p1,
        base_prices=p0,
        base_qtys=q0,
        elasticities=theta,
        propensity_locs=prop_locs,
        propensity_scale=prop_scale,
        policy_name="DR Test",
    )
    assert isinstance(dr_res, PolicyEvaluationMetrics)
    assert dr_res.total_revenue > 0
    assert dr_res.total_profit > 0


def test_local_matching_evaluator():
    """Local matching must identify observations within epsilon band."""
    evaluator = OffPolicyEvaluator(match_epsilon=0.05)
    p0 = np.array([100.0, 200.0, 300.0])
    q0 = np.array([10.0, 20.0, 30.0])
    p1 = np.array([102.0, 250.0, 295.0])  # item 0 is +2% (match), item 1 is +25% (no), item 2 is -1.67% (match)

    res = evaluator.evaluate_local_matching(
        policy_prices=p1,
        base_prices=p0,
        base_qtys=q0,
        policy_name="Match Test",
    )

    assert res["matched_items"] == 2
    assert res["total_items"] == 3
    assert pytest.approx(res["matched_fraction"], rel=1e-4) == 2 / 3


# ==============================================================================
# 3. UNCERTAINTY QUANTIFICATION & BOOTSTRAP ENGINE TESTS
# ==============================================================================


def test_policy_uncertainty_bootstrap_engine():
    """PolicyUncertaintyEngine must compute finite standard errors, 95% CIs, and p-values."""
    engine = PolicyUncertaintyEngine(n_bootstraps=50, random_state=42)
    evaluator = OffPolicyEvaluator()

    p0 = np.array([50.0, 100.0, 150.0, 200.0])
    q0 = np.array([20.0, 15.0, 10.0, 5.0])
    theta = np.array([-1.2, -1.3, -1.1, -1.4])
    costs = p0 * 0.70

    policy_map = {
        "Current Prices (Status Quo)": p0,
        "Proposed Causal Policy (DML + CATE)": p0 * 1.10,
        "Fixed 10% Discount": p0 * 0.90,
    }

    boot_out = engine.run_bootstrap_evaluation(
        evaluator=evaluator,
        policy_prices_map=policy_map,
        base_prices=p0,
        base_qtys=q0,
        elasticities=theta,
        unit_costs=costs,
    )

    assert "Current Prices (Status Quo)" in boot_out
    assert "Proposed Causal Policy (DML + CATE)" in boot_out

    causal_info = boot_out["Proposed Causal Policy (DML + CATE)"]
    assert causal_info["profit_se"] > 0
    assert causal_info["profit_ci"][0] < causal_info["profit_ci"][1]
    assert 0.0 <= causal_info["p_value_vs_current"] <= 1.0


# ==============================================================================
# 4. REALIZATION FRICTION & HAIRCUT SCENARIOS TESTS
# ==============================================================================


def test_realization_friction_engine():
    """RealizationFrictionEngine must discount simulated profit gain progressively across 100%, 70%, 50%."""
    base_prof = 10000.0
    sim_prof = 15000.0  # +5,000 gain
    base_rev = 30000.0
    sim_rev = 32000.0

    scenarios = RealizationFrictionEngine.compute_haircut_scenarios(
        base_profit=base_prof,
        simulated_profit=sim_prof,
        base_revenue=base_rev,
        simulated_revenue=sim_rev,
    )

    assert "100% (Pure Simulation)" in scenarios
    assert "70% (Moderate Haircut)" in scenarios
    assert "50% (Conservative Haircut)" in scenarios

    h100 = scenarios["100% (Pure Simulation)"]
    h70 = scenarios["70% (Moderate Haircut)"]
    h50 = scenarios["50% (Conservative Haircut)"]

    assert pytest.approx(h100["profit_gain"]) == 5000.0
    assert pytest.approx(h70["profit_gain"]) == 3500.0
    assert pytest.approx(h50["profit_gain"]) == 2500.0

    assert h100["realized_profit"] > h70["realized_profit"] > h50["realized_profit"]


# ==============================================================================
# 5. DATA PARTITIONING & ZERO-LEAKAGE TESTS
# ==============================================================================


def test_backtester_partition_data_random(sample_retail_df):
    """Random partition must produce disjoint train and holdout splits."""
    backtester = PolicyBacktester(random_state=42)
    train_df, holdout_df = backtester.partition_data(sample_retail_df, split_type="random", test_size=0.25)

    assert len(train_df) + len(holdout_df) == len(sample_retail_df)
    assert len(set(train_df.index).intersection(set(holdout_df.index))) == 0


def test_backtester_partition_data_temporal(sample_retail_df):
    """Temporal partition must separate earlier year (train) from later year (holdout)."""
    backtester = PolicyBacktester()
    train_df, holdout_df = backtester.partition_data(sample_retail_df, split_type="temporal")

    assert np.all(train_df["year"] == 2017)
    assert np.all(holdout_df["year"] == 2018)


# ==============================================================================
# 6. END-TO-END BACKTEST ORCHESTRATOR & CLI TESTS
# ==============================================================================


def test_backtest_run_and_export(sample_retail_df, fitted_pipeline):
    """End-to-end backtest execution on sample data with CSV, JSON, and Markdown export."""
    with tempfile.TemporaryDirectory() as tmp_dir:
        backtester = PolicyBacktester(
            n_splits=3,
            n_bootstraps=30,
            discount_pct=0.10,
            random_state=42,
        )

        results = backtester.run_backtest(
            df=sample_retail_df,
            split_type="random",
            test_size=0.25,
            pipeline_model=fitted_pipeline,
            verbose=False,
        )

        assert "Current Prices (Status Quo)" in results
        assert "Proposed Causal Policy (DML + CATE)" in results
        assert "Fixed 10% Discount" in results
        assert "Simple Non-Causal Baseline (Naive OLS)" in results

        # Summary table formatting
        table_str = backtester.get_summary_table()
        assert "Policy Strategy" in table_str
        assert "Proposed Causal Policy" in table_str

        # DataFrame export
        df_summary = backtester.to_dataframe()
        assert not df_summary.empty
        assert "Profit Uplift (%)" in df_summary.columns

        # Save files
        csv_path, json_path = backtester.save_results(output_dir=tmp_dir)
        assert os.path.exists(csv_path)
        assert os.path.exists(json_path)

        # Markdown report
        md_path = os.path.join(tmp_dir, "test_report.md")
        saved_md = backtester.generate_markdown_report(filepath=md_path)
        assert os.path.exists(saved_md)


def test_cli_entrypoint(sample_retail_df):
    """Test CLI execution using main function with custom args."""
    with tempfile.TemporaryDirectory() as tmp_dir:
        csv_data_path = os.path.join(tmp_dir, "test_retail.csv")
        sample_retail_df.to_csv(csv_data_path, index=False)

        report_path = os.path.join(tmp_dir, "cli_report.md")
        argv = [
            "--data", csv_data_path,
            "--split-type", "random",
            "--test-size", "0.20",
            "--n-bootstraps", "20",
            "--output-dir", tmp_dir,
            "--export-report", report_path,
        ]

        exit_code = main(argv)
        assert exit_code == 0
        assert os.path.exists(os.path.join(tmp_dir, "policy_backtest_results.csv"))
        assert os.path.exists(os.path.join(tmp_dir, "policy_backtest_results.json"))
