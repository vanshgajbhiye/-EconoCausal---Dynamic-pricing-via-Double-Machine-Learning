"""Unit and integration tests for Day 9 Streamlit Dashboard & Components."""

import ast
import os
import numpy as np
import pandas as pd
import pytest

from src.dashboard.components import (
    build_category_elasticity_chart,
    build_price_elasticity_curve_chart,
    build_scenario_waterfall_chart,
    calculate_catalog_kpis,
    compute_pricing_curves,
    simulate_competitor_response,
)


def test_compute_pricing_curves_basic():
    """Verify curve calculation returns expected columns and monotonic properties."""
    df = compute_pricing_curves(
        base_price=100.0,
        base_qty=20.0,
        unit_cost=60.0,
        elasticity=-1.5,
        p_min=50.0,
        p_max=150.0,
        n_points=100,
    )

    assert isinstance(df, pd.DataFrame)
    assert len(df) == 100
    expected_cols = {"price", "demand", "revenue", "profit", "margin_pct"}
    assert expected_cols.issubset(set(df.columns))

    # Demand should decrease monotonically as price increases when theta < 0
    demands = df["demand"].values
    assert np.all(np.diff(demands) <= 1e-6)

    # Prices start at p_min and end at p_max
    assert np.isclose(df["price"].iloc[0], 50.0)
    assert np.isclose(df["price"].iloc[-1], 150.0)

    # When price is below unit cost (60.0), profit must be negative
    below_cost = df[df["price"] < 60.0]
    assert len(below_cost) > 0
    assert np.all(below_cost["profit"] < 0)

    # When price is above unit cost, profit is positive
    above_cost = df[df["price"] > 60.0]
    assert len(above_cost) > 0
    assert np.all(above_cost["profit"] > 0)


def test_compute_pricing_curves_validation_errors():
    """Ensure invalid non-positive inputs raise ValueError."""
    with pytest.raises(ValueError, match="strictly positive"):
        compute_pricing_curves(base_price=0.0, base_qty=10.0, unit_cost=5.0, elasticity=-1.2)

    with pytest.raises(ValueError, match="strictly positive"):
        compute_pricing_curves(base_price=10.0, base_qty=-5.0, unit_cost=5.0, elasticity=-1.2)


def test_simulate_competitor_response_modes():
    """Verify game-theoretic competitor retaliation modes and cross-elasticity impact."""
    base_price = 100.0
    our_new_price = 115.0  # +15% price increase
    comp_base = 100.0
    base_qty = 50.0
    cost = 60.0
    own_theta = -1.4
    cross_theta = 0.40

    # 1. Static Mode (Competitor holds price)
    res_static = simulate_competitor_response(
        base_price=base_price,
        our_new_price=our_new_price,
        competitor_base_price=comp_base,
        base_qty=base_qty,
        unit_cost=cost,
        own_elasticity=own_theta,
        cross_elasticity=cross_theta,
        response_mode="static",
    )
    assert res_static["competitor_new_price"] == 100.0
    assert res_static["competitor_price_change_pct"] == 0.0
    assert "Static" in res_static["mode_label"]
    assert res_static["risk_level"] in ["Low Risk / Resilient", "Highly Profitable / Dominant", "Moderate Risk"]

    # 2. Exact Match Mode (Competitor matches our $115)
    res_match = simulate_competitor_response(
        base_price=base_price,
        our_new_price=our_new_price,
        competitor_base_price=comp_base,
        base_qty=base_qty,
        unit_cost=cost,
        own_elasticity=own_theta,
        cross_elasticity=cross_theta,
        response_mode="match",
    )
    assert np.isclose(res_match["competitor_new_price"], 115.0)
    # When competitor also raises price, cross-elasticity cushions our demand loss
    assert res_match["adjusted_qty"] > res_static["adjusted_qty"]

    # 3. Undercut 5% Mode
    res_undercut = simulate_competitor_response(
        base_price=base_price,
        our_new_price=our_new_price,
        competitor_base_price=comp_base,
        base_qty=base_qty,
        unit_cost=cost,
        own_elasticity=own_theta,
        cross_elasticity=cross_theta,
        response_mode="undercut_5",
    )
    assert np.isclose(res_undercut["competitor_new_price"], 115.0 * 0.95)

    # 4. Custom Mode
    res_custom = simulate_competitor_response(
        base_price=base_price,
        our_new_price=our_new_price,
        competitor_base_price=comp_base,
        base_qty=base_qty,
        unit_cost=cost,
        own_elasticity=own_theta,
        cross_elasticity=cross_theta,
        response_mode="custom",
        custom_comp_price=90.0,
    )
    assert np.isclose(res_custom["competitor_new_price"], 90.0)


def test_calculate_catalog_kpis():
    """Verify aggregate KPI portfolio computations."""
    data = {
        "baseline_price": [100.0, 50.0],
        "optimal_price": [120.0, 45.0],
        "baseline_qty": [10.0, 20.0],
        "projected_qty": [8.0, 24.0],
        "baseline_revenue": [1000.0, 1000.0],  # sum = 2000
        "projected_revenue": [960.0, 1080.0],  # sum = 2040 (+2.0%)
        "baseline_profit": [300.0, 300.0],     # sum = 600
        "projected_profit": [400.0, 336.0],    # sum = 736 (+22.67%)
        "action": ["MARKUP", "DISCOUNT"],
        "elasticity": [-1.2, -1.8],
    }
    df = pd.DataFrame(data)

    kpis = calculate_catalog_kpis(df)
    assert kpis["total_items"] == 2
    assert np.isclose(kpis["total_base_revenue"], 2000.0)
    assert np.isclose(kpis["total_optimized_revenue"], 2040.0)
    assert np.isclose(kpis["total_base_profit"], 600.0)
    assert np.isclose(kpis["total_optimized_profit"], 736.0)
    assert np.isclose(kpis["revenue_uplift_pct"], 0.02)
    assert np.isclose(kpis["profit_uplift_pct"], (736.0 - 600.0) / 600.0)
    assert kpis["action_counts"]["MARKUP"] == 1
    assert kpis["action_counts"]["DISCOUNT"] == 1
    assert kpis["margin_expansion_bps"] > 0


def test_plotly_chart_builders():
    """Verify that Plotly figures are constructed without error and have expected traces."""
    # 1. Price Curve Figure
    curve_df = compute_pricing_curves(
        base_price=80.0,
        base_qty=15.0,
        unit_cost=50.0,
        elasticity=-1.3,
        n_points=50,
    )
    fig_curve = build_price_elasticity_curve_chart(
        curve_df=curve_df,
        base_price=80.0,
        optimal_price=95.99,
        unit_cost=50.0,
        comp_price=82.0,
        p_lower=60.0,
        p_upper=100.0,
    )
    assert fig_curve is not None
    assert len(fig_curve.data) >= 3  # Demand, Profit, Revenue

    # 2. Scenario Waterfall Figure
    scenario_df = pd.DataFrame(
        {
            "strategy": ["Status Quo", "Causal Profit Maximizer"],
            "total_revenue": [100000.0, 95000.0],
            "total_profit": [30000.0, 42000.0],
        }
    )
    fig_scenario = build_scenario_waterfall_chart(scenario_df)
    assert fig_scenario is not None
    assert len(fig_scenario.data) == 2  # Revenue Bar, Profit Bar

    # 3. Category Elasticity Ranking Figure
    cat_df = pd.DataFrame(
        {
            "product_category_name": ["health_beauty", "computers_accessories"],
            "mean_elasticity": [-1.26, -1.83],
        }
    )
    fig_cat = build_category_elasticity_chart(cat_df)
    assert fig_cat is not None
    assert len(fig_cat.data) == 1
    assert fig_cat.data[0].orientation == "h"


def test_dashboard_app_syntax_and_structure():
    """Verify syntax tree and structural integrity of the main Streamlit app."""
    app_path = os.path.join("src", "dashboard", "app.py")
    assert os.path.exists(app_path), f"File {app_path} does not exist"

    with open(app_path, "r", encoding="utf-8") as f:
        code = f.read()

    # Must parse without syntax errors
    tree = ast.parse(code, filename=app_path)
    assert tree is not None


def test_dashboard_launcher_script():
    """Verify launcher script exists and has valid Python syntax."""
    launcher_path = os.path.join("scripts", "run_dashboard.py")
    assert os.path.exists(launcher_path)

    with open(launcher_path, "r", encoding="utf-8") as f:
        code = f.read()

    tree = ast.parse(code, filename=launcher_path)
    assert tree is not None


def test_docker_compose_includes_dashboard():
    """Verify docker-compose.yml defines the Streamlit dashboard service."""
    import yaml
    compose_path = "docker-compose.yml"
    assert os.path.exists(compose_path)

    with open(compose_path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    assert "econocausal-dashboard" in cfg["services"]
    dashboard_svc = cfg["services"]["econocausal-dashboard"]
    assert "8501:8501" in dashboard_svc["ports"]
