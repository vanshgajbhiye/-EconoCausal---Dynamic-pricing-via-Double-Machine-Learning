"""Unit and integration tests for DynamicPricingOptimizer & Uplift Simulation Engine."""

import numpy as np
import pandas as pd
import pytest

from src.models.optimizer import DynamicPricingOptimizer, ItemOptimizationResult


class TestDynamicPricingOptimizerInit:
    """Test initialization and input parameter validations."""

    def test_default_initialization(self):
        opt = DynamicPricingOptimizer()
        assert opt.default_margin == 0.30
        assert opt.min_price_change_pct == -0.30
        assert opt.max_price_change_pct == 0.20
        assert opt.min_margin == 0.05
        assert not opt.charm_pricing
        assert opt.charm_ending == 0.99

    def test_invalid_price_bounds(self):
        with pytest.raises(ValueError, match="min_price_change_pct"):
            DynamicPricingOptimizer(min_price_change_pct=0.20, max_price_change_pct=-0.10)

        with pytest.raises(ValueError, match="min_price_change_pct"):
            DynamicPricingOptimizer(min_price_change_pct=0.10, max_price_change_pct=0.10)

    def test_invalid_margin(self):
        with pytest.raises(ValueError, match="default_margin"):
            DynamicPricingOptimizer(default_margin=0.0)

        with pytest.raises(ValueError, match="default_margin"):
            DynamicPricingOptimizer(default_margin=1.0)

        with pytest.raises(ValueError, match="min_margin"):
            DynamicPricingOptimizer(min_margin=-0.05)


class TestOptimalPriceCalculation:
    """Test mathematical and microeconomic correctness of optimal price derivation."""

    def test_analytical_profit_maximization_unconstrained(self):
        # Marginal cost = 60, baseline price = 100, elasticity = -2.0
        # Amoroso-Robinson: P* = c * [theta / (theta + 1)] = 60 * (-2 / -1) = 120
        opt = DynamicPricingOptimizer(
            min_price_change_pct=-0.40,
            max_price_change_pct=0.40,  # Allows up to 140
        )
        p_opt, flag = opt.calculate_optimal_price(
            base_price=100.0,
            elasticity=-2.0,
            unit_cost=60.0,
            objective="profit",
        )
        assert pytest.approx(p_opt, 1e-4) == 120.0
        assert flag == "unconstrained_optimal"

        # Verify that profit at 120 is strictly greater than at 100, 110, or 130
        q0 = 100.0
        c = 60.0
        prof_100 = (100.0 - c) * q0 * (100.0 / 100.0) ** (-2.0)
        prof_110 = (110.0 - c) * q0 * (110.0 / 100.0) ** (-2.0)
        prof_120 = (120.0 - c) * q0 * (120.0 / 100.0) ** (-2.0)
        prof_130 = (130.0 - c) * q0 * (130.0 / 100.0) ** (-2.0)

        assert prof_120 > prof_100
        assert prof_120 > prof_110
        assert prof_120 > prof_130

    def test_profit_maximization_upper_bound_clipping(self):
        # Elasticity = -1.5, cost = 60.
        # Unconstrained P* = 60 * (-1.5 / -0.5) = 180 (+80% markup).
        # With max_price_change_pct = +20%, upper bound is 120.
        opt = DynamicPricingOptimizer(max_price_change_pct=0.20)
        p_opt, flag = opt.calculate_optimal_price(
            base_price=100.0,
            elasticity=-1.5,
            unit_cost=60.0,
            objective="profit",
        )
        assert pytest.approx(p_opt, 1e-4) == 120.0
        assert flag == "max_bound_clipped"

    def test_profit_maximization_lower_bound_clipping(self):
        # Elasticity = -4.0, cost = 40, base_price = 100.
        # Unconstrained P* = 40 * (-4 / -3) = 53.33 (-46.7% discount).
        # With min_price_change_pct = -25%, lower bound is 75.
        opt = DynamicPricingOptimizer(min_price_change_pct=-0.25)
        p_opt, flag = opt.calculate_optimal_price(
            base_price=100.0,
            elasticity=-4.0,
            unit_cost=40.0,
            objective="profit",
        )
        assert pytest.approx(p_opt, 1e-4) == 75.0
        assert flag == "min_bound_clipped"

    def test_inelastic_goods_profit_maximization(self):
        # Inelastic good: elasticity = -0.8 (theta >= -1.0)
        # Optimal strategy is to increase price to upper guardrail (+20%)
        opt = DynamicPricingOptimizer(max_price_change_pct=0.20)
        p_opt, flag = opt.calculate_optimal_price(
            base_price=100.0,
            elasticity=-0.8,
            objective="profit",
        )
        assert pytest.approx(p_opt, 1e-4) == 120.0
        assert flag == "inelastic_upper_bound"

    def test_revenue_optimization_objective(self):
        opt = DynamicPricingOptimizer(min_price_change_pct=-0.20, max_price_change_pct=0.15)

        # 1. Elastic good (theta = -1.8 < -1.0): lowering price maximizes revenue
        p_rev_elastic, flag_el = opt.calculate_optimal_price(
            base_price=100.0,
            elasticity=-1.8,
            objective="revenue",
        )
        assert pytest.approx(p_rev_elastic, 1e-4) == 80.0
        assert flag_el == "elastic_revenue_lower_bound"

        # 2. Inelastic good (theta = -0.5 > -1.0): raising price maximizes revenue
        p_rev_inelastic, flag_in = opt.calculate_optimal_price(
            base_price=100.0,
            elasticity=-0.5,
            objective="revenue",
        )
        assert pytest.approx(p_rev_inelastic, 1e-4) == 115.0
        assert flag_in == "inelastic_revenue_upper_bound"

        # 3. Unit elastic (theta = -1.0): neutral
        p_unit, flag_u = opt.calculate_optimal_price(
            base_price=100.0,
            elasticity=-1.0,
            objective="revenue",
        )
        assert pytest.approx(p_unit, 1e-4) == 100.0
        assert flag_u == "unit_elastic_neutral"

    def test_cost_floor_enforcement(self):
        # Even with deep discounts, price must satisfy P >= cost / (1 - min_margin)
        opt = DynamicPricingOptimizer(min_price_change_pct=-0.50, min_margin=0.10)
        # base_price = 100, cost = 80. Floor = 80 / 0.9 = 88.89
        p_opt, _ = opt.calculate_optimal_price(
            base_price=100.0,
            elasticity=-5.0,
            unit_cost=80.0,
            objective="profit",
        )
        assert p_opt >= 88.88

    def test_competitor_guardrail(self):
        # Competitor is at 90.0, tolerance is +/- 10% -> bounds [81, 99]
        opt = DynamicPricingOptimizer(max_price_change_pct=0.50)
        p_opt, _ = opt.calculate_optimal_price(
            base_price=100.0,
            elasticity=-0.5,  # Inelastic would normally jump to 150
            comp_price=90.0,
            comp_guardrail_pct=0.10,
        )
        assert pytest.approx(p_opt, 1e-4) == 99.0

    def test_charm_pricing(self):
        opt = DynamicPricingOptimizer(charm_pricing=True, charm_ending=0.99)
        p_opt, _ = opt.calculate_optimal_price(
            base_price=100.0,
            elasticity=-2.0,
            unit_cost=60.0,
        )
        # Optimal unconstrained is 120.0 -> rounded to 119.99 or 120.99
        cents = round(p_opt - int(p_opt), 2)
        assert cents == 0.99


class TestUpliftSimulation:
    """Test counterfactual quantity, revenue, and profit simulation."""

    def test_simulate_item_uplift_elastic_discount(self):
        opt = DynamicPricingOptimizer()
        # Price dropped 10% (100 -> 90) on elastic good theta = -2.0
        # Q_new = 100 * (90/100)^(-2) = 100 * (0.9)^(-2) = 123.4568 (+23.46%)
        res = opt.simulate_item_uplift(
            base_price=100.0,
            base_qty=100.0,
            new_price=90.0,
            elasticity=-2.0,
            unit_cost=50.0,
        )
        assert isinstance(res, ItemOptimizationResult)
        assert pytest.approx(res.projected_qty, 1e-2) == 123.46
        assert pytest.approx(res.demand_uplift_pct, 1e-4) == 0.234568
        # Revenue: 90 * 123.4568 = 11,111.11 (+11.11% vs 10,000)
        assert pytest.approx(res.projected_revenue, 1e-2) == 11111.11
        assert pytest.approx(res.revenue_uplift_pct, 1e-4) == 0.111111
        # Profit: (90 - 50) * 123.4568 = 4,938.27 vs 5,000 (-1.23%)
        assert pytest.approx(res.projected_profit, 1e-2) == 4938.27

    def test_simulate_item_uplift_inelastic_markup(self):
        opt = DynamicPricingOptimizer()
        # Price increased 10% (100 -> 110) on inelastic good theta = -0.5
        # Q_new = 100 * (1.1)^(-0.5) = 95.3463 (-4.65%)
        res = opt.simulate_item_uplift(
            base_price=100.0,
            base_qty=100.0,
            new_price=110.0,
            elasticity=-0.5,
            unit_cost=50.0,
        )
        assert pytest.approx(res.projected_qty, 1e-2) == 95.35
        # Revenue: 110 * 95.3463 = 10,488.09 (+4.88%)
        assert pytest.approx(res.revenue_uplift_pct, 1e-2) == 0.0488
        # Profit: (110 - 50) * 95.3463 = 5,720.78 vs 5,000 (+14.42%)
        assert pytest.approx(res.profit_uplift_pct, 1e-2) == 0.1442


class TestCatalogAndPortfolioOptimization:
    """Test dataframe catalog optimization and portfolio aggregation."""

    @pytest.fixture
    def sample_catalog(self):
        return pd.DataFrame({
            "product_id": ["P1", "P2", "P3", "P4"],
            "product_category_name": ["electronics", "electronics", "home", "luxury"],
            "unit_price": [100.0, 200.0, 50.0, 300.0],
            "qty": [50.0, 30.0, 100.0, 10.0],
            "elasticity": [-2.2, -1.8, -1.2, -0.7],
            "comp_1": [98.0, 205.0, 48.0, 310.0],
        })

    def test_optimize_catalog(self, sample_catalog):
        opt = DynamicPricingOptimizer()
        res_df = opt.optimize_catalog(sample_catalog, objective="profit")

        assert "optimal_price" in res_df.columns
        assert "projected_qty" in res_df.columns
        assert "projected_revenue" in res_df.columns
        assert "projected_profit" in res_df.columns
        assert "profit_uplift_pct" in res_df.columns
        assert len(res_df) == len(sample_catalog)

        # Prices must be strictly positive and obey bounds
        for _, row in res_df.iterrows():
            bp = row["unit_price"]
            p_opt = row["optimal_price"]
            assert bp * 0.70 <= p_opt <= bp * 1.20

    def test_simulate_portfolio_uplift(self, sample_catalog):
        opt = DynamicPricingOptimizer()
        opt_df = opt.optimize_catalog(sample_catalog, objective="profit")
        summary = opt.simulate_portfolio_uplift(opt_df)

        assert summary["total_items"] == 4
        assert summary["baseline_total_revenue"] > 0
        assert summary["projected_total_revenue"] > 0
        assert summary["projected_total_profit"] >= summary["baseline_total_profit"]

    def test_compare_pricing_scenarios(self, sample_catalog):
        opt = DynamicPricingOptimizer()
        scenarios = opt.compare_pricing_scenarios(sample_catalog)

        assert isinstance(scenarios, pd.DataFrame)
        assert len(scenarios) == 6
        assert "Status Quo (Baseline)" in scenarios["Strategy"].values
        assert "Causal Profit Maximizer" in scenarios["Strategy"].values
        assert "Revenue Uplift (%)" in scenarios.columns
        assert "Profit Uplift (%)" in scenarios.columns

    def test_format_summary_report(self, sample_catalog):
        opt = DynamicPricingOptimizer()
        opt_df = opt.optimize_catalog(sample_catalog, objective="profit")
        summary = opt.simulate_portfolio_uplift(opt_df)
        scenarios = opt.compare_pricing_scenarios(sample_catalog)

        report = opt.format_summary_report(summary, scenarios)
        assert isinstance(report, str)
        assert "ECONOCASUAL DYNAMIC PRICING OPTIMIZER" in report
        assert "STRATEGY SCENARIO COMPARISON" in report
