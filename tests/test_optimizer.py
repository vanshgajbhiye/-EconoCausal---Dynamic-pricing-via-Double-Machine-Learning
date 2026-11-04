"""Unit and integration tests for DynamicPricingOptimizer & Uplift Simulation Engine."""

import numpy as np
import pandas as pd
import pytest

from src.models.optimizer import (
    DynamicPricingOptimizer,
    ItemOptimizationResult,
    ProductConstraints,
)


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


class TestProductConstraints:
    """Test product-level business constraints and guardrail enforcement."""

    def test_global_max_discount_limit(self):
        # Base price 100, cost 40, theta = -4.0.
        # Amoroso-Robinson would recommend 40 * (-4 / -3) = 53.33 (-46.7% discount).
        # With max_discount_pct = 0.15, price floor is 100 * (1 - 0.15) = 85.0.
        opt = DynamicPricingOptimizer(max_discount_pct=0.15)
        p_opt, flag = opt.calculate_optimal_price(
            base_price=100.0,
            elasticity=-4.0,
            unit_cost=40.0,
            objective="profit",
        )
        assert pytest.approx(p_opt, 1e-4) == 85.0
        assert flag == "min_bound_clipped"

    def test_product_level_max_discount_limit(self):
        # Product constraints: max discount 10%
        opt = DynamicPricingOptimizer(min_price_change_pct=-0.30)
        constraints = ProductConstraints(max_discount_pct=0.10)
        p_opt, flag = opt.calculate_optimal_price(
            base_price=100.0,
            elasticity=-3.0,
            unit_cost=50.0,
            constraints=constraints,
        )
        # Amoroso-Robinson: 50 * (-3 / -2) = 75 (-25%), clipped to 90 (-10%)
        assert pytest.approx(p_opt, 1e-4) == 90.0
        assert flag == "min_bound_clipped"

    def test_product_level_max_markup_limit(self):
        # Inelastic good, would rise +20% (to 120), but product limit is +8% (108)
        opt = DynamicPricingOptimizer(max_price_change_pct=0.25)
        constraints = ProductConstraints(max_markup_pct=0.08)
        p_opt, flag = opt.calculate_optimal_price(
            base_price=100.0,
            elasticity=-0.7,
            constraints=constraints,
        )
        assert pytest.approx(p_opt, 1e-4) == 108.0
        assert flag == "inelastic_upper_bound"

    def test_absolute_min_and_max_price_constraints(self):
        opt = DynamicPricingOptimizer()
        constraints = ProductConstraints(min_price=88.0, max_price=112.0)
        # Elastic good clipped by absolute min_price
        p_opt_el, flag_el = opt.calculate_optimal_price(
            base_price=100.0,
            elasticity=-5.0,
            unit_cost=40.0,
            constraints=constraints,
        )
        assert pytest.approx(p_el := p_opt_el, 1e-4) == 88.0

        # Inelastic good clipped by absolute max_price
        p_opt_in, flag_in = opt.calculate_optimal_price(
            base_price=100.0,
            elasticity=-0.5,
            constraints=constraints,
        )
        assert pytest.approx(p_opt_in, 1e-4) == 112.0

    def test_map_price_enforcement(self):
        # Minimum Advertised Price ($92.00) prevents lower discounting
        opt = DynamicPricingOptimizer(min_price_change_pct=-0.30)
        constraints = ProductConstraints(map_price=92.0)
        p_opt, flag = opt.calculate_optimal_price(
            base_price=100.0,
            elasticity=-3.5,
            unit_cost=45.0,
            constraints=constraints,
        )
        assert pytest.approx(p_opt, 1e-4) == 92.0
        assert flag == "min_bound_clipped"

    def test_msrp_ceiling_enforcement(self):
        # MSRP ($106.00) prevents pricing above recommended ceiling
        opt = DynamicPricingOptimizer(max_price_change_pct=0.30)
        constraints = ProductConstraints(msrp=106.0)
        p_opt, flag = opt.calculate_optimal_price(
            base_price=100.0,
            elasticity=-0.4,
            constraints=constraints,
        )
        assert pytest.approx(p_opt, 1e-4) == 106.0

    def test_price_freeze_enforcement(self):
        opt = DynamicPricingOptimizer()
        constraints = ProductConstraints(is_price_frozen=True)
        p_opt, flag = opt.calculate_optimal_price(
            base_price=100.0,
            elasticity=-2.5,
            constraints=constraints,
        )
        assert p_opt == 100.0
        assert flag == "price_frozen"

    def test_allow_promotions_false(self):
        opt = DynamicPricingOptimizer(min_price_change_pct=-0.25)
        constraints = ProductConstraints(allow_promotions=False)
        p_opt, flag = opt.calculate_optimal_price(
            base_price=100.0,
            elasticity=-3.0,
            unit_cost=50.0,
            constraints=constraints,
        )
        # Cannot discount below base_price
        assert p_opt >= 100.0

    def test_product_specific_min_margin(self):
        # High-margin brand: cost = 80, requires min_margin = 0.20 => P >= 80 / 0.8 = 100.0
        opt = DynamicPricingOptimizer(min_margin=0.05)
        constraints = ProductConstraints(min_margin=0.20)
        p_opt, flag = opt.calculate_optimal_price(
            base_price=110.0,
            elasticity=-4.0,
            unit_cost=80.0,
            constraints=constraints,
        )
        assert p_opt >= 100.0

    def test_product_constraints_from_dict(self):
        raw_dict = {
            "min_price": 50.0,
            "max_price": 150.0,
            "discount_limit": 0.12,
            "stock": 42,
            "map": 55.0,
            "list_price": 140.0,
            "price_freeze": True,
        }
        c = ProductConstraints.from_dict(raw_dict)
        assert c.min_price == 50.0
        assert c.max_price == 150.0
        assert c.max_discount_pct == 0.12
        assert c.inventory == 42.0
        assert c.map_price == 55.0
        assert c.msrp == 140.0
        assert c.is_price_frozen is True


class TestInventoryAndCapacityConstraints:
    """Test stock limits, inventory-clearing market pricing, and portfolio budget bounds."""

    def test_item_inventory_demand_capping(self):
        opt = DynamicPricingOptimizer()
        # Price dropped 10% (100 -> 90) on elastic good theta = -2.0
        # Unconstrained demand would be 123.46. With inventory cap 110:
        res = opt.simulate_item_uplift(
            base_price=100.0,
            base_qty=100.0,
            new_price=90.0,
            elasticity=-2.0,
            unit_cost=50.0,
            inventory=110.0,
        )
        assert res.projected_qty == 110.0
        assert pytest.approx(res.stockout_units, 1e-2) == 13.46
        # Financials are capped at units physically fulfilled
        assert pytest.approx(res.projected_revenue, 1e-2) == 90.0 * 110.0
        assert pytest.approx(res.projected_profit, 1e-2) == (90.0 - 50.0) * 110.0

    def test_inventory_scarcity_market_clearing_price(self):
        opt = DynamicPricingOptimizer(max_price_change_pct=0.50)
        # Current sales = 100 units, but inventory stock is only 70 units
        # theta = -1.5, cost = 25.0 (unconstrained price would be 75.0 producing 153.96 units > 70).
        # Clearing price P_clear = 100 * (70/100)^(1 / -1.5) = 126.86
        p_opt, flag = opt.calculate_optimal_price(
            base_price=100.0,
            elasticity=-1.5,
            unit_cost=25.0,
            inventory=70.0,
            base_qty=100.0,
        )
        assert p_opt > 100.0
        assert flag == "inventory_clearing_bound"
        assert pytest.approx(p_opt, 1e-2) == 126.86

    def test_zero_inventory_condition(self):
        opt = DynamicPricingOptimizer()
        res = opt.simulate_item_uplift(
            base_price=100.0,
            base_qty=50.0,
            new_price=105.0,
            elasticity=-1.5,
            inventory=0.0,
        )
        assert res.projected_qty == 0.0
        assert res.projected_revenue == 0.0
        assert res.projected_profit == 0.0

    def test_portfolio_discount_budget_constraint(self):
        # Catalog where both items would prefer -20% discount (to $80)
        catalog = pd.DataFrame({
            "product_id": ["SKU1", "SKU2"],
            "product_category_name": ["catA", "catB"],
            "unit_price": [100.0, 100.0],
            "qty": [100.0, 100.0],
            "elasticity": [-3.0, -3.0],
            "unit_cost": [40.0, 40.0],
        })
        opt = DynamicPricingOptimizer(min_price_change_pct=-0.25)
        # Without budget constraint: total discount spend is substantial (> $2,000)
        res_unconstrained = opt.optimize_catalog(catalog)
        unconstrained_spend = float(
            ((catalog["unit_price"] - res_unconstrained["optimal_price"]) * res_unconstrained["projected_qty"]).sum()
        )
        assert unconstrained_spend > 1000.0

        # With tight promotional budget of $400:
        res_budget = opt.optimize_catalog(catalog, max_total_discount_budget=400.0)
        budget_spend = float(
            (np.maximum(0.0, catalog["unit_price"] - res_budget["optimal_price"]) * res_budget["projected_qty"]).sum()
        )
        assert budget_spend <= 401.0
        assert any("budget_constrained" in str(b) for b in res_budget["bound_hit"])

    def test_portfolio_inventory_capacity_constraint(self):
        catalog = pd.DataFrame({
            "product_id": ["SKU1", "SKU2"],
            "product_category_name": ["catA", "catB"],
            "unit_price": [100.0, 100.0],
            "qty": [50.0, 50.0],
            "elasticity": [-2.0, -2.0],
            "unit_cost": [50.0, 50.0],
        })
        opt = DynamicPricingOptimizer(min_price_change_pct=-0.20)
        # Total base volume = 100. With -20% price, volume would expand to ~156.
        # Enforce max capacity limit of 105 units:
        res = opt.optimize_catalog(catalog, max_total_inventory_capacity=105.0)
        assert res["projected_qty"].sum() <= 106.0


class TestFallbackMechanisms:
    """Test graceful fallback to safe baseline prices under invalid or unreliable predictions."""

    def test_fallback_nan_or_none_elasticity(self):
        opt = DynamicPricingOptimizer()
        p_opt, flag = opt.calculate_optimal_price(base_price=100.0, elasticity=np.nan)
        assert p_opt == 100.0
        assert flag == "fallback_nan_elasticity"

        res = opt.simulate_item_uplift(
            base_price=100.0,
            base_qty=50.0,
            new_price=p_opt,
            elasticity=np.nan,
            bound_hit=flag,
        )
        assert res.is_fallback is True
        assert res.optimal_price == 100.0
        assert res.price_change_pct == 0.0
        assert res.demand_uplift_pct == 0.0
        assert res.profit_uplift_pct == 0.0
        assert res.projected_profit == res.baseline_profit

    def test_fallback_infinite_elasticity(self):
        opt = DynamicPricingOptimizer()
        p_pos, flag_pos = opt.calculate_optimal_price(base_price=100.0, elasticity=float("inf"))
        assert p_pos == 100.0
        assert flag_pos == "fallback_nan_elasticity"

        p_neg, flag_neg = opt.calculate_optimal_price(base_price=100.0, elasticity=float("-inf"))
        assert p_neg == 100.0
        assert flag_neg == "fallback_nan_elasticity"

    def test_fallback_positive_elasticity(self):
        # Positive elasticity implies upward-sloping demand curve (violating Law of Demand)
        opt = DynamicPricingOptimizer()
        p_opt, flag = opt.calculate_optimal_price(base_price=100.0, elasticity=1.5)
        assert p_opt == 100.0
        assert flag == "fallback_positive_elasticity"

    def test_fallback_zero_elasticity(self):
        opt = DynamicPricingOptimizer()
        p_opt, flag = opt.calculate_optimal_price(base_price=100.0, elasticity=0.0)
        assert p_opt == 100.0
        assert flag == "fallback_zero_elasticity"

    def test_fallback_unreliable_extreme_elasticity(self):
        opt = DynamicPricingOptimizer(min_reliable_elasticity=-10.0, max_reliable_elasticity=-0.05)
        # Extreme negative elasticity (overfitted or explosive)
        p_opt1, flag1 = opt.calculate_optimal_price(base_price=100.0, elasticity=-15.0)
        assert p_opt1 == 100.0
        assert flag1 == "fallback_unreliable_elasticity"

        # Near-zero elasticity below noise threshold
        p_opt2, flag2 = opt.calculate_optimal_price(base_price=100.0, elasticity=-0.02)
        assert p_opt2 == 100.0
        assert flag2 == "fallback_unreliable_elasticity"

    def test_fallback_insignificant_pvalue(self):
        opt = DynamicPricingOptimizer(alpha_significance=0.05)
        # Elasticity estimate is -2.0, but p-value is 0.25 (> 0.05)
        p_opt, flag = opt.calculate_optimal_price(
            base_price=100.0,
            elasticity=-2.0,
            p_value=0.25,
        )
        assert p_opt == 100.0
        assert flag == "fallback_insignificant_pvalue"

    def test_fallback_high_standard_error(self):
        opt = DynamicPricingOptimizer()
        # Estimate = -1.2, but standard error = 2.0 (t-stat = 0.6 < 1.0)
        p_opt, flag = opt.calculate_optimal_price(
            base_price=100.0,
            elasticity=-1.2,
            elasticity_se=2.0,
        )
        assert p_opt == 100.0
        assert flag == "fallback_high_standard_error"

    def test_fallback_confidence_interval_spanning_zero(self):
        opt = DynamicPricingOptimizer()
        # CI [-2.5, +0.5] spans zero, so effect direction is ambiguous
        p_opt, flag = opt.calculate_optimal_price(
            base_price=100.0,
            elasticity=-1.5,
            ci_lower=-2.5,
            ci_upper=0.5,
        )
        assert p_opt == 100.0
        assert flag == "fallback_ci_spans_zero"

    def test_catalog_mixed_reliability(self):
        mixed_df = pd.DataFrame({
            "product_id": ["P1", "P2", "P3", "P4"],
            "product_category_name": ["cat1", "cat1", "cat2", "cat2"],
            "unit_price": [100.0, 100.0, 100.0, 100.0],
            "qty": [50.0, 50.0, 50.0, 50.0],
            "elasticity": [-2.0, np.nan, 0.5, -20.0],
        })
        opt = DynamicPricingOptimizer()
        res = opt.optimize_catalog(mixed_df)

        assert not res.at[0, "is_fallback"]
        assert res.at[0, "optimal_price"] != 100.0

        assert bool(res.at[1, "is_fallback"]) is True
        assert res.at[1, "optimal_price"] == 100.0

        assert bool(res.at[2, "is_fallback"]) is True
        assert res.at[2, "optimal_price"] == 100.0

        assert bool(res.at[3, "is_fallback"]) is True
        assert res.at[3, "optimal_price"] == 100.0


class TestBoundaryConditionsAndEdgeCases:
    """Test microeconomic boundary conditions, margin floors, and numerical stability."""

    def test_underwater_item_restores_margin(self):
        # Selling price is 50, but cost is 65 (underwater margin).
        # Margin floor (min_margin = 0.05) must push price to >= 65 / 0.95 = 68.42.
        opt = DynamicPricingOptimizer(min_margin=0.05)
        p_opt, flag = opt.calculate_optimal_price(
            base_price=50.0,
            elasticity=-2.0,
            unit_cost=65.0,
        )
        assert p_opt >= 68.42

    def test_zero_baseline_demand_safeguard(self):
        opt = DynamicPricingOptimizer()
        res = opt.simulate_item_uplift(
            base_price=100.0,
            base_qty=0.0,
            new_price=110.0,
            elasticity=-2.0,
            unit_cost=50.0,
        )
        assert res.projected_qty == 0.0
        assert res.demand_uplift_pct == 0.0
        assert res.revenue_uplift_pct == 0.0
        assert res.profit_uplift_pct == 0.0

    def test_extreme_price_ratio_numerical_stability(self):
        opt = DynamicPricingOptimizer()
        res = opt.simulate_item_uplift(
            base_price=1.0,
            base_qty=100.0,
            new_price=1000.0,
            elasticity=-5.0,
            unit_cost=0.5,
        )
        assert not np.isnan(res.projected_qty)
        assert not np.isinf(res.projected_qty)
        assert res.projected_qty >= 0.0

    def test_optimize_single_item_convenience(self):
        opt = DynamicPricingOptimizer()
        res = opt.optimize_single_item(
            base_price=100.0,
            elasticity=-2.0,
            base_qty=50.0,
            unit_cost=60.0,
        )
        assert isinstance(res, ItemOptimizationResult)
        assert pytest.approx(res.optimal_price, 1e-4) == 120.0
        assert res.projected_profit > res.baseline_profit

    def test_catalog_category_and_product_constraints(self):
        catalog = pd.DataFrame({
            "product_id": ["P1", "P2"],
            "product_category_name": ["electronics", "luxury"],
            "unit_price": [100.0, 200.0],
            "qty": [50.0, 20.0],
            "elasticity": [-2.0, -1.5],
        })
        opt = DynamicPricingOptimizer()
        cat_constraints = {
            "luxury": ProductConstraints(max_discount_pct=0.05),
        }
        prod_constraints = {
            "P1": ProductConstraints(is_price_frozen=True),
        }
        res = opt.optimize_catalog(
            catalog,
            product_constraints=prod_constraints,
            category_constraints=cat_constraints,
        )
        # P1 is price frozen
        assert res.at[0, "optimal_price"] == 100.0
        assert res.at[0, "bound_hit"] == "price_frozen"
        # P2 in luxury cannot discount more than 5%
        assert res.at[1, "optimal_price"] >= 200.0 * 0.95
