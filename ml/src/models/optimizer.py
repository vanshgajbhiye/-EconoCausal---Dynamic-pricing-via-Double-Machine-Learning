"""Dynamic Pricing Optimizer & Counterfactual Uplift Simulation Engine.

Translates causal price elasticity estimates (ATE or CATE) into profit-maximizing
and revenue-maximizing dynamic price recommendations under realistic retail constraints
(constant-elasticity demand curve, cost structures, guardrails, and competitor anchors).

Formulation:
    Constant Elasticity Demand:
        Q(P) = Q_0 * (P / P_0)^theta,   theta < 0

    Revenue:
        R(P) = P * Q(P) = Q_0 * P_0^(-theta) * P^(1 + theta)

    Profit (with unit cost c):
        Pi(P) = (P - c) * Q(P) = (P - c) * Q_0 * (P / P_0)^theta

    Analytical First-Order Condition for Profit Maximization (Amoroso-Robinson / Lerner Index):
        dPi / dP = 0  =>  P* = c * [theta / (theta + 1)]  for theta < -1
        For theta >= -1 (inelastic), profit increases monotonically with price,
        so P* is bounded by the allowable upper price guardrail.
"""

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple, Union
import numpy as np
import pandas as pd


@dataclass
class ItemOptimizationResult:
    """Dataclass holding pricing recommendation and simulated uplift for a single item."""

    baseline_price: float
    optimal_price: float
    price_change_pct: float
    elasticity: float
    unit_cost: float
    baseline_qty: float
    projected_qty: float
    demand_uplift_pct: float
    baseline_revenue: float
    projected_revenue: float
    revenue_uplift_pct: float
    baseline_profit: float
    projected_profit: float
    profit_uplift_pct: float
    baseline_margin: float
    projected_margin: float
    bound_hit: str
    objective: str


class DynamicPricingOptimizer:
    """Dynamic pricing optimizer and counterfactual uplift simulation engine.

    Parameters
    ----------
    default_margin : float, default=0.30
        Default gross margin (P - c) / P assumed if unit cost is not explicitly provided.
        Defaults to 30%.
    min_price_change_pct : float, default=-0.30
        Maximum allowed price decrease / discount relative to baseline (e.g. -0.30 = -30%).
    max_price_change_pct : float, default=0.20
        Maximum allowed price increase / markup relative to baseline (e.g. +0.20 = +20%).
    min_margin : float, default=0.05
        Hard lower floor on profit margin to prevent loss-leader pricing below cost.
    charm_pricing : bool, default=False
        Whether to round recommended prices to charm pricing endings (.99 or .95).
    charm_ending : float, default=0.99
        Cents ending for charm pricing (e.g. 0.99).
    random_state : int, default=42
        Seed for reproducibility in scenario variations.
    """

    def __init__(
        self,
        default_margin: float = 0.30,
        min_price_change_pct: float = -0.30,
        max_price_change_pct: float = 0.20,
        min_margin: float = 0.05,
        charm_pricing: bool = False,
        charm_ending: float = 0.99,
        random_state: int = 42,
    ):
        if min_price_change_pct >= max_price_change_pct:
            raise ValueError(
                f"min_price_change_pct ({min_price_change_pct}) must be less than "
                f"max_price_change_pct ({max_price_change_pct})."
            )
        if not (0.0 < default_margin < 1.0):
            raise ValueError(f"default_margin must be in (0, 1), got {default_margin}")
        if not (0.0 <= min_margin < 1.0):
            raise ValueError(f"min_margin must be in [0, 1), got {min_margin}")

        self.default_margin = default_margin
        self.min_price_change_pct = min_price_change_pct
        self.max_price_change_pct = max_price_change_pct
        self.min_margin = min_margin
        self.charm_pricing = charm_pricing
        self.charm_ending = charm_ending
        self.random_state = random_state

    def _apply_charm_pricing(self, price: float, lower_bound: float, upper_bound: float) -> float:
        """Round price to the nearest psychological ending (e.g. .99) within bounds."""
        base_int = int(np.floor(price))
        candidate = base_int + self.charm_ending

        # If candidate exceeds upper bound or price was closer to previous dollar
        if candidate > upper_bound and (base_int - 1 + self.charm_ending) >= lower_bound:
            candidate = base_int - 1 + self.charm_ending
        elif candidate < lower_bound and (base_int + 1 + self.charm_ending) <= upper_bound:
            candidate = base_int + 1 + self.charm_ending

        # Clip strictly to permissible interval
        return float(np.clip(candidate, lower_bound, upper_bound))

    def calculate_optimal_price(
        self,
        base_price: float,
        elasticity: float,
        unit_cost: Optional[float] = None,
        objective: str = "profit",
        min_price: Optional[float] = None,
        max_price: Optional[float] = None,
        comp_price: Optional[float] = None,
        comp_guardrail_pct: Optional[float] = None,
    ) -> Tuple[float, str]:
        """Compute the unconstrained or constrained optimal price for an item.

        Parameters
        ----------
        base_price : float
            Current baseline selling price of the item (P_0 > 0).
        elasticity : float
            Estimated causal price elasticity of demand theta (typically negative).
        unit_cost : Optional[float]
            Unit marginal cost c. If None, derived as base_price * (1 - default_margin).
        objective : str, default="profit"
            Optimization target: "profit" or "revenue".
        min_price : Optional[float]
            Explicit absolute minimum price allowed.
        max_price : Optional[float]
            Explicit absolute maximum price allowed.
        comp_price : Optional[float]
            Competitor benchmark price.
        comp_guardrail_pct : Optional[float]
            Maximum percentage tolerance around competitor price (e.g. 0.15 = +/-15%).

        Returns
        -------
        Tuple[float, str]
            (optimal_price, bound_hit_description)
        """
        if base_price <= 0:
            raise ValueError(f"base_price must be positive, got {base_price}")

        # Derive cost if not provided
        if unit_cost is None or np.isnan(unit_cost) or unit_cost <= 0:
            cost = base_price * (1.0 - self.default_margin)
        else:
            cost = float(unit_cost)

        # Baseline percentage bounds
        p_lower = base_price * (1.0 + self.min_price_change_pct)
        p_upper = base_price * (1.0 + self.max_price_change_pct)

        # Cost-plus minimum margin floor: P >= cost / (1 - min_margin)
        cost_floor = cost / (1.0 - self.min_margin)
        p_lower = max(p_lower, cost_floor)

        # Explicit absolute bounds
        if min_price is not None and not np.isnan(min_price):
            p_lower = max(p_lower, float(min_price))
        if max_price is not None and not np.isnan(max_price):
            p_upper = min(p_upper, float(max_price))

        # Competitor guardrail clipping
        if comp_price is not None and not np.isnan(comp_price) and comp_price > 0:
            tol = comp_guardrail_pct if comp_guardrail_pct is not None else 0.25
            comp_lower = comp_price * (1.0 - tol)
            comp_upper = comp_price * (1.0 + tol)
            # Strictly protect margin: competitor lower bound cannot breach cost floor
            comp_lower = max(comp_lower, cost_floor)
            if comp_upper >= comp_lower:
                p_lower = max(p_lower, comp_lower)
                p_upper = min(p_upper, comp_upper)

        # Reconcile if lower bound exceeds upper bound
        if p_lower > p_upper:
            p_upper = p_lower

        bound_hit = "unconstrained_optimal"

        # Theoretical optimal calculation
        if objective.lower() == "profit":
            # Amoroso-Robinson condition: P* = c * (theta / (theta + 1))
            if elasticity < -1.0:
                raw_optimal = cost * (elasticity / (elasticity + 1.0))
            else:
                # Inelastic demand (theta >= -1.0): margin increases faster than volume drops
                raw_optimal = p_upper
                bound_hit = "inelastic_upper_bound"
        elif objective.lower() == "revenue":
            # Revenue = P * Q = P^(1 + theta)
            if elasticity < -1.0:
                # Elastic: lowering price increases revenue -> push to lowest bound
                raw_optimal = p_lower
                bound_hit = "elastic_revenue_lower_bound"
            elif elasticity > -1.0:
                # Inelastic: raising price increases revenue -> push to highest bound
                raw_optimal = p_upper
                bound_hit = "inelastic_revenue_upper_bound"
            else:
                # Unit elastic (theta == -1.0): revenue is invariant to price
                raw_optimal = base_price
                bound_hit = "unit_elastic_neutral"
        else:
            raise ValueError(f"Unknown objective '{objective}'. Choose 'profit' or 'revenue'.")

        # Bound clipping
        if raw_optimal <= p_lower:
            final_price = p_lower
            if bound_hit == "unconstrained_optimal":
                bound_hit = "min_bound_clipped"
        elif raw_optimal >= p_upper:
            final_price = p_upper
            if bound_hit == "unconstrained_optimal":
                bound_hit = "max_bound_clipped"
        else:
            final_price = raw_optimal

        # Charm pricing
        if self.charm_pricing:
            final_price = self._apply_charm_pricing(final_price, p_lower, p_upper)

        return float(final_price), bound_hit

    def simulate_item_uplift(
        self,
        base_price: float,
        base_qty: float,
        new_price: float,
        elasticity: float,
        unit_cost: Optional[float] = None,
        objective: str = "profit",
        bound_hit: str = "custom",
    ) -> ItemOptimizationResult:
        """Simulate counterfactual quantity, revenue, and profit uplifts under new price.

        Parameters
        ----------
        base_price : float
            Historical baseline unit price.
        base_qty : float
            Historical baseline demand volume.
        new_price : float
            Proposed new price.
        elasticity : float
            Causal price elasticity theta.
        unit_cost : Optional[float]
            Unit marginal cost.
        objective : str, default="profit"
            Optimization target description.
        bound_hit : str, default="custom"
            Flag describing which boundary or rule determined the price.

        Returns
        -------
        ItemOptimizationResult
            Full counterfactual impact metrics.
        """
        if base_price <= 0 or base_qty < 0 or new_price <= 0:
            raise ValueError("base_price, base_qty, and new_price must be positive.")

        cost = (
            float(unit_cost)
            if (unit_cost is not None and not np.isnan(unit_cost) and unit_cost > 0)
            else base_price * (1.0 - self.default_margin)
        )

        # Constant elasticity demand projection: Q_new = Q_0 * (P_new / P_0)^theta
        price_ratio = new_price / base_price
        # Safeguard numerical exponentiation
        safe_ratio = max(price_ratio, 1e-4)
        projected_qty = base_qty * (safe_ratio**elasticity)

        # Financial metrics
        base_rev = base_price * base_qty
        proj_rev = new_price * projected_qty

        base_prof = (base_price - cost) * base_qty
        proj_prof = (new_price - cost) * projected_qty

        price_chg_pct = (new_price - base_price) / base_price
        demand_uplift_pct = (projected_qty - base_qty) / base_qty if base_qty > 0 else 0.0
        rev_uplift_pct = (proj_rev - base_rev) / base_rev if base_rev > 0 else 0.0
        prof_uplift_pct = (proj_prof - base_prof) / base_prof if base_prof != 0 else 0.0

        base_margin = (base_price - cost) / base_price
        proj_margin = (new_price - cost) / new_price

        return ItemOptimizationResult(
            baseline_price=base_price,
            optimal_price=new_price,
            price_change_pct=price_chg_pct,
            elasticity=elasticity,
            unit_cost=cost,
            baseline_qty=base_qty,
            projected_qty=projected_qty,
            demand_uplift_pct=demand_uplift_pct,
            baseline_revenue=base_rev,
            projected_revenue=proj_rev,
            revenue_uplift_pct=rev_uplift_pct,
            baseline_profit=base_prof,
            projected_profit=proj_prof,
            profit_uplift_pct=prof_uplift_pct,
            baseline_margin=base_margin,
            projected_margin=proj_margin,
            bound_hit=bound_hit,
            objective=objective,
        )

    def optimize_catalog(
        self,
        df: pd.DataFrame,
        price_col: str = "unit_price",
        qty_col: str = "qty",
        elasticity_col: str = "elasticity",
        cost_col: Optional[str] = None,
        comp_price_col: Optional[str] = None,
        objective: str = "profit",
    ) -> pd.DataFrame:
        """Apply dynamic pricing optimization across an entire product catalog or transaction dataset.

        Parameters
        ----------
        df : pd.DataFrame
            Catalog or transaction dataset.
        price_col : str, default="unit_price"
            Column name for baseline price.
        qty_col : str, default="qty"
            Column name for baseline demand volume.
        elasticity_col : str, default="elasticity"
            Column name for item or category elasticity theta.
        cost_col : Optional[str]
            Column name for unit cost, if present.
        comp_price_col : Optional[str]
            Column name for competitor benchmark price.
        objective : str, default="profit"
            Optimization target: "profit" or "revenue".

        Returns
        -------
        pd.DataFrame
            Copy of input DataFrame augmented with optimal prices and uplift projections.
        """
        required = [price_col, qty_col, elasticity_col]
        for col in required:
            if col not in df.columns:
                raise ValueError(f"Required column '{col}' not found in dataframe.")

        out = df.copy()

        opt_prices = []
        bounds = []
        proj_qtys = []
        proj_revs = []
        proj_profs = []
        price_chg_pcts = []
        demand_uplifts = []
        rev_uplifts = []
        prof_uplifts = []
        base_profs = []
        base_revs = []
        unit_costs = []

        for _, row in out.iterrows():
            bp = float(row[price_col])
            b_qty = float(row[qty_col])
            theta = float(row[elasticity_col])
            c = float(row[cost_col]) if (cost_col and cost_col in row and not pd.isna(row[cost_col])) else None
            comp = (
                float(row[comp_price_col])
                if (comp_price_col and comp_price_col in row and not pd.isna(row[comp_price_col]))
                else None
            )

            p_opt, b_flag = self.calculate_optimal_price(
                base_price=bp,
                elasticity=theta,
                unit_cost=c,
                objective=objective,
                comp_price=comp,
            )

            res = self.simulate_item_uplift(
                base_price=bp,
                base_qty=b_qty,
                new_price=p_opt,
                elasticity=theta,
                unit_cost=c,
                objective=objective,
                bound_hit=b_flag,
            )

            opt_prices.append(res.optimal_price)
            bounds.append(res.bound_hit)
            proj_qtys.append(res.projected_qty)
            proj_revs.append(res.projected_revenue)
            proj_profs.append(res.projected_profit)
            price_chg_pcts.append(res.price_change_pct)
            demand_uplifts.append(res.demand_uplift_pct)
            rev_uplifts.append(res.revenue_uplift_pct)
            prof_uplifts.append(res.profit_uplift_pct)
            base_profs.append(res.baseline_profit)
            base_revs.append(res.baseline_revenue)
            unit_costs.append(res.unit_cost)

        out["optimal_price"] = opt_prices
        out["bound_hit"] = bounds
        out["unit_cost"] = unit_costs
        out["baseline_revenue"] = base_revs
        out["projected_revenue"] = proj_revs
        out["baseline_profit"] = base_profs
        out["projected_profit"] = proj_profs
        out["projected_qty"] = proj_qtys
        out["price_change_pct"] = price_chg_pcts
        out["demand_uplift_pct"] = demand_uplifts
        out["revenue_uplift_pct"] = rev_uplifts
        out["profit_uplift_pct"] = prof_uplifts

        return out

    def simulate_portfolio_uplift(
        self,
        catalog_df: pd.DataFrame,
        base_price_col: str = "unit_price",
        base_qty_col: str = "qty",
        new_price_col: str = "optimal_price",
        elasticity_col: str = "elasticity",
        cost_col: Optional[str] = "unit_cost",
    ) -> Dict[str, Any]:
        """Aggregate counterfactual metrics across the entire portfolio/catalog.

        Parameters
        ----------
        catalog_df : pd.DataFrame
            Catalog with baseline and proposed prices.
        base_price_col : str, default="unit_price"
        base_qty_col : str, default="qty"
        new_price_col : str, default="optimal_price"
        elasticity_col : str, default="elasticity"
        cost_col : Optional[str], default="unit_cost"

        Returns
        -------
        Dict[str, Any]
            Portfolio summary dictionary with aggregate financial impacts.
        """
        total_base_rev = 0.0
        total_proj_rev = 0.0
        total_base_prof = 0.0
        total_proj_prof = 0.0
        total_base_qty = 0.0
        total_proj_qty = 0.0

        for _, row in catalog_df.iterrows():
            bp = float(row[base_price_col])
            b_qty = float(row[base_qty_col])
            np_price = float(row[new_price_col])
            theta = float(row[elasticity_col])
            cost = (
                float(row[cost_col])
                if (cost_col and cost_col in row and not pd.isna(row[cost_col]))
                else bp * (1.0 - self.default_margin)
            )

            res = self.simulate_item_uplift(
                base_price=bp,
                base_qty=b_qty,
                new_price=np_price,
                elasticity=theta,
                unit_cost=cost,
            )

            total_base_rev += res.baseline_revenue
            total_proj_rev += res.projected_revenue
            total_base_prof += res.baseline_profit
            total_proj_prof += res.projected_profit
            total_base_qty += res.baseline_qty
            total_proj_qty += res.projected_qty

        net_rev_uplift = (total_proj_rev - total_base_rev) / total_base_rev if total_base_rev > 0 else 0.0
        net_prof_uplift = (total_proj_prof - total_base_prof) / total_base_prof if total_base_prof > 0 else 0.0
        net_qty_uplift = (total_proj_qty - total_base_qty) / total_base_qty if total_base_qty > 0 else 0.0

        base_margin = total_base_prof / total_base_rev if total_base_rev > 0 else 0.0
        proj_margin = total_proj_prof / total_proj_rev if total_proj_rev > 0 else 0.0

        return {
            "total_items": len(catalog_df),
            "baseline_total_revenue": total_base_rev,
            "projected_total_revenue": total_proj_rev,
            "net_revenue_uplift_pct": net_rev_uplift,
            "net_revenue_gain": total_proj_rev - total_base_rev,
            "baseline_total_profit": total_base_prof,
            "projected_total_profit": total_proj_prof,
            "net_profit_uplift_pct": net_prof_uplift,
            "net_profit_gain": total_proj_prof - total_base_prof,
            "baseline_total_units": total_base_qty,
            "projected_total_units": total_proj_qty,
            "net_units_uplift_pct": net_qty_uplift,
            "baseline_margin": base_margin,
            "projected_margin": proj_margin,
        }

    def compare_pricing_scenarios(
        self,
        df: pd.DataFrame,
        price_col: str = "unit_price",
        qty_col: str = "qty",
        elasticity_col: str = "elasticity",
        cost_col: Optional[str] = None,
        comp_col: Optional[str] = "comp_1",
    ) -> pd.DataFrame:
        """Compare multiple retail pricing strategies against baseline.

        Strategies compared:
        1. Status Quo (Baseline)
        2. Flat 10% Discount
        3. Flat 10% Price Increase
        4. Match Key Competitor (comp_1)
        5. Causal Revenue Maximizer
        6. Causal Profit Maximizer

        Parameters
        ----------
        df : pd.DataFrame
            Catalog dataframe with prices, volumes, and elasticities.
        price_col : str, default="unit_price"
        qty_col : str, default="qty"
        elasticity_col : str, default="elasticity"
        cost_col : Optional[str]
        comp_col : Optional[str], default="comp_1"

        Returns
        -------
        pd.DataFrame
            Comparison summary table across scenarios.
        """
        temp = df.copy()

        # Generate scenario prices
        # 1. Status Quo
        temp["price_status_quo"] = temp[price_col]

        # 2. Flat 10% Discount
        temp["price_discount_10"] = temp[price_col] * 0.90

        # 3. Flat 10% Markup
        temp["price_markup_10"] = temp[price_col] * 1.10

        # 4. Match Key Competitor (fall back to baseline if missing; protected by cost floor)
        cost_floors = temp[price_col] * (1.0 - self.default_margin) / (1.0 - self.min_margin)
        if comp_col and comp_col in temp.columns:
            raw_comp = temp[comp_col].fillna(temp[price_col])
            temp["price_match_comp"] = np.maximum(raw_comp, cost_floors)
        else:
            temp["price_match_comp"] = temp[price_col]

        # 5. Causal Revenue Maximizer
        opt_rev = self.optimize_catalog(
            temp,
            price_col=price_col,
            qty_col=qty_col,
            elasticity_col=elasticity_col,
            cost_col=cost_col,
            objective="revenue",
        )
        temp["price_causal_rev"] = opt_rev["optimal_price"]

        # 6. Causal Profit Maximizer
        opt_prof = self.optimize_catalog(
            temp,
            price_col=price_col,
            qty_col=qty_col,
            elasticity_col=elasticity_col,
            cost_col=cost_col,
            objective="profit",
        )
        temp["price_causal_prof"] = opt_prof["optimal_price"]

        scenarios = [
            ("Status Quo (Baseline)", "price_status_quo"),
            ("Flat 10% Discount", "price_discount_10"),
            ("Flat 10% Markup", "price_markup_10"),
            ("Competitor Benchmark", "price_match_comp"),
            ("Causal Revenue Maximizer", "price_causal_rev"),
            ("Causal Profit Maximizer", "price_causal_prof"),
        ]

        records = []
        for name, p_col in scenarios:
            port = self.simulate_portfolio_uplift(
                temp,
                base_price_col=price_col,
                base_qty_col=qty_col,
                new_price_col=p_col,
                elasticity_col=elasticity_col,
                cost_col=cost_col,
            )
            records.append({
                "Strategy": name,
                "Total Revenue ($)": round(port["projected_total_revenue"], 2),
                "Revenue Uplift (%)": round(port["net_revenue_uplift_pct"] * 100, 2),
                "Total Profit ($)": round(port["projected_total_profit"], 2),
                "Profit Uplift (%)": round(port["net_profit_uplift_pct"] * 100, 2),
                "Total Units Sold": round(port["projected_total_units"], 1),
                "Units Uplift (%)": round(port["net_units_uplift_pct"] * 100, 2),
                "Average Margin (%)": round(port["projected_margin"] * 100, 2),
            })

        return pd.DataFrame(records)

    def format_summary_report(
        self,
        portfolio_summary: Dict[str, Any],
        scenario_df: Optional[pd.DataFrame] = None,
    ) -> str:
        """Format dynamic pricing optimization findings into an executive text report."""
        lines = [
            "=" * 78,
            "       ECONOCASUAL DYNAMIC PRICING OPTIMIZER & UPLIFT SIMULATION",
            "=" * 78,
            f"Catalog Sample Size:                    {portfolio_summary['total_items']} items",
            "-" * 78,
            "FINANCIAL PERFORMANCE (Baseline vs Causal Profit Maximizer):",
            f"  - Baseline Total Revenue:             ${portfolio_summary['baseline_total_revenue']:,.2f}",
            f"  - Projected Total Revenue:            ${portfolio_summary['projected_total_revenue']:,.2f}",
            f"  - Net Revenue Uplift:                 {portfolio_summary['net_revenue_uplift_pct'] * 100:+.2f}% (${portfolio_summary['net_revenue_gain']:+,.2f})",
            "",
            f"  - Baseline Total Gross Profit:        ${portfolio_summary['baseline_total_profit']:,.2f}",
            f"  - Projected Total Gross Profit:       ${portfolio_summary['projected_total_profit']:,.2f}",
            f"  - Net Profit Uplift:                  {portfolio_summary['net_profit_uplift_pct'] * 100:+.2f}% (${portfolio_summary['net_profit_gain']:+,.2f})",
            "",
            f"  - Baseline Unit Demand:               {portfolio_summary['baseline_total_units']:,.1f} units",
            f"  - Projected Unit Demand:              {portfolio_summary['projected_total_units']:,.1f} units",
            f"  - Net Demand Uplift:                  {portfolio_summary['net_units_uplift_pct'] * 100:+.2f}%",
            "",
            f"  - Baseline Portfolio Margin:          {portfolio_summary['baseline_margin'] * 100:.2f}%",
            f"  - Projected Portfolio Margin:         {portfolio_summary['projected_margin'] * 100:.2f}%",
        ]

        if scenario_df is not None:
            lines.extend([
                "-" * 78,
                "STRATEGY SCENARIO COMPARISON:",
                scenario_df.to_string(index=False),
            ])

        lines.append("=" * 78)
        return "\n".join(lines)
