"""Dynamic Pricing Optimizer & Counterfactual Uplift Simulation Engine.

Translates causal price elasticity estimates (ATE or CATE) into profit-maximizing
and revenue-maximizing dynamic price recommendations under realistic retail constraints
(constant-elasticity demand curve, cost structures, guardrails, competitor anchors,
discount limits, inventory limits, and promotional budgets).

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

    Inventory Scarcity / Market-Clearing Condition:
        Q(P_clear) = I  =>  P_clear = P_0 * (I / Q_0)^(1 / theta)
"""

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple, Union
import numpy as np
import pandas as pd


@dataclass
class ProductConstraints:
    """Dataclass holding product-level business rules, guardrails, and operational constraints.

    Attributes
    ----------
    min_price : Optional[float]
        Hard lower floor on absolute selling price ($).
    max_price : Optional[float]
        Hard upper ceiling on absolute selling price ($).
    max_discount_pct : Optional[float]
        Maximum allowed price decrease / discount relative to baseline (e.g. 0.15 = 15% max discount).
    max_markup_pct : Optional[float]
        Maximum allowed price increase / markup relative to baseline (e.g. 0.25 = 25% max markup).
    min_price_change_pct : Optional[float]
        Signed price change lower bound (e.g. -0.20 = max 20% discount).
    max_price_change_pct : Optional[float]
        Signed price change upper bound (e.g. +0.20 = max 20% markup).
    min_margin : Optional[float]
        Product-specific minimum profit margin floor (e.g. 0.15 = 15%).
    map_price : Optional[float]
        Minimum Advertised Price floor; price cannot fall below this anchor.
    msrp : Optional[float]
        Manufacturer's Suggested Retail Price ceiling; price cannot exceed this anchor.
    is_price_frozen : bool
        If True, lock product at baseline price (e.g. promotional freeze, contract fixed price).
    allow_promotions : bool
        If False, discounts below baseline price are strictly forbidden.
    inventory : Optional[float]
        Current available inventory / stock on hand for this SKU.
    target_profit : Optional[float]
        Minimum required dollar gross profit for the item.
    """

    min_price: Optional[float] = None
    max_price: Optional[float] = None
    max_discount_pct: Optional[float] = None
    max_markup_pct: Optional[float] = None
    min_price_change_pct: Optional[float] = None
    max_price_change_pct: Optional[float] = None
    min_margin: Optional[float] = None
    map_price: Optional[float] = None
    msrp: Optional[float] = None
    is_price_frozen: bool = False
    allow_promotions: bool = True
    inventory: Optional[float] = None
    target_profit: Optional[float] = None

    @classmethod
    def from_dict(cls, data: Optional[Dict[str, Any]]) -> "ProductConstraints":
        """Construct ProductConstraints safely from a dictionary or row mapping."""
        if not data:
            return cls()
        valid_keys = {
            "min_price",
            "max_price",
            "max_discount_pct",
            "max_markup_pct",
            "min_price_change_pct",
            "max_price_change_pct",
            "min_margin",
            "map_price",
            "msrp",
            "is_price_frozen",
            "allow_promotions",
            "inventory",
            "target_profit",
        }
        filtered: Dict[str, Any] = {}
        for k, v in data.items():
            if pd.isna(v) if not isinstance(v, (list, dict, tuple)) else False:
                continue
            if k in valid_keys:
                filtered[k] = v
            elif k in ("stock", "stock_qty", "inventory_level"):
                filtered["inventory"] = float(v)
            elif k in ("price_freeze", "frozen"):
                filtered["is_price_frozen"] = bool(v)
            elif k in ("map", "min_advertised_price"):
                filtered["map_price"] = float(v)
            elif k in ("list_price", "suggested_retail_price"):
                filtered["msrp"] = float(v)
            elif k in ("discount_limit", "max_discount"):
                filtered["max_discount_pct"] = float(v)
        return cls(**filtered)


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
    inventory: Optional[float] = None
    stockout_units: float = 0.0
    is_fallback: bool = False
    fallback_reason: Optional[str] = None


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
    max_discount_pct : Optional[float], default=None
        Global cap on maximum allowed discount percentage (e.g. 0.20 = max 20% discount).
        If provided, enforces price >= base_price * (1 - max_discount_pct).
    min_reliable_elasticity : float, default=-10.0
        Minimum economically plausible elasticity threshold. Elasticities below this
        (e.g. -20) trigger safe baseline fallback.
    max_reliable_elasticity : float, default=-0.05
        Maximum economically plausible elasticity threshold. Elasticities between this
        and 0 (e.g. -0.01) trigger safe baseline fallback.
    alpha_significance : float, default=0.10
        Significance level alpha for p-values. If p-value > alpha, elasticity is considered
        unreliable and triggers safe baseline fallback.
    enforce_profit_guardrail : bool, default=False
        Whether to strictly prevent recommendations that reduce gross profit below baseline.
    global_inventory_limit : Optional[float], default=None
        Global inventory/capacity ceiling across portfolio.
    global_promotional_budget : Optional[float], default=None
        Global discount dollar budget ceiling across portfolio.
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
        max_discount_pct: Optional[float] = None,
        min_reliable_elasticity: float = -10.0,
        max_reliable_elasticity: float = -0.05,
        alpha_significance: float = 0.10,
        enforce_profit_guardrail: bool = False,
        global_inventory_limit: Optional[float] = None,
        global_promotional_budget: Optional[float] = None,
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
        if max_discount_pct is not None and not (0.0 <= max_discount_pct <= 1.0):
            raise ValueError(f"max_discount_pct must be in [0, 1], got {max_discount_pct}")
        if min_reliable_elasticity >= max_reliable_elasticity:
            raise ValueError(
                f"min_reliable_elasticity ({min_reliable_elasticity}) must be less than "
                f"max_reliable_elasticity ({max_reliable_elasticity})."
            )

        self.default_margin = default_margin
        self.min_price_change_pct = min_price_change_pct
        self.max_price_change_pct = max_price_change_pct
        self.min_margin = min_margin
        self.charm_pricing = charm_pricing
        self.charm_ending = charm_ending
        self.random_state = random_state
        self.max_discount_pct = max_discount_pct
        self.min_reliable_elasticity = min_reliable_elasticity
        self.max_reliable_elasticity = max_reliable_elasticity
        self.alpha_significance = alpha_significance
        self.enforce_profit_guardrail = enforce_profit_guardrail
        self.global_inventory_limit = global_inventory_limit
        self.global_promotional_budget = global_promotional_budget

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

    def validate_elasticity_reliability(
        self,
        elasticity: Any,
        elasticity_se: Optional[float] = None,
        p_value: Optional[float] = None,
        ci_lower: Optional[float] = None,
        ci_upper: Optional[float] = None,
    ) -> Tuple[bool, str]:
        """Verify whether an elasticity estimate is mathematically valid and economically reliable.

        Parameters
        ----------
        elasticity : Any
            Estimated price elasticity.
        elasticity_se : Optional[float]
            Standard error of the elasticity estimate.
        p_value : Optional[float]
            Statistical p-value for the elasticity estimate.
        ci_lower : Optional[float]
            Lower bound of confidence interval.
        ci_upper : Optional[float]
            Upper bound of confidence interval.

        Returns
        -------
        Tuple[bool, str]
            (is_reliable, reason_code)
            reason_code is "valid" if reliable, or descriptive "fallback_*" string.
        """
        if elasticity is None:
            return False, "fallback_nan_elasticity"

        try:
            val = float(elasticity)
        except (ValueError, TypeError):
            return False, "fallback_invalid_elasticity"

        if np.isnan(val) or np.isinf(val):
            return False, "fallback_nan_elasticity"

        # Microeconomic Law of Demand: elasticity must be strictly negative
        if val > 0:
            return False, "fallback_positive_elasticity"
        if val == 0.0:
            return False, "fallback_zero_elasticity"

        # Empirical plausibility boundaries
        min_rel = getattr(self, "min_reliable_elasticity", -10.0)
        max_rel = getattr(self, "max_reliable_elasticity", -0.05)
        if val < min_rel or val > max_rel:
            return False, "fallback_unreliable_elasticity"

        # Statistical significance check
        alpha_sig = getattr(self, "alpha_significance", 0.10)
        if p_value is not None and not np.isnan(p_value):
            if float(p_value) > alpha_sig:
                return False, "fallback_insignificant_pvalue"

        # Standard error / precision check (t-statistic < 1.0 indicates noise exceeds signal)
        if elasticity_se is not None and not np.isnan(elasticity_se) and float(elasticity_se) > 0:
            se = float(elasticity_se)
            if abs(val) / se < 1.0:
                return False, "fallback_high_standard_error"

        # Confidence interval check: if confidence interval spans zero, treatment effect is ambiguous
        if ci_lower is not None and ci_upper is not None:
            if not np.isnan(ci_lower) and not np.isnan(ci_upper):
                if float(ci_lower) <= 0.0 <= float(ci_upper):
                    return False, "fallback_ci_spans_zero"

        return True, "valid"

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
        constraints: Optional[Union[ProductConstraints, Dict[str, Any]]] = None,
        elasticity_se: Optional[float] = None,
        p_value: Optional[float] = None,
        ci_lower: Optional[float] = None,
        ci_upper: Optional[float] = None,
        inventory: Optional[float] = None,
        base_qty: Optional[float] = None,
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
        constraints : Optional[Union[ProductConstraints, Dict[str, Any]]]
            Product-level business rules, discounts limits, MAP, MSRP, or freeze flags.
        elasticity_se : Optional[float]
            Standard error of elasticity for reliability verification.
        p_value : Optional[float]
            p-value of elasticity for statistical significance check.
        ci_lower : Optional[float]
            Lower bound of confidence interval.
        ci_upper : Optional[float]
            Upper bound of confidence interval.
        inventory : Optional[float]
            Available inventory / stock for this item.
        base_qty : Optional[float]
            Baseline quantity demanded (needed for inventory clearing calculation).

        Returns
        -------
        Tuple[float, str]
            (optimal_price, bound_hit_description)
        """
        if base_price <= 0:
            raise ValueError(f"base_price must be positive, got {base_price}")

        obj_clean = objective.lower()
        if obj_clean not in ("profit", "revenue"):
            raise ValueError(f"Unknown objective '{objective}'. Choose 'profit' or 'revenue'.")

        # 1. Parse constraints
        if isinstance(constraints, ProductConstraints):
            c_obj = constraints
        elif isinstance(constraints, dict):
            c_obj = ProductConstraints.from_dict(constraints)
        else:
            c_obj = ProductConstraints()

        # 2. Check price freeze constraint
        if c_obj.is_price_frozen:
            return float(base_price), "price_frozen"

        # 3. Check elasticity validity and reliability
        is_reliable, reason = self.validate_elasticity_reliability(
            elasticity=elasticity,
            elasticity_se=elasticity_se,
            p_value=p_value,
            ci_lower=ci_lower,
            ci_upper=ci_upper,
        )
        if not is_reliable:
            return float(base_price), reason

        theta = float(elasticity)

        # 4. Derive unit cost and margin floor
        if unit_cost is None or np.isnan(unit_cost) or unit_cost <= 0:
            cost = base_price * (1.0 - self.default_margin)
        else:
            cost = float(unit_cost)

        eff_min_margin = c_obj.min_margin if c_obj.min_margin is not None else self.min_margin
        cost_floor = cost / (1.0 - eff_min_margin) if eff_min_margin < 1.0 else cost

        # 5. Baseline percentage bounds & discount limits
        min_chg = c_obj.min_price_change_pct if c_obj.min_price_change_pct is not None else self.min_price_change_pct
        max_chg = c_obj.max_price_change_pct if c_obj.max_price_change_pct is not None else self.max_price_change_pct

        # Apply global or product-level discount limit
        global_max_disc = getattr(self, "max_discount_pct", None)
        eff_max_discount = c_obj.max_discount_pct if c_obj.max_discount_pct is not None else global_max_disc
        if eff_max_discount is not None:
            # e.g. eff_max_discount = 0.15 => min_chg cannot be less than -0.15
            min_chg = max(min_chg, -float(eff_max_discount))

        if c_obj.max_markup_pct is not None:
            max_chg = min(max_chg, float(c_obj.max_markup_pct))

        p_lower = base_price * (1.0 + min_chg)
        p_upper = base_price * (1.0 + max_chg)

        # If promotions forbidden, cannot discount below baseline price
        if not c_obj.allow_promotions:
            p_lower = max(p_lower, base_price)

        # Cost-plus minimum margin floor
        p_lower = max(p_lower, cost_floor)

        # Explicit absolute bounds & MAP / MSRP
        eff_min_p = c_obj.min_price if c_obj.min_price is not None else min_price
        if eff_min_p is not None and not np.isnan(eff_min_p):
            p_lower = max(p_lower, float(eff_min_p))

        eff_max_p = c_obj.max_price if c_obj.max_price is not None else max_price
        if eff_max_p is not None and not np.isnan(eff_max_p):
            p_upper = min(p_upper, float(eff_max_p))

        # Minimum Advertised Price (MAP) floor
        if c_obj.map_price is not None and not np.isnan(c_obj.map_price):
            p_lower = max(p_lower, float(c_obj.map_price))

        # MSRP ceiling
        if c_obj.msrp is not None and not np.isnan(c_obj.msrp):
            p_upper = min(p_upper, float(c_obj.msrp))

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

        # 6. Theoretical optimal calculation
        if obj_clean == "profit":
            # Amoroso-Robinson condition: P* = c * (theta / (theta + 1))
            if theta < -1.0:
                raw_optimal = cost * (theta / (theta + 1.0))
            else:
                # Inelastic demand (theta >= -1.0): margin increases faster than volume drops
                raw_optimal = p_upper
                bound_hit = "inelastic_upper_bound"
        elif obj_clean == "revenue":
            # Revenue = P * Q = P^(1 + theta)
            if theta < -1.0:
                # Elastic: lowering price increases revenue -> push to lowest bound
                raw_optimal = p_lower
                bound_hit = "elastic_revenue_lower_bound"
            elif theta > -1.0:
                # Inelastic: raising price increases revenue -> push to highest bound
                raw_optimal = p_upper
                bound_hit = "inelastic_revenue_upper_bound"
            else:
                # Unit elastic (theta == -1.0): revenue is invariant to price
                raw_optimal = base_price
                bound_hit = "unit_elastic_neutral"

        # 7. Inventory-aware market-clearing check
        eff_inv = c_obj.inventory if c_obj.inventory is not None else inventory
        if (
            eff_inv is not None
            and not np.isnan(eff_inv)
            and eff_inv > 0
            and base_qty is not None
            and base_qty > 0
        ):
            # If proposed price produces demand exceeding available inventory,
            # raise price to the market-clearing price to prevent stockout and capture surplus
            safe_ratio = max(raw_optimal / base_price, 1e-4)
            projected_unconstrained_demand = base_qty * (safe_ratio**theta)
            if projected_unconstrained_demand > eff_inv:
                inv_ratio = max(eff_inv / base_qty, 1e-4)
                p_clearing = base_price * (inv_ratio ** (1.0 / theta))
                if p_clearing > raw_optimal:
                    raw_optimal = p_clearing
                    bound_hit = "inventory_clearing_bound"

        # 8. Bound clipping
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

        # 9. Charm pricing
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
        inventory: Optional[float] = None,
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
        inventory : Optional[float]
            Available inventory stock constraint.

        Returns
        -------
        ItemOptimizationResult
            Full counterfactual impact metrics accounting for costs, demand, and inventory.
        """
        if base_price <= 0 or base_qty < 0 or new_price <= 0:
            raise ValueError("base_price, base_qty, and new_price must be positive.")

        cost = (
            float(unit_cost)
            if (unit_cost is not None and not np.isnan(unit_cost) and unit_cost > 0)
            else base_price * (1.0 - self.default_margin)
        )

        is_fallback = (
            bound_hit.startswith("fallback_")
            or elasticity is None
            or np.isnan(elasticity)
            or np.isinf(elasticity)
            or float(elasticity) > 0
        )

        if is_fallback:
            # Safe baseline fallback: preserve baseline metrics with zero uplifts
            new_price = base_price
            projected_qty = base_qty
            base_rev = base_price * base_qty
            proj_rev = base_rev
            base_prof = (base_price - cost) * base_qty
            proj_prof = base_prof
            price_chg_pct = 0.0
            demand_uplift_pct = 0.0
            rev_uplift_pct = 0.0
            prof_uplift_pct = 0.0
            base_margin = (base_price - cost) / base_price if base_price > 0 else 0.0
            proj_margin = base_margin
            stockout_units = 0.0
        else:
            # Constant elasticity demand projection: Q_new = Q_0 * (P_new / P_0)^theta
            price_ratio = new_price / base_price
            safe_ratio = max(min(price_ratio, 1e4), 1e-4)

            if base_qty == 0.0:
                raw_projected_qty = 0.0
            else:
                raw_projected_qty = base_qty * (safe_ratio**float(elasticity))
                raw_projected_qty = max(0.0, raw_projected_qty)

            # Enforce inventory constraints if stock is bounded
            stockout_units = 0.0
            if inventory is not None and not np.isnan(inventory) and inventory >= 0:
                inv_cap = float(inventory)
                if raw_projected_qty > inv_cap:
                    stockout_units = raw_projected_qty - inv_cap
                    projected_qty = inv_cap
                else:
                    projected_qty = raw_projected_qty
            else:
                projected_qty = raw_projected_qty

            # Financial metrics based on actual fulfilled units
            base_rev = base_price * base_qty
            proj_rev = new_price * projected_qty

            base_prof = (base_price - cost) * base_qty
            proj_prof = (new_price - cost) * projected_qty

            price_chg_pct = (new_price - base_price) / base_price
            demand_uplift_pct = (projected_qty - base_qty) / base_qty if base_qty > 0 else 0.0
            rev_uplift_pct = (proj_rev - base_rev) / base_rev if base_rev > 0 else 0.0
            prof_uplift_pct = (proj_prof - base_prof) / base_prof if base_prof != 0 else 0.0

            base_margin = (base_price - cost) / base_price if base_price > 0 else 0.0
            proj_margin = (new_price - cost) / new_price if new_price > 0 else 0.0

        return ItemOptimizationResult(
            baseline_price=base_price,
            optimal_price=new_price,
            price_change_pct=price_chg_pct,
            elasticity=float(elasticity) if (elasticity is not None and not np.isnan(elasticity)) else 0.0,
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
            inventory=inventory,
            stockout_units=stockout_units,
            is_fallback=is_fallback,
            fallback_reason=bound_hit if is_fallback else None,
        )

    def optimize_single_item(
        self,
        base_price: float,
        elasticity: float,
        base_qty: float = 10.0,
        unit_cost: Optional[float] = None,
        objective: str = "profit",
        min_price: Optional[float] = None,
        max_price: Optional[float] = None,
        comp_price: Optional[float] = None,
        comp_guardrail_pct: Optional[float] = None,
        constraints: Optional[Union[ProductConstraints, Dict[str, Any]]] = None,
        elasticity_se: Optional[float] = None,
        p_value: Optional[float] = None,
        ci_lower: Optional[float] = None,
        ci_upper: Optional[float] = None,
        inventory: Optional[float] = None,
    ) -> ItemOptimizationResult:
        """One-stop helper to calculate constrained optimal price and simulate uplift."""
        p_opt, flag = self.calculate_optimal_price(
            base_price=base_price,
            elasticity=elasticity,
            unit_cost=unit_cost,
            objective=objective,
            min_price=min_price,
            max_price=max_price,
            comp_price=comp_price,
            comp_guardrail_pct=comp_guardrail_pct,
            constraints=constraints,
            elasticity_se=elasticity_se,
            p_value=p_value,
            ci_lower=ci_lower,
            ci_upper=ci_upper,
            inventory=inventory,
            base_qty=base_qty,
        )
        return self.simulate_item_uplift(
            base_price=base_price,
            base_qty=base_qty,
            new_price=p_opt,
            elasticity=elasticity,
            unit_cost=unit_cost,
            objective=objective,
            bound_hit=flag,
            inventory=inventory,
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
        inventory_col: Optional[str] = None,
        product_id_col: Optional[str] = "product_id",
        category_col: Optional[str] = "product_category_name",
        product_constraints: Optional[Dict[str, Union[ProductConstraints, Dict[str, Any]]]] = None,
        category_constraints: Optional[Dict[str, Union[ProductConstraints, Dict[str, Any]]]] = None,
        max_total_discount_budget: Optional[float] = None,
        max_total_inventory_capacity: Optional[float] = None,
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
        inventory_col : Optional[str]
            Column name for item available inventory stock.
        product_id_col : Optional[str], default="product_id"
            Column name for product unique identifier.
        category_col : Optional[str], default="product_category_name"
            Column name for product category.
        product_constraints : Optional[Dict[str, Any]]
            Mapping of product_id -> ProductConstraints.
        category_constraints : Optional[Dict[str, Any]]
            Mapping of category_name -> ProductConstraints.
        max_total_discount_budget : Optional[float]
            Maximum allowable portfolio promotional discount budget ($).
        max_total_inventory_capacity : Optional[float]
            Maximum allowable portfolio fulfillment volume (units).

        Returns
        -------
        pd.DataFrame
            Copy of input DataFrame augmented with optimal prices, guardrails, and uplift projections.
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
        base_margins = []
        proj_margins = []
        is_fallbacks = []
        stockouts = []

        # Detect potential unit cost column
        actual_cost_col = cost_col
        if actual_cost_col is None:
            for cand in ["unit_cost", "cost", "cogs"]:
                if cand in out.columns:
                    actual_cost_col = cand
                    break

        # Detect potential inventory column
        inv_col = inventory_col
        if inv_col is None:
            for cand in ["inventory", "stock", "stock_qty", "inventory_level"]:
                if cand in out.columns:
                    inv_col = cand
                    break

        for _, row in out.iterrows():
            bp = float(row[price_col]) if not pd.isna(row[price_col]) else 0.0
            b_qty = float(row[qty_col]) if not pd.isna(row[qty_col]) else 0.0
            raw_theta = row[elasticity_col]
            theta = float(raw_theta) if not pd.isna(raw_theta) else np.nan

            c = (
                float(row[actual_cost_col])
                if (actual_cost_col and actual_cost_col in row and not pd.isna(row[actual_cost_col]))
                else None
            )
            comp = (
                float(row[comp_price_col])
                if (comp_price_col and comp_price_col in row and not pd.isna(row[comp_price_col]))
                else None
            )
            inv = float(row[inv_col]) if (inv_col and inv_col in row and not pd.isna(row[inv_col])) else None

            # Gather product-level constraints from dictionary or dataframe columns
            item_constraints_dict: Dict[str, Any] = {}

            # Category-level defaults
            if category_constraints and category_col in row and row[category_col] in category_constraints:
                cat_c = category_constraints[row[category_col]]
                if isinstance(cat_c, ProductConstraints):
                    item_constraints_dict.update({k: v for k, v in cat_c.__dict__.items() if v is not None})
                elif isinstance(cat_c, dict):
                    item_constraints_dict.update(cat_c)

            # Product-specific overrides
            if product_constraints and product_id_col in row and row[product_id_col] in product_constraints:
                prod_c = product_constraints[row[product_id_col]]
                if isinstance(prod_c, ProductConstraints):
                    item_constraints_dict.update({k: v for k, v in prod_c.__dict__.items() if v is not None})
                elif isinstance(prod_c, dict):
                    item_constraints_dict.update(prod_c)

            # In-row column overrides
            for col_name in [
                "min_price",
                "max_price",
                "max_discount_pct",
                "max_markup_pct",
                "min_margin",
                "map_price",
                "msrp",
                "is_price_frozen",
                "allow_promotions",
            ]:
                if col_name in row and not pd.isna(row[col_name]):
                    item_constraints_dict[col_name] = row[col_name]

            # In-row statistical uncertainty columns
            se = float(row["elasticity_se"]) if ("elasticity_se" in row and not pd.isna(row["elasticity_se"])) else None
            p_val = float(row["p_value"]) if ("p_value" in row and not pd.isna(row["p_value"])) else None
            ci_low = float(row["ci_lower"]) if ("ci_lower" in row and not pd.isna(row["ci_lower"])) else None
            ci_high = float(row["ci_upper"]) if ("ci_upper" in row and not pd.isna(row["ci_upper"])) else None

            # Safe execution
            if bp <= 0 or b_qty < 0:
                p_opt = bp
                b_flag = "fallback_invalid_inputs"
            else:
                p_opt, b_flag = self.calculate_optimal_price(
                    base_price=bp,
                    elasticity=theta,
                    unit_cost=c,
                    objective=objective,
                    comp_price=comp,
                    constraints=item_constraints_dict,
                    elasticity_se=se,
                    p_value=p_val,
                    ci_lower=ci_low,
                    ci_upper=ci_high,
                    inventory=inv,
                    base_qty=b_qty,
                )

            if bp <= 0 or b_qty < 0:
                res = ItemOptimizationResult(
                    baseline_price=bp,
                    optimal_price=bp,
                    price_change_pct=0.0,
                    elasticity=theta if not np.isnan(theta) else 0.0,
                    unit_cost=c if c is not None else 0.0,
                    baseline_qty=b_qty,
                    projected_qty=b_qty,
                    demand_uplift_pct=0.0,
                    baseline_revenue=bp * b_qty,
                    projected_revenue=bp * b_qty,
                    revenue_uplift_pct=0.0,
                    baseline_profit=0.0,
                    projected_profit=0.0,
                    profit_uplift_pct=0.0,
                    baseline_margin=0.0,
                    projected_margin=0.0,
                    bound_hit=b_flag,
                    objective=objective,
                    is_fallback=True,
                    fallback_reason=b_flag,
                )
            else:
                res = self.simulate_item_uplift(
                    base_price=bp,
                    base_qty=b_qty,
                    new_price=p_opt,
                    elasticity=theta,
                    unit_cost=c,
                    objective=objective,
                    bound_hit=b_flag,
                    inventory=inv,
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
            base_margins.append(res.baseline_margin)
            proj_margins.append(res.projected_margin)
            is_fallbacks.append(res.is_fallback)
            stockouts.append(res.stockout_units)

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
        out["baseline_margin"] = base_margins
        out["projected_margin"] = proj_margins
        out["is_fallback"] = is_fallbacks
        out["stockout_units"] = stockouts

        # -------------------------------------------------------------
        # Portfolio-level constraints: Promotional Budget & Inventory Capacity
        # -------------------------------------------------------------
        budget = (
            max_total_discount_budget
            if max_total_discount_budget is not None
            else self.global_promotional_budget
        )
        if budget is not None and budget >= 0:
            # Calculate total promotional spend on discounted items
            discounts = np.maximum(0.0, out[price_col] - out["optimal_price"]) * out["projected_qty"]
            total_discount_spend = float(discounts.sum())

            if total_discount_spend > budget:
                # Proportional scaling factor for discounts
                scale = budget / total_discount_spend if total_discount_spend > 0 else 0.0
                for idx in out.index:
                    bp = float(out.at[idx, price_col])
                    p_opt = float(out.at[idx, "optimal_price"])
                    if p_opt < bp:
                        adjusted_price = bp - (bp - p_opt) * scale
                        inv = float(out.at[idx, inv_col]) if (inv_col and inv_col in out.columns and not pd.isna(out.at[idx, inv_col])) else None
                        c = float(out.at[idx, "unit_cost"])
                        theta = float(out.at[idx, elasticity_col])
                        b_qty = float(out.at[idx, qty_col])

                        res_adj = self.simulate_item_uplift(
                            base_price=bp,
                            base_qty=b_qty,
                            new_price=adjusted_price,
                            elasticity=theta,
                            unit_cost=c,
                            objective=objective,
                            bound_hit="budget_constrained_discount",
                            inventory=inv,
                        )
                        out.at[idx, "optimal_price"] = res_adj.optimal_price
                        out.at[idx, "bound_hit"] = res_adj.bound_hit
                        out.at[idx, "projected_qty"] = res_adj.projected_qty
                        out.at[idx, "projected_revenue"] = res_adj.projected_revenue
                        out.at[idx, "projected_profit"] = res_adj.projected_profit
                        out.at[idx, "price_change_pct"] = res_adj.price_change_pct
                        out.at[idx, "demand_uplift_pct"] = res_adj.demand_uplift_pct
                        out.at[idx, "revenue_uplift_pct"] = res_adj.revenue_uplift_pct
                        out.at[idx, "profit_uplift_pct"] = res_adj.profit_uplift_pct
                        out.at[idx, "projected_margin"] = res_adj.projected_margin

        capacity = (
            max_total_inventory_capacity
            if max_total_inventory_capacity is not None
            else self.global_inventory_limit
        )
        if capacity is not None and capacity > 0:
            total_proj_vol = float(out["projected_qty"].sum())
            if total_proj_vol > capacity:
                # Scale back volume on items with positive volume demand expansions
                scale_vol = capacity / total_proj_vol
                for idx in out.index:
                    bp = float(out.at[idx, price_col])
                    p_opt = float(out.at[idx, "optimal_price"])
                    if p_opt < bp:
                        # Price reduction drove demand expansion; damp price discount
                        adjusted_price = bp - (bp - p_opt) * scale_vol
                        inv = float(out.at[idx, inv_col]) if (inv_col and inv_col in out.columns and not pd.isna(out.at[idx, inv_col])) else None
                        c = float(out.at[idx, "unit_cost"])
                        theta = float(out.at[idx, elasticity_col])
                        b_qty = float(out.at[idx, qty_col])

                        res_adj = self.simulate_item_uplift(
                            base_price=bp,
                            base_qty=b_qty,
                            new_price=adjusted_price,
                            elasticity=theta,
                            unit_cost=c,
                            objective=objective,
                            bound_hit="portfolio_capacity_constrained",
                            inventory=inv,
                        )
                        out.at[idx, "optimal_price"] = res_adj.optimal_price
                        out.at[idx, "bound_hit"] = res_adj.bound_hit
                        out.at[idx, "projected_qty"] = res_adj.projected_qty
                        out.at[idx, "projected_revenue"] = res_adj.projected_revenue
                        out.at[idx, "projected_profit"] = res_adj.projected_profit
                        out.at[idx, "price_change_pct"] = res_adj.price_change_pct
                        out.at[idx, "demand_uplift_pct"] = res_adj.demand_uplift_pct
                        out.at[idx, "revenue_uplift_pct"] = res_adj.revenue_uplift_pct
                        out.at[idx, "profit_uplift_pct"] = res_adj.profit_uplift_pct
                        out.at[idx, "projected_margin"] = res_adj.projected_margin

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
            theta = float(row[elasticity_col]) if not pd.isna(row[elasticity_col]) else 0.0
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
