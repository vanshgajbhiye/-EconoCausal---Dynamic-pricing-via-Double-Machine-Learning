"""Interactive Dashboard Components & Mathematical Visualizers for EconoCausal.

Provides dynamic pricing curves, microeconomic simulation math,
competitor reaction models, KPI computations, and Plotly visualizer builders.
"""

from typing import Any, Dict, List, Optional, Tuple, Union
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots


def compute_pricing_curves(
    base_price: float,
    base_qty: float,
    unit_cost: float,
    elasticity: float,
    p_min: Optional[float] = None,
    p_max: Optional[float] = None,
    n_points: int = 150,
) -> pd.DataFrame:
    """Compute demand, revenue, and gross profit curves across a fine price grid.

    Formulation:
        Q(P) = Q_0 * (P / P_0)^theta
        R(P) = P * Q(P)
        Pi(P) = (P - c) * Q(P)
        Margin(P) = (P - c) / P

    Parameters
    ----------
    base_price : float
        Current historical baseline price P_0 > 0.
    base_qty : float
        Current historical baseline volume Q_0 > 0.
    unit_cost : float
        Unit marginal cost c > 0.
    elasticity : float
        Causal price elasticity theta (typically negative).
    p_min : Optional[float], default=None
        Lower bound of price grid. Defaults to max(unit_cost * 1.01, base_price * 0.50).
    p_max : Optional[float], default=None
        Upper bound of price grid. Defaults to base_price * 1.60.
    n_points : int, default=150
        Number of grid evaluation points.

    Returns
    -------
    pd.DataFrame
        DataFrame columns: ['price', 'demand', 'revenue', 'profit', 'margin_pct'].
    """
    if base_price <= 0 or base_qty <= 0:
        raise ValueError("base_price and base_qty must be strictly positive.")

    # Guard cost
    cost = float(unit_cost) if unit_cost > 0 else base_price * 0.70

    # Grid boundaries
    grid_min = float(p_min) if p_min is not None and p_min > 0 else max(cost * 1.01, base_price * 0.50)
    grid_max = float(p_max) if p_max is not None and p_max > grid_min else base_price * 1.60

    if grid_max <= grid_min:
        grid_max = grid_min * 1.50

    prices = np.linspace(grid_min, grid_max, n_points)
    price_ratios = np.maximum(prices / base_price, 1e-4)

    # Constant elasticity demand
    demands = np.maximum(base_qty * (price_ratios ** elasticity), 0.0)
    revenues = prices * demands
    profits = (prices - cost) * demands
    margins = np.where(prices > 0, ((prices - cost) / prices) * 100.0, 0.0)

    return pd.DataFrame(
        {
            "price": prices,
            "demand": demands,
            "revenue": revenues,
            "profit": profits,
            "margin_pct": margins,
        }
    )


def simulate_competitor_response(
    base_price: float,
    our_new_price: float,
    competitor_base_price: float,
    base_qty: float,
    unit_cost: float,
    own_elasticity: float,
    cross_elasticity: float = 0.40,
    response_mode: str = "static",
    custom_comp_price: Optional[float] = None,
) -> Dict[str, Any]:
    """Simulate game-theoretic competitor reactions and net demand impact.

    Cross-Elasticity Formulation:
        Q_adj = Q_0 * (P_our / P_our_0)^theta_own * (P_comp / P_comp_0)^theta_cross

    Parameters
    ----------
    base_price : float
        Our baseline price.
    our_new_price : float
        Our proposed new price.
    competitor_base_price : float
        Competitor baseline price.
    base_qty : float
        Our baseline sales volume.
    unit_cost : float
        Our unit marginal cost.
    own_elasticity : float
        Our own-price causal elasticity theta_own (< 0).
    cross_elasticity : float, default=0.40
        Cross-price elasticity with respect to competitor price theta_cross (> 0).
    response_mode : str, default="static"
        Competitor reaction mode:
        - "static": Competitor keeps price unchanged.
        - "match": Competitor matches our price exactly.
        - "undercut_5": Competitor undercuts our price by 5%.
        - "undercut_10": Competitor aggressively undercuts by 10%.
        - "cooperative": Competitor follows our price change proportionally.
        - "custom": Competitor sets custom_comp_price.
    custom_comp_price : Optional[float], default=None
        Custom competitor price when response_mode="custom".

    Returns
    -------
    Dict[str, Any]
        Detailed simulation metrics including our new profit, demand, and risk rating.
    """
    if competitor_base_price <= 0:
        competitor_base_price = base_price

    mode = response_mode.lower()
    if mode == "static":
        comp_new_price = competitor_base_price
        mode_label = "Static / No Reaction"
    elif mode == "match":
        comp_new_price = our_new_price
        mode_label = "Price Matching"
    elif mode == "undercut_5":
        comp_new_price = our_new_price * 0.95
        mode_label = "Moderate Undercut (-5%)"
    elif mode == "undercut_10":
        comp_new_price = our_new_price * 0.90
        mode_label = "Aggressive Undercut (-10%)"
    elif mode == "cooperative":
        # Competitor shifts price in same percentage as our move
        pct_change = (our_new_price - base_price) / base_price
        comp_new_price = competitor_base_price * (1.0 + pct_change)
        mode_label = "Cooperative Parallel Shift"
    elif mode == "custom" and custom_comp_price is not None and custom_comp_price > 0:
        comp_new_price = float(custom_comp_price)
        mode_label = f"Custom Price (${comp_new_price:.2f})"
    else:
        comp_new_price = competitor_base_price
        mode_label = "Static / No Reaction"

    # Safeguard ratios
    our_price_ratio = max(our_new_price / base_price, 1e-4)
    comp_price_ratio = max(comp_new_price / competitor_base_price, 1e-4)

    # Net adjusted demand incorporating cross-elasticity
    # If competitor cuts price (comp_price_ratio < 1), cross_elasticity > 0 causes demand to decline
    adjusted_qty = base_qty * (our_price_ratio ** own_elasticity) * (comp_price_ratio ** cross_elasticity)
    adjusted_qty = max(float(adjusted_qty), 0.0)

    # Financials
    base_rev = base_price * base_qty
    base_profit = (base_price - unit_cost) * base_qty
    base_margin = (base_price - unit_cost) / base_price if base_price > 0 else 0.0

    adj_rev = our_new_price * adjusted_qty
    adj_profit = (our_new_price - unit_cost) * adjusted_qty
    adj_margin = (our_new_price - unit_cost) / our_new_price if our_new_price > 0 else 0.0

    profit_uplift_pct = (adj_profit - base_profit) / base_profit if base_profit != 0 else 0.0
    revenue_uplift_pct = (adj_rev - base_rev) / base_rev if base_rev > 0 else 0.0
    demand_uplift_pct = (adjusted_qty - base_qty) / base_qty if base_qty > 0 else 0.0
    comp_price_chg_pct = (comp_new_price - competitor_base_price) / competitor_base_price

    # Strategic Risk Assessment
    if profit_uplift_pct < -0.10:
        risk_level = "High Risk"
        risk_color = "#e63946"
        advice = "Competitor retaliation erodes profitability. Consider narrowing price adjustment or offering non-price value incentives."
    elif profit_uplift_pct < 0.0:
        risk_level = "Moderate Risk"
        risk_color = "#f4a261"
        advice = "Slight profit erosion due to competitor move. Close monitoring of sales velocity advised."
    elif profit_uplift_pct < 0.15:
        risk_level = "Low Risk / Resilient"
        risk_color = "#2a9d8f"
        advice = "Strategy maintains positive profit gains even after accounting for competitor reaction."
    else:
        risk_level = "Highly Profitable / Dominant"
        risk_color = "#2a9d8f"
        advice = "Robust strategic outcome. Significant margin and profit gains withstand competitor response."

    return {
        "response_mode": mode,
        "mode_label": mode_label,
        "competitor_base_price": competitor_base_price,
        "competitor_new_price": comp_new_price,
        "competitor_price_change_pct": comp_price_chg_pct,
        "our_new_price": our_new_price,
        "base_qty": base_qty,
        "adjusted_qty": adjusted_qty,
        "demand_uplift_pct": demand_uplift_pct,
        "base_revenue": base_rev,
        "adjusted_revenue": adj_rev,
        "revenue_uplift_pct": revenue_uplift_pct,
        "base_profit": base_profit,
        "adjusted_profit": adj_profit,
        "profit_uplift_pct": profit_uplift_pct,
        "base_margin": base_margin,
        "adjusted_margin": adj_margin,
        "risk_level": risk_level,
        "risk_color": risk_color,
        "advice": advice,
    }


def calculate_catalog_kpis(recommendations_df: pd.DataFrame) -> Dict[str, Any]:
    """Compute aggregate executive KPIs and portfolio performance metrics.

    Parameters
    ----------
    recommendations_df : pd.DataFrame
        DataFrame produced by EconoCausalPipeline.recommend_prices().

    Returns
    -------
    Dict[str, Any]
        Dictionary of portfolio summary figures.
    """
    total_base_rev = float(recommendations_df["baseline_revenue"].sum())
    total_opt_rev = float(recommendations_df["projected_revenue"].sum())
    rev_uplift_pct = (total_opt_rev - total_base_rev) / total_base_rev if total_base_rev > 0 else 0.0

    total_base_prof = float(recommendations_df["baseline_profit"].sum())
    total_opt_prof = float(recommendations_df["projected_profit"].sum())
    prof_uplift_pct = (total_opt_prof - total_base_prof) / total_base_prof if total_base_prof != 0 else 0.0

    total_base_qty = float(recommendations_df["baseline_qty"].sum())
    total_opt_qty = float(recommendations_df["projected_qty"].sum())
    qty_uplift_pct = (total_opt_qty - total_base_qty) / total_base_qty if total_base_qty > 0 else 0.0

    avg_base_margin = total_base_prof / total_base_rev if total_base_rev > 0 else 0.0
    avg_opt_margin = total_opt_prof / total_opt_rev if total_opt_rev > 0 else 0.0
    margin_expansion_bps = (avg_opt_margin - avg_base_margin) * 10000.0

    # Pricing Action Counts
    action_counts = recommendations_df["action"].value_counts().to_dict()
    total_items = len(recommendations_df)

    # Elasticity statistics
    elasticities = recommendations_df["elasticity"]
    mean_elasticity = float(elasticities.mean())
    median_elasticity = float(elasticities.median())

    return {
        "total_items": total_items,
        "total_base_revenue": total_base_rev,
        "total_optimized_revenue": total_opt_rev,
        "revenue_uplift_dollars": total_opt_rev - total_base_rev,
        "revenue_uplift_pct": rev_uplift_pct,
        "total_base_profit": total_base_prof,
        "total_optimized_profit": total_opt_prof,
        "profit_uplift_dollars": total_opt_prof - total_base_prof,
        "profit_uplift_pct": prof_uplift_pct,
        "total_base_qty": total_base_qty,
        "total_optimized_qty": total_opt_qty,
        "qty_uplift_pct": qty_uplift_pct,
        "avg_base_margin": avg_base_margin,
        "avg_optimized_margin": avg_opt_margin,
        "margin_expansion_bps": margin_expansion_bps,
        "action_counts": action_counts,
        "mean_elasticity": mean_elasticity,
        "median_elasticity": median_elasticity,
    }


def build_price_elasticity_curve_chart(
    curve_df: pd.DataFrame,
    base_price: float,
    optimal_price: float,
    unit_cost: float,
    comp_price: Optional[float] = None,
    p_lower: Optional[float] = None,
    p_upper: Optional[float] = None,
    currency: str = "$",
) -> go.Figure:
    """Build an interactive dual-axis microeconomic curve visualization in Plotly.

    Plots:
    - Primary Axis (Left): Demand Volume Q(P) and Gross Profit Pi(P)
    - Secondary Axis (Right): Total Revenue R(P)
    - Vertical lines for Baseline, Optimal, Competitor, and Guardrails.
    """
    fig = make_subplots(
        specs=[[{"secondary_y": True}]],
    )

    # Shaded Guardrail Region if bounds given
    if p_lower is not None and p_upper is not None and p_upper > p_lower:
        fig.add_vrect(
            x0=p_lower,
            x1=p_upper,
            fillcolor="rgba(42, 157, 143, 0.12)",
            layer="below",
            line_width=0,
            annotation_text="Permissible Range",
            annotation_position="top left",
            annotation_font_size=11,
            annotation_font_color="#2a9d8f",
        )

    # 1. Demand Curve (Units) - Left Axis
    fig.add_trace(
        go.Scatter(
            x=curve_df["price"],
            y=curve_df["demand"],
            name="Demand Volume Q(P)",
            mode="lines",
            line=dict(color="#457b9d", width=2.5, dash="dot"),
            hovertemplate="Price: " + currency + "%{x:.2f}<br>Demand: %{y:.1f} units<extra></extra>",
        ),
        secondary_y=False,
    )

    # 2. Profit Curve (Profit $) - Left Axis
    fig.add_trace(
        go.Scatter(
            x=curve_df["price"],
            y=curve_df["profit"],
            name="Gross Profit \u03a0(P)",
            mode="lines",
            line=dict(color="#2a9d8f", width=3.5),
            hovertemplate="Price: " + currency + "%{x:.2f}<br>Gross Profit: " + currency + "%{y:,.2f}<extra></extra>",
        ),
        secondary_y=False,
    )

    # 3. Revenue Curve (Revenue $) - Right Axis
    fig.add_trace(
        go.Scatter(
            x=curve_df["price"],
            y=curve_df["revenue"],
            name="Total Revenue R(P)",
            mode="lines",
            line=dict(color="#e76f51", width=2.5),
            hovertemplate="Price: " + currency + "%{x:.2f}<br>Total Revenue: " + currency + "%{y:,.2f}<extra></extra>",
        ),
        secondary_y=True,
    )

    # Vertical Reference Marker: Baseline Price
    fig.add_vline(
        x=base_price,
        line_width=2,
        line_dash="dash",
        line_color="#6c757d",
        annotation_text=f"Current: {currency}{base_price:.2f}",
        annotation_position="top left",
        annotation_font_size=11,
    )

    # Vertical Reference Marker: Causal Optimal Price
    fig.add_vline(
        x=optimal_price,
        line_width=2.5,
        line_dash="solid",
        line_color="#1d3557",
        annotation_text=f"Optimal P*: {currency}{optimal_price:.2f}",
        annotation_position="top right",
        annotation_font_size=12,
        annotation_font_color="#1d3557",
    )

    # Vertical Reference Marker: Competitor Benchmark (if available)
    if comp_price is not None and comp_price > 0:
        fig.add_vline(
            x=comp_price,
            line_width=1.5,
            line_dash="dot",
            line_color="#e63946",
            annotation_text=f"Competitor: {currency}{comp_price:.2f}",
            annotation_position="bottom right",
            annotation_font_size=10,
            annotation_font_color="#e63946",
        )

    # Vertical Reference Marker: Unit Marginal Cost
    if unit_cost > 0:
        fig.add_vline(
            x=unit_cost,
            line_width=1.5,
            line_dash="longdashdot",
            line_color="#e76f51",
            annotation_text=f"Cost Floor: {currency}{unit_cost:.2f}",
            annotation_position="bottom left",
            annotation_font_size=10,
            annotation_font_color="#e76f51",
        )

    # Layout styling
    fig.update_layout(
        title=dict(
            text="<b>Causal Demand & Profit Surface Across Prices</b>",
            font=dict(size=16),
        ),
        xaxis=dict(
            title="<b>Price (" + currency + ")</b>",
            showgrid=True,
            gridcolor="rgba(200, 200, 200, 0.2)",
        ),
        yaxis=dict(
            title="<b>Demand Volume (Units) & Gross Profit (" + currency + ")</b>",
            showgrid=True,
            gridcolor="rgba(200, 200, 200, 0.2)",
        ),
        yaxis2=dict(
            title="<b>Total Revenue (" + currency + ")</b>",
            overlaying="y",
            side="right",
            showgrid=False,
        ),
        hovermode="x unified",
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="center",
            x=0.5,
        ),
        margin=dict(l=60, r=60, t=80, b=50),
        template="plotly_white",
    )

    return fig


def build_scenario_waterfall_chart(scenario_df: pd.DataFrame) -> go.Figure:
    """Build a multi-strategy comparison bar chart comparing financial outcomes."""
    fig = go.Figure()

    strategies = scenario_df["strategy"].tolist()
    profits = scenario_df["total_profit"].tolist()
    revenues = scenario_df["total_revenue"].tolist()

    # Revenue Bar
    fig.add_trace(
        go.Bar(
            name="Total Revenue ($)",
            x=strategies,
            y=revenues,
            marker_color="#457b9d",
            text=[f"${v:,.0f}" for v in revenues],
            textposition="outside",
        )
    )

    # Profit Bar
    fig.add_trace(
        go.Bar(
            name="Gross Profit ($)",
            x=strategies,
            y=profits,
            marker_color="#2a9d8f",
            text=[f"${v:,.0f}" for v in profits],
            textposition="outside",
        )
    )

    fig.update_layout(
        title="<b>Counterfactual Strategic Pricing Comparison across 6 Policies</b>",
        barmode="group",
        xaxis_title="<b>Pricing Strategy</b>",
        yaxis_title="<b>Financial Total ($)</b>",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="center", x=0.5),
        template="plotly_white",
        margin=dict(l=50, r=50, t=80, b=60),
    )

    return fig


def build_category_elasticity_chart(category_summary_df: pd.DataFrame) -> go.Figure:
    """Build an elasticity ranking bar chart across product categories."""
    sorted_df = category_summary_df.sort_values(by="mean_elasticity", ascending=True)

    fig = go.Figure()

    # Color code bars: highly elastic (dark teal), unit elastic (slate)
    colors = [
        "#1d3557" if e < -1.4 else ("#457b9d" if e < -1.0 else "#e76f51")
        for e in sorted_df["mean_elasticity"]
    ]

    fig.add_trace(
        go.Bar(
            x=sorted_df["mean_elasticity"],
            y=sorted_df["product_category_name"],
            orientation="h",
            marker_color=colors,
            text=[f"\u03b8 = {e:.3f}" for e in sorted_df["mean_elasticity"]],
            textposition="inside",
        )
    )

    # Add vertical line at unit elasticity -1.0
    fig.add_vline(
        x=-1.0,
        line_width=2,
        line_dash="dash",
        line_color="#e63946",
        annotation_text="Unit Elasticity (\u03b8 = -1.0)",
        annotation_position="top left",
        annotation_font_color="#e63946",
    )

    fig.update_layout(
        title="<b>Heterogeneous CATE Price Elasticity by Product Category</b>",
        xaxis_title="<b>Estimated Causal Price Elasticity (\u03b8)</b>",
        yaxis_title="<b>Product Category</b>",
        template="plotly_white",
        margin=dict(l=140, r=40, t=60, b=50),
    )

    return fig
