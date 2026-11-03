"""Empirical Day 5 simulation script for EconoCausal Dynamic Pricing Optimizer."""

import os
import sys

# Ensure repo root is on sys.path
repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if repo_root not in sys.path:
    sys.path.insert(0, repo_root)

import numpy as np
import pandas as pd

from src.data.load_data import load_retail_price_dataset, extract_causal_matrices
from src.data.preprocess import CausalPreprocessor, split_causal_data
from src.models.train_dml import DoubleMachineLearning
from src.models.cate import CATEEstimator
from src.models.optimizer import DynamicPricingOptimizer


def run_simulation():
    repo_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    data_path = os.path.join(repo_dir, "data", "raw", "retail_price.csv")

    # 1. Load data
    df = load_retail_price_dataset(data_path)

    # 2. Split and Preprocess with anti-leakage guarantee
    splits = split_causal_data(
        df,
        test_size=0.20,
        random_state=42,
        categorical_confounders=["product_category_name"],
        log_transform_treatment=True,
        log_transform_outcome=True,
    )
    X_train = splits["X_train"]
    X_test = splits["X_test"]
    T_train = splits["T_train"]
    Y_train = splits["Y_train"]

    # 3. Fit DML + CATE on train set
    dml = DoubleMachineLearning(random_state=42)
    cate = CATEEstimator(dml_base=dml, random_state=42)
    cate.fit(X_train, T_train, Y_train)

    # Predict heterogeneous elasticity on holdout test set
    test_cate = cate.predict_cate(X_test)

    # 4. Attach elasticity to holdout test dataframe
    test_indices = splits["X_test"].index
    test_df = df.loc[test_indices].copy()
    test_df["elasticity"] = test_cate

    # 5. Initialize Optimizer
    opt = DynamicPricingOptimizer(
        default_margin=0.30,
        min_price_change_pct=-0.30,
        max_price_change_pct=0.20,
        min_margin=0.05,
        charm_pricing=True,
    )

    optimized_catalog = opt.optimize_catalog(test_df, objective="profit")
    port_summary = opt.simulate_portfolio_uplift(optimized_catalog)
    scenarios = opt.compare_pricing_scenarios(test_df, comp_col="comp_1")

    print(opt.format_summary_report(port_summary, scenarios))

    print("\n" + "=" * 80)
    print("CATEGORY-LEVEL DYNAMIC PRICING BREAKDOWN:")
    print("=" * 80)
    cat_summary = optimized_catalog.groupby("product_category_name").agg(
        item_count=("unit_price", "count"),
        mean_elasticity=("elasticity", "mean"),
        avg_base_price=("unit_price", "mean"),
        avg_opt_price=("optimal_price", "mean"),
        total_base_profit=("baseline_profit", "sum"),
        total_proj_profit=("projected_profit", "sum"),
        total_base_rev=("baseline_revenue", "sum"),
        total_proj_rev=("projected_revenue", "sum"),
    ).reset_index()

    cat_summary["cat_profit_gain"] = cat_summary["total_proj_profit"] - cat_summary["total_base_profit"]
    cat_summary["cat_profit_uplift_pct"] = (cat_summary["cat_profit_gain"] / cat_summary["total_base_profit"]) * 100
    cat_summary["cat_price_chg_pct"] = ((cat_summary["avg_opt_price"] - cat_summary["avg_base_price"]) / cat_summary["avg_base_price"]) * 100

    for _, r in cat_summary.iterrows():
        print(
            f"{r['product_category_name']:<24} | Count: {r['item_count']:2d} | "
            f"Elast: {r['mean_elasticity']:.3f} | "
            f"BaseP: ${r['avg_base_price']:6.2f} -> OptP: ${r['avg_opt_price']:6.2f} ({r['cat_price_chg_pct']:+5.1f}%) | "
            f"Prof Gain: ${r['cat_profit_gain']:+9.2f} ({r['cat_profit_uplift_pct']:+5.1f}%)"
        )
    print("=" * 80)


if __name__ == "__main__":
    run_simulation()
