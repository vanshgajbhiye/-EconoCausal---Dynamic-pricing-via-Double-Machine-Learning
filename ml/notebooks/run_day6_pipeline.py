"""Day 6 Demonstration Script: End-to-End Orchestration & Inference Pipeline.

Demonstrates:
1. Full Pipeline Training with Double Machine Learning & CATE Heterogeneous Elasticity
2. Causal Refutation Diagnostic Audit (Placebo, Random Common Cause, Data Subsetting)
3. Production Model Artifact Serialization & Deserialization via Joblib
4. Real-time Single-SKU Pricing Inference with Economic Guardrails
5. Batch Catalog Pricing Recommendations Export (CSV)
6. Counterfactual Pricing Strategy Scenario Comparison Matrix
"""

import os
import sys

# Ensure repository root is on sys.path
repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if repo_root not in sys.path:
    sys.path.insert(0, repo_root)

import pandas as pd
from src.data.load_data import load_retail_price_dataset
from src.pipeline import EconoCausalPipeline


def run_day6_demonstration():
    print("=" * 80)
    print("       ECONOCAUSAL DAY 6: END-TO-END ORCHESTRATION & INFERENCE PIPELINE")
    print("=" * 80)

    # 1. Dataset Ingestion
    data_path = os.path.join(repo_root, "data", "raw", "retail_price.csv")
    print(f"\n[Step 1] Loading official Kaggle dataset from:\n  -> {data_path}")
    df = load_retail_price_dataset(data_path)
    print(f"  -> Total records loaded: {len(df)}")
    print(f"  -> Available categories: {df['product_category_name'].nunique()} unique segments")

    # 2. Pipeline Initialization & Training
    print("\n[Step 2] Initializing and fitting EconoCausalPipeline...")
    pipeline = EconoCausalPipeline(
        n_splits=5,
        default_margin=0.30,
        min_price_change_pct=-0.30,
        max_price_change_pct=0.20,
        min_margin=0.05,
        charm_pricing=True,
        random_state=42,
    )

    pipeline.fit(
        df=df,
        test_size=0.20,
        run_refutations=True,
        refutation_repeats=2,
    )

    # Print Executive Audit Summary
    print("\n" + pipeline.generate_executive_summary())

    # 3. Model Persistence (Save & Load)
    model_dir = os.path.join(repo_root, "models")
    model_path = os.path.join(model_dir, "econocausal_pipeline.joblib")
    print(f"\n[Step 3] Persisting trained pipeline artifact to:\n  -> {model_path}")
    saved_path = pipeline.save(model_path)
    file_size_kb = os.path.getsize(saved_path) / 1024.0
    print(f"  -> Successfully saved ({file_size_kb:.1f} KB).")

    print("\n[Step 4] Testing model deserialization (loading artifact from disk)...")
    loaded_pipeline = EconoCausalPipeline.load(saved_path)
    print(f"  -> Loaded pipeline verified. Fitted status: {loaded_pipeline.is_fitted_}")
    print(f"  -> Verified Baseline ATE: {loaded_pipeline.baseline_ate_:.4f}")

    # 4. Real-time Single-SKU Pricing Inference
    print("\n[Step 5] Executing Real-Time Single-SKU Pricing Inferences:")
    sample_skus = [
        {
            "name": "Health & Beauty Care Item",
            "features": {
                "product_category_name": "health_beauty",
                "unit_price": 105.00,
                "qty": 25.0,
                "comp_1": 108.00,
                "product_score": 4.2,
                "customers": 150,
            },
        },
        {
            "name": "Computers & Electronics Accessory",
            "features": {
                "product_category_name": "computers_accessories",
                "unit_price": 140.00,
                "qty": 18.0,
                "comp_1": 135.00,
                "product_score": 3.9,
                "customers": 80,
            },
        },
        {
            "name": "Specialty Cool Stuff Item (Inelastic)",
            "features": {
                "product_category_name": "cool_stuff",
                "unit_price": 115.00,
                "qty": 30.0,
                "comp_1": 120.00,
                "product_score": 4.5,
                "customers": 210,
            },
        },
    ]

    for sku in sample_skus:
        res = loaded_pipeline.predict_item(sku["features"], objective="profit")
        print(f"\n  --- {sku['name']} ---")
        print(f"  Category:               {sku['features']['product_category_name']}")
        print(f"  Baseline Price:         ${res['baseline_price']:.2f}")
        print(f"  Optimal Price:          ${res['optimal_price']:.2f} ({res['price_change_pct']:+.1f}%)")
        print(f"  Action / Constraint:    {res['action']} ({res['bound_hit']})")
        print(f"  Causal Elasticity:      {res['causal_elasticity']:.4f}")
        print(f"  Expected Demand Uplift: {res['demand_uplift_pct']:+.1f}% ({res['baseline_qty']} -> {res['projected_qty']} units)")
        print(f"  Expected Profit Uplift: {res['profit_uplift_pct']:+.1f}% (${res['baseline_profit']:.2f} -> ${res['projected_profit']:.2f})")
        print(f"  Expected Margin Shift:  {res['baseline_margin_pct']:.1f}% -> {res['projected_margin_pct']:.1f}%")

    # 5. Batch Catalog Price Recommendations
    rec_path = os.path.join(repo_root, "data", "pricing_recommendations_day6.csv")
    print(f"\n[Step 6] Generating Batch Price Recommendations across entire catalog...")
    recs_df = loaded_pipeline.batch_recommend(
        catalog_df=df,
        objective="profit",
        export_path=rec_path,
    )
    print(f"  -> Recommendations exported to:\n  -> {rec_path}")
    print(f"  -> Total recommendations generated: {len(recs_df)} SKUs")

    # Display Action Breakdown
    action_counts = recs_df["pricing_action"].value_counts().to_dict()
    print(f"  -> Pricing Actions Distribution: {action_counts}")

    # 6. Counterfactual Strategy Scenario Simulation
    print("\n[Step 7] Running Multi-Strategy Counterfactual Portfolio Simulation:")
    scenarios_df = loaded_pipeline.simulate_scenarios(catalog_df=df)
    print("\n" + scenarios_df.to_string(index=False))

    print("\n" + "=" * 80)
    print("      DAY 6 PIPELINE & CLI ORCHESTRATION COMPLETED SUCCESSFULLY")
    print("=" * 80)


if __name__ == "__main__":
    run_day6_demonstration()
