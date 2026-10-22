"""Command Line Interface (CLI) for EconoCausal Dynamic Pricing Pipeline.

Provides production-ready subcommands for:
- train: End-to-end model training, refutation diagnostics, and model artifact export
- recommend: Catalog price optimization and recommendation sheet generation
- predict-item: Real-time single-SKU dynamic pricing recommendation
- simulate: Counterfactual scenario comparison (Status Quo vs Causal vs Competitor)

Usage Examples:
    python -m src.cli train --data data/raw/retail_price.csv --save-model models/pipeline.joblib
    python -m src.cli recommend --model models/pipeline.joblib --data data/raw/retail_price.csv --output data/recs.csv
    python -m src.cli predict-item --model models/pipeline.joblib --category health_beauty --price 100.0 --cost 70.0
    python -m src.cli simulate --model models/pipeline.joblib --data data/raw/retail_price.csv
"""

import argparse
import json
import os
import sys
from typing import List, Optional

import pandas as pd

from src.data.load_data import load_retail_price_dataset
from src.pipeline import EconoCausalPipeline


def create_parser() -> argparse.ArgumentParser:
    """Construct CLI argument parser with subcommands."""
    parser = argparse.ArgumentParser(
        prog="econocausal",
        description="EconoCausal: Dynamic Pricing via Double Machine Learning CLI",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    subparsers = parser.add_subparsers(dest="command", help="Available subcommands")

    # 1. TRAIN SUBCOMMAND
    train_parser = subparsers.add_parser(
        "train",
        help="Train end-to-end causal pipeline, run diagnostics, and save model artifact.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    train_parser.add_argument(
        "--data",
        type=str,
        default="data/raw/retail_price.csv",
        help="Path to training dataset CSV",
    )
    train_parser.add_argument(
        "--save-model",
        type=str,
        default="models/econocausal_pipeline.joblib",
        help="Destination path for serialized pipeline artifact",
    )
    train_parser.add_argument(
        "--test-size",
        type=float,
        default=0.20,
        help="Holdout validation fraction",
    )
    train_parser.add_argument(
        "--n-splits",
        type=int,
        default=5,
        help="Number of cross-fitting folds for DML",
    )
    train_parser.add_argument(
        "--no-refutations",
        action="store_true",
        help="Skip causal refutation tests to speed up training",
    )
    train_parser.add_argument(
        "--output-report",
        type=str,
        default=None,
        help="Optional destination path for formatted text audit report",
    )

    # 2. RECOMMEND SUBCOMMAND
    rec_parser = subparsers.add_parser(
        "recommend",
        help="Run batch dynamic pricing optimization across product catalog and export sheet.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    rec_parser.add_argument(
        "--model",
        type=str,
        default="models/econocausal_pipeline.joblib",
        help="Path to trained pipeline .joblib model",
    )
    rec_parser.add_argument(
        "--data",
        type=str,
        default="data/raw/retail_price.csv",
        help="Path to catalog dataset CSV",
    )
    rec_parser.add_argument(
        "--output",
        type=str,
        default="data/pricing_recommendations.csv",
        help="Destination CSV path for recommendations",
    )
    rec_parser.add_argument(
        "--objective",
        type=str,
        choices=["profit", "revenue"],
        default="profit",
        help="Optimization objective",
    )
    rec_parser.add_argument(
        "--top",
        type=int,
        default=10,
        help="Number of top recommendation rows to preview in console",
    )

    # 3. PREDICT-ITEM SUBCOMMAND
    item_parser = subparsers.add_parser(
        "predict-item",
        help="Real-time on-demand optimal price prediction for a single product / SKU.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    item_parser.add_argument(
        "--model",
        type=str,
        default="models/econocausal_pipeline.joblib",
        help="Path to trained pipeline .joblib model",
    )
    item_parser.add_argument(
        "--category",
        type=str,
        default="health_beauty",
        help="Product category name (e.g. 'health_beauty', 'computers_accessories')",
    )
    item_parser.add_argument(
        "--price",
        type=float,
        required=True,
        help="Current baseline selling price (unit_price)",
    )
    item_parser.add_argument(
        "--cost",
        type=float,
        default=None,
        help="Unit cost (if omitted, estimated using default gross margin)",
    )
    item_parser.add_argument(
        "--comp-price",
        type=float,
        default=None,
        help="Competitor benchmark selling price",
    )
    item_parser.add_argument(
        "--qty",
        type=float,
        default=10.0,
        help="Baseline demand volume per period",
    )
    item_parser.add_argument(
        "--objective",
        type=str,
        choices=["profit", "revenue"],
        default="profit",
        help="Optimization objective",
    )
    item_parser.add_argument(
        "--format",
        type=str,
        choices=["table", "json"],
        default="table",
        help="Output display format",
    )

    # 4. SIMULATE SUBCOMMAND
    sim_parser = subparsers.add_parser(
        "simulate",
        help="Simulate counterfactual scenario comparison across pricing strategies.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    sim_parser.add_argument(
        "--model",
        type=str,
        default="models/econocausal_pipeline.joblib",
        help="Path to trained pipeline .joblib model",
    )
    sim_parser.add_argument(
        "--data",
        type=str,
        default="data/raw/retail_price.csv",
        help="Path to catalog dataset CSV",
    )
    sim_parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Optional destination path for scenario comparison CSV",
    )

    return parser


def handle_train(args: argparse.Namespace) -> int:
    """Execute model training workflow."""
    print(f"\n[EconoCausal CLI] Loading raw training data from: {args.data}")
    if not os.path.exists(args.data):
        print(f"Error: Dataset file not found at: {args.data}", file=sys.stderr)
        return 1

    df = load_retail_price_dataset(args.data)
    print(f"[EconoCausal CLI] Loaded {len(df)} records with {len(df.columns)} columns.")

    print(f"[EconoCausal CLI] Initializing EconoCausalPipeline (K={args.n_splits} folds)...")
    pipeline = EconoCausalPipeline(
        n_splits=args.n_splits,
        random_state=42,
    )

    run_refutations = not args.no_refutations
    print(f"[EconoCausal CLI] Fitting pipeline (Causal Refutations: {run_refutations})...")
    pipeline.fit(
        df=df,
        test_size=args.test_size,
        run_refutations=run_refutations,
        refutation_repeats=2,
    )

    # Serialize model
    saved_path = pipeline.save(args.save_model)
    print(f"[EconoCausal CLI] Model artifact persisted successfully to: {saved_path}")

    # Executive report
    summary_report = pipeline.generate_executive_summary()
    print("\n" + summary_report)

    if args.output_report:
        os.makedirs(os.path.dirname(os.path.abspath(args.output_report)), exist_ok=True)
        with open(args.output_report, "w", encoding="utf-8") as f:
            f.write(summary_report)
        print(f"\n[EconoCausal CLI] Audit report saved to: {args.output_report}")

    return 0


def handle_recommend(args: argparse.Namespace) -> int:
    """Execute batch catalog pricing optimization workflow."""
    print(f"\n[EconoCausal CLI] Loading model pipeline from: {args.model}")
    if not os.path.exists(args.model):
        print(f"Error: Pipeline model not found at: {args.model}", file=sys.stderr)
        return 1

    pipeline = EconoCausalPipeline.load(args.model)

    print(f"[EconoCausal CLI] Loading catalog data from: {args.data}")
    if not os.path.exists(args.data):
        print(f"Error: Catalog data file not found at: {args.data}", file=sys.stderr)
        return 1

    df = load_retail_price_dataset(args.data)
    print(f"[EconoCausal CLI] Generating price recommendations (Objective: {args.objective})...")
    recs_df = pipeline.batch_recommend(
        catalog_df=df,
        objective=args.objective,
        export_path=args.output,
    )

    print(f"[EconoCausal CLI] Recommendations successfully exported to: {args.output}")
    print(f"\n[PREVIEW: Top {args.top} Price Recommendations]")
    display_cols = [
        col for col in [
            "product_id",
            "product_category_name",
            "pricing_action",
            "baseline_price",
            "optimal_price",
            "price_change_pct",
            "elasticity",
            "profit_uplift_pct",
        ] if col in recs_df.columns
    ]
    preview_df = recs_df[display_cols].head(args.top).copy()
    if "price_change_pct" in preview_df.columns:
        preview_df["price_change_pct"] = preview_df["price_change_pct"].apply(lambda v: f"{v:+.1f}%")
    if "profit_uplift_pct" in preview_df.columns:
        preview_df["profit_uplift_pct"] = preview_df["profit_uplift_pct"].apply(lambda v: f"{v:+.1f}%")
    print(preview_df.to_string(index=False))

    return 0


def handle_predict_item(args: argparse.Namespace) -> int:
    """Execute real-time single item pricing inference."""
    if not os.path.exists(args.model):
        print(f"Error: Pipeline model not found at: {args.model}", file=sys.stderr)
        return 1

    pipeline = EconoCausalPipeline.load(args.model)

    item = {
        "unit_price": args.price,
        "qty": args.qty,
        "product_category_name": args.category,
    }
    if args.cost is not None:
        item["unit_cost"] = args.cost
    if args.comp_price is not None:
        item["comp_1"] = args.comp_price

    res = pipeline.predict_item(item=item, objective=args.objective)

    if args.format == "json":
        print(json.dumps(res, indent=2))
    else:
        print("\n" + "=" * 60)
        print("     ECONOCAUSAL REAL-TIME PRICING INFERENCE RESULT")
        print("=" * 60)
        print(f"Product Category:         {args.category}")
        print(f"Current Baseline Price:   ${res['baseline_price']:.2f}")
        print(f"Recommended Price:        ${res['optimal_price']:.2f} ({res['price_change_pct']:+.1f}%)")
        print(f"Pricing Action:           {res['action']} ({res['bound_hit']})")
        print(f"Causal Elasticity:        {res['causal_elasticity']:.4f}")
        print(f"Unit Cost:                ${res['unit_cost']:.2f}")
        print(f"Margin Shift:             {res['baseline_margin_pct']:.1f}% -> {res['projected_margin_pct']:.1f}%")
        print(f"Projected Sales Volume:   {res['baseline_qty']:.1f} -> {res['projected_qty']:.1f} ({res['demand_uplift_pct']:+.1f}%)")
        print(f"Projected Revenue Uplift: {res['revenue_uplift_pct']:+.1f}% (${res['baseline_revenue']:.2f} -> ${res['projected_revenue']:.2f})")
        print(f"Projected Profit Uplift:  {res['profit_uplift_pct']:+.1f}% (${res['baseline_profit']:.2f} -> ${res['projected_profit']:.2f})")
        print("=" * 60)

    return 0


def handle_simulate(args: argparse.Namespace) -> int:
    """Execute scenario simulation workflow."""
    if not os.path.exists(args.model):
        print(f"Error: Pipeline model not found at: {args.model}", file=sys.stderr)
        return 1

    pipeline = EconoCausalPipeline.load(args.model)

    if not os.path.exists(args.data):
        print(f"Error: Catalog data not found at: {args.data}", file=sys.stderr)
        return 1

    df = load_retail_price_dataset(args.data)
    print(f"\n[EconoCausal CLI] Evaluating pricing scenarios across {len(df)} catalog observations...")

    scenarios_df = pipeline.simulate_scenarios(catalog_df=df)

    print("\n" + "=" * 80)
    print("                PRICING STRATEGY SCENARIO COMPARISON")
    print("=" * 80)
    print(scenarios_df.to_string(index=False))
    print("=" * 80)

    if args.output:
        os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
        scenarios_df.to_csv(args.output, index=False)
        print(f"\n[EconoCausal CLI] Scenario comparison matrix exported to: {args.output}")

    return 0


def main(argv: Optional[List[str]] = None) -> int:
    """Main CLI entry point."""
    parser = create_parser()
    args = parser.parse_args(argv)

    if args.command is None:
        parser.print_help()
        return 0

    if args.command == "train":
        return handle_train(args)
    elif args.command == "recommend":
        return handle_recommend(args)
    elif args.command == "predict-item":
        return handle_predict_item(args)
    elif args.command == "simulate":
        return handle_simulate(args)
    else:
        parser.print_help()
        return 1


if __name__ == "__main__":
    sys.exit(main())
