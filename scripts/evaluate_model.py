#!/usr/bin/env python
"""Repeatable Model Evaluation and Performance Profiler Script.

Milestone: Day 18 — Reproducibility and model performance.

Usage Examples:
    # Evaluate default model artifact against raw retail dataset:
    python scripts/evaluate_model.py

    # Evaluate specific versioned bundle with custom batch sizes:
    python scripts/evaluate_model.py --model models/v1.0.0 --batches 1,10,50,100,500

    # Export structured evaluation report and JSON results:
    python scripts/evaluate_model.py --output-report experiments/results/model_evaluation_report.md --output-json experiments/results/model_evaluation_results.json
"""

import argparse
import os
import sys

# Ensure repository root is on sys.path
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.cli import create_parser, handle_evaluate


def main():
    """Execute repeatable evaluation command."""
    parser = argparse.ArgumentParser(
        prog="evaluate_model",
        description="EconoCausal Model Evaluation & Performance Profiler",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--model",
        type=str,
        default="models/econocausal_pipeline.joblib",
        help="Path to pipeline .joblib model or versioned artifact bundle directory",
    )
    parser.add_argument(
        "--data",
        type=str,
        default="data/raw/retail_price.csv",
        help="Path to evaluation dataset CSV",
    )
    parser.add_argument(
        "--output-report",
        type=str,
        default="experiments/results/model_evaluation_report.md",
        help="Destination path for Markdown evaluation audit report",
    )
    parser.add_argument(
        "--output-json",
        type=str,
        default="experiments/results/model_evaluation_results.json",
        help="Destination path for structured JSON evaluation results",
    )
    parser.add_argument(
        "--batches",
        type=str,
        default="1,10,50,100",
        help="Comma-separated representative batch sizes for latency & memory benchmarking",
    )
    parser.add_argument(
        "--runs",
        type=int,
        default=50,
        help="Number of single-item inference iterations for latency percentile calculation",
    )
    parser.add_argument(
        "--no-memory",
        action="store_true",
        help="Disable memory profiling with tracemalloc",
    )

    args = parser.parse_args()
    return handle_evaluate(args)


if __name__ == "__main__":
    sys.exit(main())
