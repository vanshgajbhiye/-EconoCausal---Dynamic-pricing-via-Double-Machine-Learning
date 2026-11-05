"""Unit and Integration Tests for Model Inference Latency, Memory Profiling, and Repeatable Evaluation.

Milestone: Day 18 — Reproducibility and model performance.
"""

import json
import os
import tempfile
import numpy as np
import pandas as pd
import pytest

from src.data.load_data import generate_synthetic_retail_data
from src.models.benchmark import (
    ModelBenchmarkSummary,
    run_inference_benchmarks,
)
from src.models.evaluator import EvaluationAuditReport, ModelEvaluator
from src.pipeline import EconoCausalPipeline


@pytest.fixture
def fitted_pipeline():
    """Fitted pipeline on synthetic data for inference benchmarking."""
    df = generate_synthetic_retail_data(n_samples=150, random_seed=42)
    pipeline = EconoCausalPipeline(n_splits=3, random_state=42, model_version="1.0.0")
    pipeline.fit(df, test_size=0.20, run_refutations=False)
    return pipeline, df


def test_single_item_latency_benchmarking(fitted_pipeline):
    """Verify single-SKU real-time inference latency distribution metrics."""
    pipeline, df = fitted_pipeline

    summary = run_inference_benchmarks(
        pipeline=pipeline,
        df=df,
        batch_sizes=[1, 5],
        n_single_runs=30,
        measure_memory=True,
    )

    assert isinstance(summary, ModelBenchmarkSummary)
    single = summary.single_item

    assert single.iterations == 30
    assert single.mean_ms > 0
    assert single.p50_ms > 0
    assert single.p90_ms >= single.p50_ms
    assert single.p99_ms >= single.p90_ms
    assert single.min_ms <= single.max_ms
    assert single.throughput_items_per_sec > 0
    assert single.memory_peak_mb >= 0.0


def test_representative_batch_latency_and_memory(fitted_pipeline):
    """Verify inference latency and memory tracking on representative batches."""
    pipeline, df = fitted_pipeline

    batch_sizes = [1, 10, 50]
    summary = pipeline.benchmark_inference(
        df=df,
        batch_sizes=batch_sizes,
        n_single_runs=20,
        measure_memory=True,
    )

    assert len(summary.batches) >= len(batch_sizes)

    # Check each batch entry
    sizes_tested = [b.batch_size for b in summary.batches]
    for b_size in batch_sizes:
        assert b_size in sizes_tested

    for b_metric in summary.batches:
        assert b_metric.mean_batch_latency_ms > 0
        assert b_metric.mean_per_item_latency_ms > 0
        assert b_metric.throughput_items_per_sec > 0
        assert b_metric.memory_peak_mb >= 0.0


def test_inference_reproducibility_audit(fitted_pipeline):
    """Verify that repeatedly running inference produces identical predictions."""
    pipeline, df = fitted_pipeline

    summary = run_inference_benchmarks(
        pipeline=pipeline,
        df=df,
        batch_sizes=[5],
        n_single_runs=10,
        measure_memory=False,
    )

    repro = summary.reproducibility
    assert repro.is_reproducible is True
    assert repro.max_elasticity_abs_diff < 1e-7
    assert repro.max_optimal_price_abs_diff < 1e-7
    assert repro.identical_actions is True


def test_model_evaluator_engine(fitted_pipeline):
    """Verify the repeatable evaluation engine produces structured reports."""
    pipeline, df = fitted_pipeline

    evaluator = ModelEvaluator(pipeline=pipeline, dataset_path="synthetic_test.csv")
    report = evaluator.evaluate(
        df=df,
        batch_sizes=[1, 10],
        n_single_runs=15,
        measure_memory=True,
        run_refutations=False,
    )

    assert isinstance(report, EvaluationAuditReport)
    assert report.overall_status in ["PASSED", "WARNING"]
    assert report.dataset_records == len(df)
    assert "baseline_ate" in report.ate_metrics
    assert "mean_elasticity" in report.cate_metrics
    assert "profit_uplift_pct" in report.simulation_metrics
    assert "single_item" in report.performance_metrics
    assert "is_reproducible" in report.reproducibility_metrics

    # Verify JSON and Markdown export
    with tempfile.TemporaryDirectory() as tmp_dir:
        json_path = os.path.join(tmp_dir, "eval.json")
        md_path = os.path.join(tmp_dir, "eval.md")

        report.save_json(json_path)
        report.save_markdown(md_path)

        assert os.path.exists(json_path)
        assert os.path.exists(md_path)

        with open(json_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        assert data["overall_status"] == report.overall_status

        with open(md_path, "r", encoding="utf-8") as f:
            md_text = f.read()
        assert "EconoCausal Model Evaluation & Reproducibility Audit" in md_text


def test_cli_repeatable_evaluation_command(fitted_pipeline):
    """Verify the CLI evaluate subcommand executes end-to-end and exits cleanly."""
    pipeline, df = fitted_pipeline

    with tempfile.TemporaryDirectory() as tmp_dir:
        model_path = os.path.join(tmp_dir, "model.joblib")
        data_path = os.path.join(tmp_dir, "data.csv")
        report_path = os.path.join(tmp_dir, "report.md")
        json_path = os.path.join(tmp_dir, "results.json")

        pipeline.save(model_path)
        df.to_csv(data_path, index=False)

        from src.cli import main

        exit_code = main([
            "evaluate",
            "--model", model_path,
            "--data", data_path,
            "--output-report", report_path,
            "--output-json", json_path,
            "--batches", "1,5",
            "--runs", "10",
        ])

        assert exit_code == 0
        assert os.path.exists(report_path)
        assert os.path.exists(json_path)
