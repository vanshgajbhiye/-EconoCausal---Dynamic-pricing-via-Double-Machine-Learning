"""Unit and integration tests for EconoCausal Command Line Interface (CLI)."""

import json
import os
import tempfile
import pytest

from src.cli import main
from src.data.load_data import generate_synthetic_retail_data
from src.pipeline import EconoCausalPipeline


@pytest.fixture
def trained_model_and_catalog():
    """Create a temporary trained model artifact and catalog CSV for CLI tests."""
    with tempfile.TemporaryDirectory() as tmp_dir:
        data_path = os.path.join(tmp_dir, "catalog.csv")
        model_path = os.path.join(tmp_dir, "test_pipeline.joblib")

        # Create synthetic catalog
        df = generate_synthetic_retail_data(n_samples=100, random_seed=42)
        df.to_csv(data_path, index=False)

        # Train and save a small pipeline
        p = EconoCausalPipeline(n_splits=3, random_state=42, categorical_confounders=["product_category_name"])
        p.fit(df, test_size=0.20, run_refutations=False)
        p.save(model_path)

        yield {
            "tmp_dir": tmp_dir,
            "data_path": data_path,
            "model_path": model_path,
        }


def test_cli_help(capsys):
    """Test that main with no args or --help prints help message and exits 0."""
    code = main([])
    assert code == 0
    captured = capsys.readouterr()
    assert "usage: econocausal" in captured.out


def test_cli_train_command(trained_model_and_catalog):
    """Test CLI train command with synthetic data."""
    ctx = trained_model_and_catalog
    new_model_path = os.path.join(ctx["tmp_dir"], "trained_via_cli.joblib")
    report_path = os.path.join(ctx["tmp_dir"], "audit_report.txt")

    argv = [
        "train",
        "--data", ctx["data_path"],
        "--save-model", new_model_path,
        "--n-splits", "3",
        "--no-refutations",
        "--output-report", report_path,
    ]
    code = main(argv)
    assert code == 0
    assert os.path.exists(new_model_path)
    assert os.path.exists(report_path)


def test_cli_recommend_command(trained_model_and_catalog):
    """Test CLI recommend command."""
    ctx = trained_model_and_catalog
    output_csv = os.path.join(ctx["tmp_dir"], "recommendations.csv")

    argv = [
        "recommend",
        "--model", ctx["model_path"],
        "--data", ctx["data_path"],
        "--output", output_csv,
        "--objective", "profit",
        "--top", "5",
    ]
    code = main(argv)
    assert code == 0
    assert os.path.exists(output_csv)


def test_cli_predict_item_table_format(trained_model_and_catalog, capsys):
    """Test CLI predict-item command with default table format."""
    ctx = trained_model_and_catalog
    argv = [
        "predict-item",
        "--model", ctx["model_path"],
        "--price", "99.99",
        "--category", "bed_bath_table",
        "--cost", "65.0",
        "--objective", "profit",
        "--format", "table",
    ]
    code = main(argv)
    assert code == 0
    captured = capsys.readouterr()
    assert "ECONOCAUSAL REAL-TIME PRICING INFERENCE RESULT" in captured.out
    assert "Recommended Price:" in captured.out


def test_cli_predict_item_json_format(trained_model_and_catalog, capsys):
    """Test CLI predict-item command with JSON format."""
    ctx = trained_model_and_catalog
    argv = [
        "predict-item",
        "--model", ctx["model_path"],
        "--price", "49.50",
        "--category", "bed_bath_table",
        "--objective", "revenue",
        "--format", "json",
    ]
    code = main(argv)
    assert code == 0
    captured = capsys.readouterr()
    data = json.loads(captured.out)
    assert data["status"] == "SUCCESS"
    assert data["baseline_price"] == 49.50
    assert "optimal_price" in data


def test_cli_simulate_command(trained_model_and_catalog):
    """Test CLI simulate command."""
    ctx = trained_model_and_catalog
    scenario_csv = os.path.join(ctx["tmp_dir"], "scenarios.csv")

    argv = [
        "simulate",
        "--model", ctx["model_path"],
        "--data", ctx["data_path"],
        "--output", scenario_csv,
    ]
    code = main(argv)
    assert code == 0
    assert os.path.exists(scenario_csv)


def test_cli_invalid_args_handling(capsys):
    """Ensure invalid or missing model path returns non-zero error code."""
    argv = [
        "recommend",
        "--model", "non_existent_model_file.joblib",
        "--data", "non_existent_data.csv",
    ]
    code = main(argv)
    assert code == 1
    captured = capsys.readouterr()
    assert "Error:" in captured.err
