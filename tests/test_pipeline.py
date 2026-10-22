"""Unit and integration tests for EconoCausalPipeline."""

import os
import tempfile
import numpy as np
import pandas as pd
import pytest

from src.data.load_data import generate_synthetic_retail_data, load_retail_price_dataset
from src.pipeline import EconoCausalPipeline


@pytest.fixture
def synthetic_data():
    """Generate synthetic retail dataset for fast unit testing."""
    return generate_synthetic_retail_data(n_samples=150, random_seed=42)


@pytest.fixture
def fitted_synthetic_pipeline(synthetic_data):
    """Provide a fitted pipeline instance on synthetic data."""
    pipeline = EconoCausalPipeline(
        n_splits=3,
        random_state=42,
        categorical_confounders=["product_category_name"],
    )
    pipeline.fit(synthetic_data, test_size=0.25, run_refutations=False)
    return pipeline


def test_pipeline_initialization():
    """Verify default parameters of EconoCausalPipeline."""
    p = EconoCausalPipeline()
    assert p.treatment_col == "unit_price"
    assert p.outcome_col == "qty"
    assert p.n_splits == 5
    assert p.is_fitted_ is False
    assert p.baseline_ate_ is None


def test_pipeline_unfitted_errors():
    """Ensure unfitted pipeline raises informative errors when calling inference methods."""
    p = EconoCausalPipeline()
    with pytest.raises(ValueError, match="not fitted yet"):
        p.predict_elasticity({"unit_price": 50.0})

    with pytest.raises(ValueError, match="not fitted yet"):
        p.predict_item({"unit_price": 50.0})

    with pytest.raises(ValueError, match="not fitted yet"):
        p.batch_recommend(pd.DataFrame({"unit_price": [50.0]}))

    with pytest.raises(ValueError, match="Cannot save an unfitted pipeline"):
        p.save("tmp_model.joblib")


def test_pipeline_fit_synthetic(synthetic_data):
    """Test full pipeline fit workflow on synthetic dataset."""
    pipeline = EconoCausalPipeline(
        n_splits=3,
        random_state=42,
        categorical_confounders=["product_category_name"],
    )
    result = pipeline.fit(synthetic_data, test_size=0.20, run_refutations=False)

    assert result is pipeline
    assert pipeline.is_fitted_ is True
    assert pipeline.baseline_ate_ is not None
    assert isinstance(pipeline.baseline_ate_, float)
    assert pipeline.baseline_ate_se_ is not None
    assert pipeline.baseline_ate_se_ > 0
    assert "trained_at" in pipeline.metadata_
    assert pipeline.metadata_["n_samples_total"] == len(synthetic_data)


def test_pipeline_predict_item_profit_mode(fitted_synthetic_pipeline):
    """Verify single-SKU real-time inference under profit maximization objective."""
    p = fitted_synthetic_pipeline

    item = {
        "unit_price": 100.0,
        "qty": 20.0,
        "product_category_name": "bed_bath_table",
        "unit_cost": 65.0,
        "comp_1": 105.0,
    }
    res = p.predict_item(item, objective="profit")

    assert res["status"] == "SUCCESS"
    assert res["baseline_price"] == 100.0
    assert res["unit_cost"] == 65.0
    assert res["optimal_price"] > 0
    assert res["action"] in ["MARKUP", "DISCOUNT", "HOLD"]
    assert isinstance(res["causal_elasticity"], float)
    assert res["projected_qty"] > 0
    assert "profit_uplift_pct" in res
    assert res["objective"] == "profit"


def test_pipeline_predict_item_revenue_mode(fitted_synthetic_pipeline):
    """Verify single-SKU real-time inference under revenue maximization objective."""
    p = fitted_synthetic_pipeline

    item = {
        "unit_price": 80.0,
        "qty": 15.0,
        "product_category_name": "bed_bath_table",
    }
    res = p.predict_item(item, objective="revenue")

    assert res["status"] == "SUCCESS"
    assert res["baseline_price"] == 80.0
    assert res["objective"] == "revenue"
    assert res["optimal_price"] > 0


def test_pipeline_predict_item_missing_features_imputation(fitted_synthetic_pipeline):
    """Ensure pipeline handles missing non-critical features via stored training medians/modes."""
    p = fitted_synthetic_pipeline

    # Minimal item with only unit_price
    item = {
        "unit_price": 50.0,
    }
    res = p.predict_item(item)

    assert res["status"] == "SUCCESS"
    assert res["baseline_price"] == 50.0
    assert res["optimal_price"] > 0


def test_pipeline_batch_recommend_and_export(fitted_synthetic_pipeline, synthetic_data):
    """Verify batch catalog recommendations generation and CSV/JSON export."""
    p = fitted_synthetic_pipeline

    with tempfile.TemporaryDirectory() as tmp_dir:
        csv_path = os.path.join(tmp_dir, "recs.csv")
        json_path = os.path.join(tmp_dir, "recs.json")

        recs_df = p.batch_recommend(synthetic_data.head(20), objective="profit", export_path=csv_path)

        assert len(recs_df) == 20
        assert "pricing_action" in recs_df.columns
        assert "baseline_price" in recs_df.columns
        assert "optimal_price" in recs_df.columns
        assert "profit_uplift_pct" in recs_df.columns
        assert os.path.exists(csv_path)

        # Verify JSON export
        p.batch_recommend(synthetic_data.head(10), objective="revenue", export_path=json_path)
        assert os.path.exists(json_path)


def test_pipeline_simulate_scenarios(fitted_synthetic_pipeline, synthetic_data):
    """Verify multi-strategy scenario simulation matrix generation."""
    p = fitted_synthetic_pipeline
    scenarios_df = p.simulate_scenarios(synthetic_data.head(30), comp_col=None)

    assert isinstance(scenarios_df, pd.DataFrame)
    assert len(scenarios_df) >= 5
    assert "Strategy" in scenarios_df.columns
    assert "Total Revenue ($)" in scenarios_df.columns
    assert "Total Profit ($)" in scenarios_df.columns


def test_pipeline_save_and_load(fitted_synthetic_pipeline, synthetic_data):
    """Verify joblib serialization and deserialization of the trained pipeline."""
    p = fitted_synthetic_pipeline

    with tempfile.TemporaryDirectory() as tmp_dir:
        model_path = os.path.join(tmp_dir, "models", "pipeline.joblib")
        saved_path = p.save(model_path)

        assert os.path.exists(saved_path)

        # Load back
        loaded_p = EconoCausalPipeline.load(saved_path)

        assert loaded_p.is_fitted_ is True
        assert loaded_p.baseline_ate_ == p.baseline_ate_
        assert loaded_p.training_feature_names_ == p.training_feature_names_

        # Verify predictions match between original and loaded model
        test_item = {"unit_price": 75.0, "product_category_name": "bed_bath_table"}
        pred_orig = p.predict_item(test_item)
        pred_loaded = loaded_p.predict_item(test_item)

        assert pred_orig["optimal_price"] == pred_loaded["optimal_price"]
        assert pred_orig["causal_elasticity"] == pred_loaded["causal_elasticity"]


def test_pipeline_executive_summary(fitted_synthetic_pipeline):
    """Verify generation of readable ASCII executive summary report."""
    p = fitted_synthetic_pipeline
    summary = p.generate_executive_summary()

    assert "ECONOCAUSAL END-TO-END PIPELINE EXECUTIVE AUDIT" in summary
    assert "Baseline ATE Elasticity" in summary
    assert "DYNAMIC PRICING OPTIMIZER GUARDRAILS" in summary


def test_pipeline_real_retail_price_dataset():
    """Integration test: Verify pipeline on official Kaggle Retail Price Optimization dataset."""
    data_path = "data/raw/retail_price.csv"
    if not os.path.exists(data_path):
        pytest.skip(f"Data file not found at {data_path}")

    df = load_retail_price_dataset(data_path)
    pipeline = EconoCausalPipeline(
        n_splits=3,
        random_state=42,
        categorical_confounders=["product_category_name"],
    )

    # Train with refutations
    pipeline.fit(df, test_size=0.20, run_refutations=True, refutation_repeats=2)

    assert pipeline.is_fitted_ is True
    # Ground truth baseline elasticity is negative (economic law of demand)
    assert pipeline.baseline_ate_ < 0
    assert pipeline.refutation_pass_rate_ >= 66.0

    # Test single-item prediction on real retail category
    item = {
        "product_category_name": "health_beauty",
        "unit_price": 107.12,
        "qty": 20.0,
        "comp_1": 105.0,
        "product_score": 4.1,
    }
    rec = pipeline.predict_item(item, objective="profit")
    assert rec["status"] == "SUCCESS"
    assert rec["optimal_price"] > 0
    assert rec["action"] in ["MARKUP", "DISCOUNT", "HOLD"]
