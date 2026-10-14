"""Unit tests for data loading, schema validation, and causal feature extraction."""

import numpy as np
import pandas as pd
import pytest

from src.data.load_data import (
    DEFAULT_CONFOUNDER_COLS,
    DEFAULT_OUTCOME_COL,
    DEFAULT_TREATMENT_COL,
    extract_causal_matrices,
    generate_synthetic_retail_data,
    summarize_data_quality,
    validate_schema,
)


@pytest.fixture
def valid_dataframe() -> pd.DataFrame:
    """Fixture providing valid retail dataset."""
    return generate_synthetic_retail_data(n_samples=100, random_seed=123)


def test_synthetic_data_schema(valid_dataframe: pd.DataFrame):
    """Test that synthetic data generator adheres to data dictionary specifications."""
    assert len(valid_dataframe) == 100
    assert "price" in valid_dataframe.columns
    assert "units_sold" in valid_dataframe.columns
    for col in DEFAULT_CONFOUNDER_COLS:
        assert col in valid_dataframe.columns

    # Verify constraints
    assert (valid_dataframe["price"] > 0).all()
    assert (valid_dataframe["units_sold"] >= 0).all()
    assert (valid_dataframe["competitor_price"] >= 0).all()
    assert set(valid_dataframe["is_promotion"].unique()).issubset({0, 1})


def test_validate_schema_valid(valid_dataframe: pd.DataFrame):
    """Test validation on valid dataset passes without errors."""
    result = validate_schema(valid_dataframe)
    assert result["is_valid"] is True
    assert len(result["errors"]) == 0
    assert result["row_count"] == 100


def test_validate_schema_empty():
    """Test validation fails gracefully on empty DataFrame."""
    empty_df = pd.DataFrame()
    result = validate_schema(empty_df)
    assert result["is_valid"] is False
    assert "empty" in result["errors"][0].lower()


def test_validate_schema_missing_columns(valid_dataframe: pd.DataFrame):
    """Test validation fails when a required causal column is missing."""
    corrupted_df = valid_dataframe.drop(columns=["price"])
    result = validate_schema(corrupted_df)
    assert result["is_valid"] is False
    assert any("price" in err for err in result["errors"])


def test_validate_schema_post_treatment_leakage(valid_dataframe: pd.DataFrame):
    """Test that adding post-treatment variables into confounders triggers validation error."""
    # revenue is post-treatment (outcome * price) and must not be a confounder
    result = validate_schema(
        valid_dataframe,
        confounder_cols=DEFAULT_CONFOUNDER_COLS + ["revenue"],
    )
    assert result["is_valid"] is False
    assert any("leakage" in err.lower() for err in result["errors"])


def test_validate_schema_invalid_price(valid_dataframe: pd.DataFrame):
    """Test validation fails when price contains non-positive values."""
    corrupted_df = valid_dataframe.copy()
    corrupted_df.loc[0, "price"] = -5.0
    result = validate_schema(corrupted_df)
    assert result["is_valid"] is False
    assert any("price <= 0" in err for err in result["errors"])


def test_validate_schema_invalid_units_sold(valid_dataframe: pd.DataFrame):
    """Test validation fails when units_sold contains negative values."""
    corrupted_df = valid_dataframe.copy()
    corrupted_df.loc[0, "units_sold"] = -10
    result = validate_schema(corrupted_df)
    assert result["is_valid"] is False
    assert any("units_sold < 0" in err for err in result["errors"])


def test_extract_causal_matrices(valid_dataframe: pd.DataFrame):
    """Test extraction of (X, T, Y) returns matching dimensions and correct separation."""
    X, T, Y = extract_causal_matrices(valid_dataframe)

    assert isinstance(X, pd.DataFrame)
    assert isinstance(T, pd.Series)
    assert isinstance(Y, pd.Series)

    assert len(X) == len(T) == len(Y) == 100
    assert list(X.columns) == DEFAULT_CONFOUNDER_COLS
    assert T.name == DEFAULT_TREATMENT_COL
    assert Y.name == DEFAULT_OUTCOME_COL

    # Ensure no outcome leakage inside X
    assert DEFAULT_OUTCOME_COL not in X.columns
    assert DEFAULT_TREATMENT_COL not in X.columns


def test_summarize_data_quality(valid_dataframe: pd.DataFrame):
    """Test data quality summary outputs expected metrics."""
    summary = summarize_data_quality(valid_dataframe)
    assert summary["total_records"] == 100
    assert "price_summary" in summary
    assert "units_sold_summary" in summary
    assert summary["price_summary"]["min"] > 0
    assert summary["units_sold_summary"]["min"] >= 0
