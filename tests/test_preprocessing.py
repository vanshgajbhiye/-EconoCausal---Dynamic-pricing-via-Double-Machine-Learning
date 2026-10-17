"""Unit tests for data loading, schema validation, and causal feature extraction.

Tests against the official Kaggle Retail Price Optimization dataset.
"""

import pandas as pd
import pytest

from src.data.load_data import (
    DEFAULT_CONFOUNDER_COLS,
    DEFAULT_OUTCOME_COL,
    DEFAULT_TREATMENT_COL,
    extract_causal_matrices,
    generate_synthetic_retail_data,
    load_retail_price_dataset,
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
    assert "unit_price" in valid_dataframe.columns
    assert "qty" in valid_dataframe.columns
    for col in DEFAULT_CONFOUNDER_COLS:
        assert col in valid_dataframe.columns

    # Verify constraints
    assert (valid_dataframe["unit_price"] > 0).all()
    assert (valid_dataframe["qty"] >= 0).all()
    assert (valid_dataframe["comp_1"] > 0).all()


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
    corrupted_df = valid_dataframe.drop(columns=["unit_price"])
    result = validate_schema(corrupted_df)
    assert result["is_valid"] is False
    assert any("unit_price" in err for err in result["errors"])


def test_validate_schema_post_treatment_leakage(valid_dataframe: pd.DataFrame):
    """Test that adding post-treatment variables into confounders triggers validation error."""
    # total_price is post-treatment (revenue = qty * unit_price) and must not be a confounder
    result = validate_schema(
        valid_dataframe,
        confounder_cols=DEFAULT_CONFOUNDER_COLS + ["total_price"],
    )
    assert result["is_valid"] is False
    assert any("leakage" in err.lower() for err in result["errors"])


def test_validate_schema_invalid_price(valid_dataframe: pd.DataFrame):
    """Test validation fails when unit_price contains non-positive values."""
    corrupted_df = valid_dataframe.copy()
    corrupted_df.loc[0, "unit_price"] = -5.0
    result = validate_schema(corrupted_df)
    assert result["is_valid"] is False
    assert any("price <= 0" in err for err in result["errors"])


def test_validate_schema_invalid_qty(valid_dataframe: pd.DataFrame):
    """Test validation fails when qty contains negative values."""
    corrupted_df = valid_dataframe.copy()
    corrupted_df.loc[0, "qty"] = -10
    result = validate_schema(corrupted_df)
    assert result["is_valid"] is False
    assert any("qty < 0" in err for err in result["errors"])


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
    assert "total_price" not in X.columns


def test_summarize_data_quality(valid_dataframe: pd.DataFrame):
    """Test data quality summary outputs expected metrics."""
    summary = summarize_data_quality(valid_dataframe)
    assert summary["total_records"] == 100
    assert "price_summary" in summary
    assert "qty_summary" in summary
    assert summary["price_summary"]["min"] > 0
    assert summary["qty_summary"]["min"] >= 0


def test_load_retail_price_dataset():
    """Test loading and validation of official Kaggle retail_price.csv."""
    df = load_retail_price_dataset("data/raw/retail_price.csv")

    assert len(df) == 676
    assert "unit_price" in df.columns
    assert "qty" in df.columns
    assert "comp_1" in df.columns
    assert "s" in df.columns
    assert "customers" in df.columns
    assert "freight_price" in df.columns

    # Verify zero nulls
    assert df.isnull().sum().sum() == 0

    # Verify constraints
    assert (df["unit_price"] > 0).all()
    assert (df["qty"] >= 0).all()
    assert (df["comp_1"] > 0).all()


def test_extract_causal_matrices_real_retail_price():
    """Test extracting (X, T, Y) from official Kaggle retail_price.csv."""
    df = load_retail_price_dataset("data/raw/retail_price.csv")
    X, T, Y = extract_causal_matrices(df)

    assert len(X) == len(T) == len(Y) == 676
    assert list(X.columns) == DEFAULT_CONFOUNDER_COLS
    assert T.name == "unit_price"
    assert Y.name == "qty"

    # Strict anti-leakage verification
    assert "qty" not in X.columns
    assert "total_price" not in X.columns
    assert "revenue" not in X.columns
