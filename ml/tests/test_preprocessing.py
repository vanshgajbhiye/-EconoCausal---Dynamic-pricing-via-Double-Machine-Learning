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


def test_causal_preprocessor():
    """Test CausalPreprocessor fits and transforms numerical and categorical confounders."""
    from src.data.preprocess import CausalPreprocessor

    df = pd.DataFrame(
        {
            "comp_1": [10.0, 20.0, 30.0, 40.0],
            "freight_price": [5.0, 15.0, 25.0, 35.0],
            "category": ["A", "B", "A", "B"],
        }
    )

    preprocessor = CausalPreprocessor(
        numerical_cols=["comp_1", "freight_price"],
        categorical_cols=["category"],
    )

    # Must raise error if transformed before fit
    with pytest.raises(ValueError, match="not fitted"):
        preprocessor.transform(df)

    transformed = preprocessor.fit_transform(df)

    assert isinstance(transformed, pd.DataFrame)
    assert len(transformed) == 4
    # Check that scaled numerical columns have mean ~ 0 and std ~ 1
    assert abs(transformed["comp_1"].mean()) < 1e-6
    assert abs(transformed["comp_1"].std(ddof=0) - 1.0) < 1e-6
    # Check categorical feature names
    assert any("category" in col for col in transformed.columns)
    assert list(transformed.columns) == preprocessor.feature_names_out_


def test_transform_treatment_and_outcome():
    """Test log transformations on treatment and outcome series."""
    import numpy as np
    from src.data.preprocess import transform_treatment_and_outcome

    T = pd.Series([10.0, 20.0, 50.0], name="unit_price")
    Y = pd.Series([5, 10, 25], name="qty")

    T_log, Y_log = transform_treatment_and_outcome(
        T, Y, log_transform_treatment=True, log_transform_outcome=True
    )

    np.testing.assert_allclose(T_log.values, np.log([10.0, 20.0, 50.0]))
    np.testing.assert_allclose(Y_log.values, np.log([5, 10, 25]))
    assert T_log.name == "log_unit_price"
    assert Y_log.name == "log_qty"

    # Test error on non-positive treatment
    T_invalid = pd.Series([-5.0, 10.0], name="unit_price")
    with pytest.raises(ValueError, match="non-positive"):
        transform_treatment_and_outcome(T_invalid, Y, log_transform_treatment=True)


def test_split_causal_data_real_dataset():
    """Test split_causal_data on real Kaggle retail_price dataset with anti-leakage check."""
    from src.data.preprocess import split_causal_data

    df = load_retail_price_dataset("data/raw/retail_price.csv")
    splits = split_causal_data(
        df,
        test_size=0.2,
        random_state=42,
        categorical_confounders=["product_category_name"],
    )

    X_train = splits["X_train"]
    X_test = splits["X_test"]
    T_train = splits["T_train"]
    T_test = splits["T_test"]
    Y_train = splits["Y_train"]
    Y_test = splits["Y_test"]

    # 676 rows total -> 80% train (540), 20% test (136)
    assert len(X_train) == len(T_train) == len(Y_train) == 540
    assert len(X_test) == len(T_test) == len(Y_test) == 136

    # Test that preprocessor feature names match columns
    assert list(X_train.columns) == splits["feature_names"]
    assert list(X_test.columns) == splits["feature_names"]

    # Strict anti-leakage check: test set must not be standardized with its own mean
    # X_train mean is 0, but X_test mean is slightly different from 0
    assert abs(X_train["comp_1"].mean()) < 1e-6
    assert abs(X_test["comp_1"].mean()) != 0.0


def test_get_cross_fitting_folds():
    """Test K-fold cross-fitting fold partitioner for DML."""
    import numpy as np
    from src.data.preprocess import get_cross_fitting_folds

    n_samples = 676
    n_folds = 5
    folds = get_cross_fitting_folds(n_samples=n_samples, n_folds=n_folds, random_state=42)

    assert len(folds) == n_folds

    all_test_indices = []
    for train_idx, test_idx in folds:
        # Check no overlap between train and test in each fold
        assert len(np.intersect1d(train_idx, test_idx)) == 0
        assert len(train_idx) + len(test_idx) == n_samples
        all_test_indices.extend(test_idx)

    # Check complete partition (every sample tested exactly once)
    assert len(all_test_indices) == n_samples
    assert len(np.unique(all_test_indices)) == n_samples

