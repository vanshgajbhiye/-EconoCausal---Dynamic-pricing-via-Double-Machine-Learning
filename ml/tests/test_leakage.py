"""Comprehensive unit and integration tests for causal feature engineering, confounder audit,
anti-leakage guardrails, missing-data handling, and schema consistency (Day 13).

Validates:
1. Confounder audit: separating pre-treatment confounders from treatment/post-treatment variables
2. Direct treatment leakage prevention (unit_price, price, log_unit_price)
3. Direct outcome target leakage prevention (qty, units_sold)
4. Post-treatment collider and mediator leakage prevention (total_price, revenue, profit, current discount)
5. Missing-data imputation strictly learned on training partition (anti-leakage guarantee)
6. Categorical encoding handling unseen categories without crashing
7. Schema consistency (column ordering, extra columns discarded, missing columns imputed)
8. Domain-engineered causal features (competitor price gaps, cyclical seasonality, discount history)
9. End-to-end split_causal_data integration with engineered features on Kaggle retail_price.csv
"""

import numpy as np
import pandas as pd
import pytest

from src.data.load_data import (
    DEFAULT_CONFOUNDER_COLS,
    generate_synthetic_retail_data,
    load_retail_price_dataset,
)
from src.data.preprocess import (
    CONFOUNDER_AUDIT_CATALOG,
    CausalLeakageError,
    CausalPreprocessor,
    audit_confounders,
    engineer_causal_features,
    split_causal_data,
)


@pytest.fixture
def synthetic_data() -> pd.DataFrame:
    """Generate reproducible synthetic retail transactions matching schema."""
    return generate_synthetic_retail_data(n_samples=200, random_seed=42)


@pytest.fixture
def real_retail_data() -> pd.DataFrame:
    """Load the official Kaggle retail_price.csv dataset."""
    return load_retail_price_dataset("data/raw/retail_price.csv")


# ==============================================================================
# 1. CONFOUNDER AUDIT & PRE-TREATMENT SEPARATION TESTS
# ==============================================================================


def test_audit_confounders_baseline_is_clean():
    """Verify that all 13 canonical baseline confounders pass the causal audit cleanly."""
    audit = audit_confounders(confounder_cols=DEFAULT_CONFOUNDER_COLS, raise_on_leakage=False)

    assert audit["is_clean"] is True
    assert len(audit["confounders"]) == 13
    assert len(audit["treatment_leakage"]) == 0
    assert len(audit["outcome_leakage"]) == 0
    assert len(audit["post_treatment_leakage"]) == 0

    for col in DEFAULT_CONFOUNDER_COLS:
        assert col in audit["audit_details"]
        assert audit["audit_details"][col]["status"] == "Pre-treatment"
        assert len(audit["audit_details"][col]["description"]) > 0


def test_confounder_audit_catalog_coverage():
    """Verify that CONFOUNDER_AUDIT_CATALOG covers every column in DEFAULT_CONFOUNDER_COLS."""
    for col in DEFAULT_CONFOUNDER_COLS:
        assert col in CONFOUNDER_AUDIT_CATALOG
        entry = CONFOUNDER_AUDIT_CATALOG[col]
        assert "role" in entry
        assert "category" in entry
        assert entry["status"] == "Pre-treatment"
        assert len(entry["description"]) > 0


# ==============================================================================
# 2. ANTI-LEAKAGE DETECTION & EXCEPTION TESTS
# ==============================================================================


def test_treatment_leakage_detected(synthetic_data: pd.DataFrame):
    """Verify that including treatment (unit_price) in confounders raises CausalLeakageError."""
    leaked_cols = list(DEFAULT_CONFOUNDER_COLS) + ["unit_price"]

    with pytest.raises(CausalLeakageError, match="Treatment leakage detected"):
        audit_confounders(confounder_cols=leaked_cols, raise_on_leakage=True)

    with pytest.raises(CausalLeakageError, match="Treatment leakage detected"):
        preprocessor = CausalPreprocessor()
        preprocessor.fit(synthetic_data[leaked_cols])


def test_outcome_target_leakage_detected(synthetic_data: pd.DataFrame):
    """Verify that including outcome (qty) in confounders raises CausalLeakageError."""
    leaked_cols = list(DEFAULT_CONFOUNDER_COLS) + ["qty"]

    with pytest.raises(CausalLeakageError, match="Outcome target leakage detected"):
        audit_confounders(confounder_cols=leaked_cols, raise_on_leakage=True)

    with pytest.raises(CausalLeakageError, match="Outcome target leakage detected"):
        preprocessor = CausalPreprocessor()
        preprocessor.fit(synthetic_data[leaked_cols])


def test_post_treatment_collider_leakage_detected(synthetic_data: pd.DataFrame):
    """Verify that including colliders (total_price, revenue, profit) raises CausalLeakageError."""
    for bad_col in ["total_price", "revenue", "profit", "conversion_rate"]:
        leaked_cols = list(DEFAULT_CONFOUNDER_COLS) + [bad_col]
        synthetic_data_copy = synthetic_data.copy()
        synthetic_data_copy[bad_col] = 100.0

        with pytest.raises(CausalLeakageError, match="Post-treatment collider/mediator leakage"):
            audit_confounders(confounder_cols=leaked_cols, raise_on_leakage=True)

        with pytest.raises(CausalLeakageError, match="Post-treatment collider/mediator leakage"):
            preprocessor = CausalPreprocessor()
            preprocessor.fit(synthetic_data_copy[leaked_cols])


def test_derived_treatment_discount_leakage():
    """Verify that including current-period derived discount (derived from unit_price) triggers leakage."""
    bad_cols = ["comp_1", "freight_price", "discount_current"]
    with pytest.raises(CausalLeakageError, match="Treatment leakage detected"):
        audit_confounders(confounder_cols=bad_cols, raise_on_leakage=True)


def test_empirical_treatment_collinearity_leakage(synthetic_data: pd.DataFrame):
    """Verify that an adversarial feature collinear with treatment is detected via empirical audit."""
    df_adv = synthetic_data.copy()
    # Create an obfuscated feature that is perfectly collinear with unit_price
    df_adv["custom_feature_x"] = df_adv["unit_price"] * 1.0000000001

    with pytest.raises(CausalLeakageError, match="Treatment leakage detected"):
        audit_confounders(
            df=df_adv,
            confounder_cols=["comp_1", "custom_feature_x"],
            treatment_col="unit_price",
            raise_on_leakage=True,
        )


# ==============================================================================
# 3. MISSING-DATA HANDLING & ZERO-LEAKAGE IMPUTATION TESTS
# ==============================================================================


def test_numerical_missing_data_imputation():
    """Test that CausalPreprocessor imputes missing numerical values using training median."""
    train_df = pd.DataFrame(
        {
            "comp_1": [10.0, 20.0, 30.0, 40.0, 50.0],  # Median = 30.0
            "freight_price": [5.0, 15.0, 25.0, 35.0, 45.0],  # Median = 25.0
        }
    )

    test_df = pd.DataFrame(
        {
            "comp_1": [np.nan, 20.0],
            "freight_price": [15.0, np.nan],
        }
    )

    preprocessor = CausalPreprocessor(numerical_cols=["comp_1", "freight_price"], impute_missing=True)
    preprocessor.fit(train_df)

    transformed_test = preprocessor.transform(test_df)
    assert isinstance(transformed_test, pd.DataFrame)
    assert transformed_test.isnull().sum().sum() == 0
    assert len(transformed_test) == 2


def test_strict_anti_leakage_imputation_statistics():
    """Verify that test set imputation strictly uses training median, preventing information leakage."""
    # Training set with median 10.0
    train_df = pd.DataFrame({"comp_1": [8.0, 10.0, 12.0]})
    # Test set with drastically different values (median 1000.0) and a missing value
    test_df = pd.DataFrame({"comp_1": [np.nan, 1000.0, 1000.0]})

    preprocessor = CausalPreprocessor(numerical_cols=["comp_1"], impute_missing=True)
    preprocessor.fit(train_df)

    # Imputer statistic should be exactly the training median (10.0), not influenced by test set
    assert preprocessor.numerical_imputer.statistics_[0] == 10.0

    # Test that preprocessor transforms test_df without updating training statistics
    transformed_test = preprocessor.transform(test_df)
    assert not transformed_test.isnull().values.any()
    # The NaN row was replaced with 10.0, which was standardized using train mean/std (mean 10, std ~1.63) -> 0.0
    assert abs(transformed_test.iloc[0, 0]) < 1e-6


def test_categorical_missing_data_imputation():
    """Test that CausalPreprocessor gracefully imputes missing categorical confounders."""
    train_df = pd.DataFrame(
        {
            "category": ["bed_bath_table", "bed_bath_table", "health_beauty"],
            "comp_1": [10.0, 20.0, 30.0],
        }
    )
    test_df = pd.DataFrame(
        {
            "category": [None, "health_beauty"],
            "comp_1": [15.0, 25.0],
        }
    )

    preprocessor = CausalPreprocessor(
        numerical_cols=["comp_1"],
        categorical_cols=["category"],
        impute_missing=True,
    )
    preprocessor.fit(train_df)
    transformed_test = preprocessor.transform(test_df)

    assert not transformed_test.isnull().values.any()
    assert len(transformed_test) == 2


# ==============================================================================
# 4. CATEGORICAL ENCODING & UNSEEN CATEGORY TESTS
# ==============================================================================


def test_unseen_category_handling():
    """Verify that OneHotEncoder handles unseen categories without raising an error."""
    train_df = pd.DataFrame(
        {
            "category": ["bed_bath_table", "health_beauty", "bed_bath_table"],
            "comp_1": [10.0, 20.0, 30.0],
        }
    )
    # Test dataset introduces brand new unseen categories
    test_df = pd.DataFrame(
        {
            "category": ["unseen_category_electronics", "health_beauty"],
            "comp_1": [15.0, 25.0],
        }
    )

    preprocessor = CausalPreprocessor(
        numerical_cols=["comp_1"],
        categorical_cols=["category"],
    )
    preprocessor.fit(train_df)
    transformed_test = preprocessor.transform(test_df)

    assert not transformed_test.isnull().values.any()
    assert len(transformed_test) == 2

    # For the unseen category row, all category one-hot columns should be 0.0
    cat_cols = [c for c in transformed_test.columns if "category" in c]
    assert len(cat_cols) == 2  # 'bed_bath_table' and 'health_beauty'
    assert (transformed_test.loc[0, cat_cols] == 0.0).all()
    # For the known category row, exactly one indicator is 1.0
    assert transformed_test.loc[1, cat_cols].sum() == 1.0


# ==============================================================================
# 5. SCHEMA CONSISTENCY & RESILIENCE TESTS
# ==============================================================================


def test_schema_consistency_column_order():
    """Verify that preprocessor produces identical column output regardless of input column order."""
    train_df = pd.DataFrame(
        {
            "comp_1": [10.0, 20.0, 30.0],
            "freight_price": [5.0, 15.0, 25.0],
            "customers": [100, 200, 300],
        }
    )
    # Reverse the column order
    test_df = pd.DataFrame(
        {
            "customers": [150, 250],
            "comp_1": [15.0, 25.0],
            "freight_price": [10.0, 20.0],
        }
    )

    preprocessor = CausalPreprocessor(numerical_cols=["comp_1", "freight_price", "customers"])
    preprocessor.fit(train_df)

    transformed_train = preprocessor.transform(train_df)
    transformed_test = preprocessor.transform(test_df)

    # Column ordering must be identical
    assert list(transformed_train.columns) == list(transformed_test.columns)
    assert list(transformed_test.columns) == preprocessor.feature_names_out_


def test_schema_consistency_extra_columns_ignored():
    """Verify that extra unexpected columns in transform input are safely ignored."""
    train_df = pd.DataFrame({"comp_1": [10.0, 20.0], "freight_price": [5.0, 15.0]})
    test_df = pd.DataFrame(
        {
            "comp_1": [12.0, 22.0],
            "freight_price": [7.0, 17.0],
            "extra_irrelevant_col": [999, 888],
            "another_random_string": ["foo", "bar"],
        }
    )

    preprocessor = CausalPreprocessor(numerical_cols=["comp_1", "freight_price"])
    preprocessor.fit(train_df)

    transformed_test = preprocessor.transform(test_df)
    assert list(transformed_test.columns) == ["comp_1", "freight_price"]
    assert "extra_irrelevant_col" not in transformed_test.columns


def test_schema_consistency_missing_columns_imputed():
    """Verify that missing expected columns in inference DataFrame are imputed with training defaults."""
    train_df = pd.DataFrame({"comp_1": [10.0, 20.0, 30.0], "freight_price": [5.0, 15.0, 25.0]})
    # Input missing freight_price
    test_df = pd.DataFrame({"comp_1": [15.0, 25.0]})

    preprocessor = CausalPreprocessor(
        numerical_cols=["comp_1", "freight_price"],
        allow_missing_cols=True,
    )
    preprocessor.fit(train_df)

    transformed_test = preprocessor.transform(test_df)
    assert list(transformed_test.columns) == ["comp_1", "freight_price"]
    assert not transformed_test.isnull().values.any()
    # Missing freight_price was filled with training median (15.0) -> scaled value ~ 0.0
    assert abs(transformed_test["freight_price"].iloc[0]) < 1e-6


def test_dataframe_index_preservation():
    """Verify that transform preserves non-standard custom DataFrame indices."""
    train_df = pd.DataFrame({"comp_1": [10.0, 20.0]}, index=["sku_A", "sku_B"])
    test_df = pd.DataFrame({"comp_1": [15.0, 25.0, 35.0]}, index=[101, 102, 103])

    preprocessor = CausalPreprocessor(numerical_cols=["comp_1"])
    preprocessor.fit(train_df)

    transformed_test = preprocessor.transform(test_df)
    assert list(transformed_test.index) == [101, 102, 103]


# ==============================================================================
# 6. CAUSAL FEATURE ENGINEERING TESTS
# ==============================================================================


def test_engineer_causal_features_competitor_gap(real_retail_data: pd.DataFrame):
    """Test that competitor-price gap features are engineered accurately and without leakage."""
    features = engineer_causal_features(real_retail_data)

    assert "comp_price_diff_1" in features.columns
    assert "comp_price_ratio_1" in features.columns
    assert "comp_price_min" in features.columns
    assert "comp_price_max" in features.columns
    assert "comp_price_mean" in features.columns
    assert "comp_price_spread" in features.columns
    assert "lag_price_diff_comp_min" in features.columns

    # Verify mathematical identity: comp_price_diff_1 = lag_price - comp_1
    expected_diff = real_retail_data["lag_price"] - real_retail_data["comp_1"]
    np.testing.assert_allclose(features["comp_price_diff_1"].values, expected_diff.values)

    # Competitor spread must be strictly non-negative
    assert (features["comp_price_spread"] >= 0.0).all()

    # CRITICAL: Verify unit_price is NOT present in any form
    assert "unit_price" not in features.columns
    assert "price" not in features.columns


def test_engineer_causal_features_seasonality(real_retail_data: pd.DataFrame):
    """Test cyclical seasonality and calendar feature transformations."""
    features = engineer_causal_features(real_retail_data)

    assert "sin_month" in features.columns
    assert "cos_month" in features.columns
    assert "is_q4" in features.columns
    assert "weekend_ratio" in features.columns
    assert "holiday_ratio" in features.columns

    # Cyclical sin and cos must remain within [-1, 1]
    assert (features["sin_month"] >= -1.0 - 1e-6).all() and (features["sin_month"] <= 1.0 + 1e-6).all()
    assert (features["cos_month"] >= -1.0 - 1e-6).all() and (features["cos_month"] <= 1.0 + 1e-6).all()

    # Trigonometric identity: sin^2 + cos^2 = 1
    trig_sum = np.square(features["sin_month"]) + np.square(features["cos_month"])
    np.testing.assert_allclose(trig_sum.values, 1.0, atol=1e-5)

    # Q4 indicator correctness
    q4_mask = real_retail_data["month"].isin([10, 11, 12])
    np.testing.assert_array_equal(features["is_q4"].values, q4_mask.astype(float).values)


def test_engineer_causal_features_discount_history(real_retail_data: pd.DataFrame):
    """Test that pre-treatment historical discount depth is computed properly."""
    features = engineer_causal_features(real_retail_data)

    assert "historical_discount_rate" in features.columns
    assert "is_lag_discounted" in features.columns

    # Discount rates must be bounded between [0.0, 1.0]
    assert (features["historical_discount_rate"] >= 0.0).all()
    assert (features["historical_discount_rate"] <= 1.0).all()

    # Binary flag consistency
    expected_flag = (features["historical_discount_rate"] > 0.05).astype(float)
    np.testing.assert_array_equal(features["is_lag_discounted"].values, expected_flag.values)


def test_engineer_causal_features_anti_leakage_filter():
    """Verify that engineer_causal_features automatically purges post-treatment columns."""
    corrupted_df = pd.DataFrame(
        {
            "product_id": ["p1", "p2"],
            "unit_price": [50.0, 60.0],
            "qty": [10, 20],
            "total_price": [500.0, 1200.0],
            "revenue": [500.0, 1200.0],
            "profit": [200.0, 400.0],
            "comp_1": [48.0, 58.0],
            "lag_price": [52.0, 62.0],
            "freight_price": [10.0, 12.0],
            "weekday": [20, 21],
            "weekend": [8, 9],
            "holiday": [1, 2],
            "month": [5, 6],
        }
    )

    feat = engineer_causal_features(corrupted_df, drop_leakage=True)

    # None of the post-treatment or treatment columns should remain in features
    for forbidden in ["qty", "total_price", "revenue", "profit", "unit_price"]:
        assert forbidden not in feat.columns


# ==============================================================================
# 7. END-TO-END SPLIT CAUSAL DATA WITH FEATURE ENGINEERING
# ==============================================================================


def test_split_causal_data_with_feature_engineering(real_retail_data: pd.DataFrame):
    """Test split_causal_data with engineer_features=True on Kaggle retail dataset."""
    splits = split_causal_data(
        real_retail_data,
        test_size=0.2,
        random_state=42,
        engineer_features=True,
        categorical_confounders=["product_category_name"],
    )

    X_train = splits["X_train"]
    X_test = splits["X_test"]
    T_train = splits["T_train"]
    T_test = splits["T_test"]
    Y_train = splits["Y_train"]
    Y_test = splits["Y_test"]
    audit = splits["leakage_audit"]

    assert audit["is_clean"] is True
    assert len(X_train) == 540
    assert len(X_test) == 136
    assert len(T_train) == 540
    assert len(T_test) == 136
    assert len(Y_train) == 540
    assert len(Y_test) == 136

    # Ensure engineered features are present in feature names
    feature_names = splits["feature_names"]
    assert "comp_price_diff_1" in feature_names
    assert "sin_month" in feature_names
    assert "historical_discount_rate" in feature_names
    assert any("product_category_name" in f for f in feature_names)

    # Strict anti-leakage checks on matrices
    assert "unit_price" not in X_train.columns
    assert "qty" not in X_train.columns
    assert "total_price" not in X_train.columns
    assert not X_train.isnull().values.any()
    assert not X_test.isnull().values.any()
