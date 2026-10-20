"""Unit tests for Heterogeneous Treatment Effects (CATE) engine.

Tests synthetic ground-truth moderation recovery, segment aggregation,
moderator subset selection, and real-data execution on retail_price.csv.
"""

import numpy as np
import pandas as pd
import pytest
from sklearn.linear_model import Ridge

from src.data.load_data import load_retail_price_dataset
from src.data.preprocess import split_causal_data
from src.models.cate import CATEEstimator
from src.models.train_dml import DoubleMachineLearning


@pytest.fixture
def synthetic_heterogeneous_data():
    """Generate synthetic data with true heterogeneous treatment effect:
        theta(X) = -2.0 + 1.2 * X_0 - 0.8 * X_1
    """
    np.random.seed(42)
    n_samples = 800
    n_features = 4

    X = np.random.randn(n_samples, n_features)
    X_df = pd.DataFrame(X, columns=[f"feat_{i}" for i in range(n_features)])

    # True heterogeneous causal elasticity function
    true_theta_i = -2.0 + 1.2 * X[:, 0] - 0.8 * X[:, 1]

    # Treatment depends on confounders + noise
    T = 0.5 * X[:, 0] + 0.3 * X[:, 2] + np.random.randn(n_samples)
    # Outcome depends on heterogeneous treatment + baseline confounders + noise
    Y = true_theta_i * T + 1.0 * X[:, 0] - 0.5 * X[:, 3] + np.random.randn(n_samples)

    return X_df, pd.Series(T, name="treatment"), pd.Series(Y, name="outcome"), true_theta_i


def test_cate_synthetic_heterogeneity_recovery(synthetic_heterogeneous_data):
    """Test that CATEEstimator recovers known ground-truth moderation slopes."""
    X_df, T, Y, true_theta_i = synthetic_heterogeneous_data

    dml_base = DoubleMachineLearning(
        model_y=Ridge(alpha=0.5),
        model_t=Ridge(alpha=0.5),
        n_folds=3,
        random_state=42,
    )
    cate_est = CATEEstimator(
        dml_base=dml_base,
        moderator_cols=["feat_0", "feat_1"],
        alpha=0.1,
        random_state=42,
    )
    cate_est.fit(X_df, T, Y)

    # 1. Recovered baseline intercept should be near -2.0
    beta = cate_est.beta_
    assert abs(beta[0] - (-2.0)) < 0.35, f"Baseline intercept {beta[0]} deviates from true -2.0"

    # 2. Moderation slopes should match true coefficients (+1.2 and -0.8)
    assert abs(beta[1] - 1.2) < 0.35, f"Moderator feat_0 slope {beta[1]} deviates from true 1.2"
    assert abs(beta[2] - (-0.8)) < 0.35, f"Moderator feat_1 slope {beta[2]} deviates from true -0.8"

    # 3. Individual predictions should correlate strongly with ground-truth
    preds = cate_est.predict_cate(X_df)
    corr = np.corrcoef(preds, true_theta_i)[0, 1]
    assert corr > 0.85, f"Correlation between predicted and true CATE {corr:.3f} is too low"


def test_cate_by_segment_aggregation(synthetic_heterogeneous_data):
    """Test CATE aggregation by segments/categories and pricing recommendations."""
    X_df, T, Y, _ = synthetic_heterogeneous_data
    cate_est = CATEEstimator(
        dml_base=DoubleMachineLearning(model_y=Ridge(), model_t=Ridge(), n_folds=3),
        moderator_cols=["feat_0"],
    ).fit(X_df, T, Y)

    # Assign synthetic segment labels
    segments = np.where(X_df["feat_0"] > 0, "High_Sensitivity", "Low_Sensitivity")
    segment_table = cate_est.cate_by_segment(segments)

    assert "segment" in segment_table.columns
    assert "mean_elasticity" in segment_table.columns
    assert "pricing_strategy" in segment_table.columns
    assert len(segment_table) == 2
    # Verify High_Sensitivity has higher (less negative) elasticity than Low_Sensitivity due to +1.2 slope
    high_e = segment_table.loc[segment_table["segment"] == "High_Sensitivity", "mean_elasticity"].iloc[0]
    low_e = segment_table.loc[segment_table["segment"] == "Low_Sensitivity", "mean_elasticity"].iloc[0]
    assert high_e > low_e


def test_cate_error_handling(synthetic_heterogeneous_data):
    """Test validation errors for unfitted calls and mismatched segment lengths."""
    X_df, T, Y, _ = synthetic_heterogeneous_data
    cate_est = CATEEstimator()

    with pytest.raises(ValueError, match="not fitted"):
        cate_est.predict_cate(X_df)

    with pytest.raises(ValueError, match="not fitted"):
        cate_est.summary()

    with pytest.raises(ValueError, match="not fitted"):
        cate_est.cate_by_segment(pd.Series(["A", "B"]))

    # Mismatched segment length after fitting
    cate_est.fit(
        X_df, T, Y,
    )
    with pytest.raises(ValueError, match="Length mismatch"):
        cate_est.cate_by_segment(pd.Series(["A", "B"]))  # 2 labels vs 800 samples


def test_cate_summary_and_formatting(synthetic_heterogeneous_data):
    """Test summary dictionary and ASCII report formatting."""
    X_df, T, Y, _ = synthetic_heterogeneous_data
    cate_est = CATEEstimator(
        dml_base=DoubleMachineLearning(model_y=Ridge(), model_t=Ridge(), n_folds=3),
        moderator_cols=["feat_0", "feat_1"],
    ).fit(X_df, T, Y)

    summary = cate_est.summary()
    assert "mean_cate" in summary
    assert "median_cate" in summary
    assert "std_cate" in summary
    assert "moderation_table" in summary

    table_str = cate_est.format_summary_table()
    assert "CONDITIONAL AVERAGE TREATMENT EFFECT" in table_str
    assert "Mean Predicted Elasticity" in table_str


def test_cate_real_retail_price_dataset():
    """Test CATE estimation on official Kaggle Retail Price Optimization dataset."""
    df = load_retail_price_dataset("data/raw/retail_price.csv")

    splits = split_causal_data(
        df=df,
        test_size=0.2,
        random_state=42,
        treatment_col="unit_price",
        outcome_col="qty",
        categorical_confounders=["product_category_name"],
        log_transform_treatment=True,
        log_transform_outcome=True,
    )

    X_train = splits["X_train"]
    T_train = splits["T_train"]
    Y_train = splits["Y_train"]

    # Filter category indicator columns to use as moderators
    cat_cols = [c for c in X_train.columns if "product_category_name_" in c]
    assert len(cat_cols) > 0, "Category columns must exist in preprocessed data"

    dml_base = DoubleMachineLearning(n_folds=5, random_state=42)
    cate_est = CATEEstimator(
        dml_base=dml_base,
        moderator_cols=cat_cols,
        alpha=1.0,
        random_state=42,
    )
    cate_est.fit(X_train, T_train, Y_train)

    cates = cate_est.predict_cate(X_train)
    assert len(cates) == len(X_train)
    # The average CATE should align closely with the overall ATE
    assert abs(np.mean(cates) - dml_base.theta_) < 0.20

    # Test category aggregation using raw category labels for training indices
    train_cats = df.loc[X_train.index, "product_category_name"]
    category_summary = cate_est.cate_by_segment(train_cats)

    assert len(category_summary) >= 5
    assert "mean_elasticity" in category_summary.columns
    # Check that categories have distinct elasticities (heterogeneity is present)
    assert category_summary["mean_elasticity"].nunique() > 1
