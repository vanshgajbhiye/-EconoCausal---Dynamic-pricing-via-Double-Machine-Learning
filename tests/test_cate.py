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


def test_cate_uncertainty_quantification(synthetic_heterogeneous_data):
    """Test individual CATE prediction with sandwich standard errors and confidence intervals."""
    X_df, T, Y, _ = synthetic_heterogeneous_data
    cate_est = CATEEstimator(
        dml_base=DoubleMachineLearning(model_y=Ridge(alpha=0.5), model_t=Ridge(alpha=0.5), n_folds=3),
        moderator_cols=["feat_0", "feat_1"],
        alpha=0.1,
        random_state=42,
    ).fit(X_df, T, Y)

    uncert_df = cate_est.predict_cate_with_uncertainty(X_df, alpha=0.05)

    assert isinstance(uncert_df, pd.DataFrame)
    assert len(uncert_df) == len(X_df)
    for col in [
        "predicted_elasticity",
        "std_error",
        "ci_lower",
        "ci_upper",
        "t_stat_unit",
        "p_value_unit",
        "elasticity_regime",
        "is_statistically_significant",
    ]:
        assert col in uncert_df.columns

    # Standard errors must be strictly positive
    assert (uncert_df["std_error"] > 0).all()

    # Confidence interval ordering: ci_lower < predicted_elasticity < ci_upper
    assert (uncert_df["ci_lower"] < uncert_df["predicted_elasticity"]).all()
    assert (uncert_df["predicted_elasticity"] < uncert_df["ci_upper"]).all()

    # Point estimate must match predict_cate
    preds = cate_est.predict_cate(X_df)
    np.testing.assert_allclose(uncert_df["predicted_elasticity"].values, preds, rtol=1e-5)


def test_unit_elasticity_boundary_interpretation():
    """Test statistical hypothesis testing against the unit-elasticity boundary theta = -1.0.

    Under log-log demand (Y = log(Q), T = log(P)):
        dR / dP = (1 + theta) * Q(P)
    - theta < -1: Elastic demand (price cuts expand revenue).
    - theta = -1: Unitary elasticity (marginal revenue = 0).
    - -1 < theta < 0: Inelastic demand (price hikes expand revenue).
    """
    np.random.seed(42)
    n = 600
    # Group indicator: 0 -> elastic (theta=-2.2), 1 -> unit-elastic (theta=-1.0), 2 -> inelastic (theta=-0.4)
    group = np.random.choice([0, 1, 2], size=n)
    G = pd.get_dummies(group, prefix="grp", drop_first=False, dtype=float)

    # Elasticity per group
    true_thetas = np.array([-2.2, -1.0, -0.4])
    theta_i = true_thetas[group]

    T = np.random.randn(n)
    Y = theta_i * T + np.random.randn(n) * 0.5
    X = G.copy()

    cate_est = CATEEstimator(
        dml_base=DoubleMachineLearning(model_y=Ridge(), model_t=Ridge(), n_folds=3),
        moderator_cols=list(G.columns),
        include_intercept=False,
        alpha=0.01,
        random_state=42,
    ).fit(X, T, Y)

    labels = pd.Series(group).map({0: "Elastic_Group", 1: "Unit_Elastic_Group", 2: "Inelastic_Group"})
    summary = cate_est.cate_by_segment(labels, X=X, alpha=0.05)

    assert "t_stat_unit" in summary.columns
    assert "p_value_unit" in summary.columns
    assert "elasticity_regime" in summary.columns

    # 1. Unit_Elastic_Group (true theta = -1.0) must fail to reject H0: theta = -1.0 (p >= 0.05)
    unit_row = summary.loc[summary["segment"] == "Unit_Elastic_Group"].iloc[0]
    assert abs(unit_row["mean_elasticity"] - (-1.0)) < 0.25
    assert unit_row["p_value_unit"] >= 0.05, f"Unit group p-value {unit_row['p_value_unit']} should be >= 0.05"
    assert unit_row["elasticity_regime"] == "Unit Elastic"

    # 2. Elastic_Group (true theta = -2.2) must reject H0 and have theta < -1.0
    elastic_row = summary.loc[summary["segment"] == "Elastic_Group"].iloc[0]
    assert elastic_row["mean_elasticity"] < -1.5
    assert elastic_row["p_value_unit"] < 0.05
    assert elastic_row["elasticity_regime"] in ["Elastic", "Highly Elastic"]

    # 3. Inelastic_Group (true theta = -0.4) must reject H0 and have -1.0 < theta < 0.0
    inelastic_row = summary.loc[summary["segment"] == "Inelastic_Group"].iloc[0]
    assert inelastic_row["mean_elasticity"] > -0.7
    assert inelastic_row["p_value_unit"] < 0.05
    assert inelastic_row["elasticity_regime"] == "Inelastic"


def test_subgroup_heterogeneity_vs_overall_ate():
    """Test comparison of subgroup elasticities against overall catalog ATE."""
    np.random.seed(42)
    n = 400
    group = np.random.choice(["High_Elasticity", "Average_Elasticity"], size=n)
    g_num = (group == "High_Elasticity").astype(float)

    # Treatment effect: average = -1.0, high elasticity = -2.5
    theta_i = np.where(g_num == 1, -2.5, -1.0)
    T = np.random.randn(n)
    Y = theta_i * T + np.random.randn(n) * 0.4
    X = pd.DataFrame({"is_high": g_num})

    cate_est = CATEEstimator(
        dml_base=DoubleMachineLearning(model_y=Ridge(), model_t=Ridge(), n_folds=3),
        moderator_cols=["is_high"],
        alpha=0.01,
        random_state=42,
    ).fit(X, T, Y)

    summary = cate_est.cate_by_segment(group, X=X)
    assert "t_stat_vs_overall" in summary.columns
    assert "p_value_vs_overall" in summary.columns
    assert "is_heterogeneous_vs_overall" in summary.columns

    # High_Elasticity must be statistically significantly different from overall ATE
    high_row = summary.loc[summary["segment"] == "High_Elasticity"].iloc[0]
    assert high_row["is_heterogeneous_vs_overall"] is True or high_row["p_value_vs_overall"] < 0.10


def test_flag_unsupported_extrapolations():
    """Test detection of small sample size, high uncertainty, and out-of-support covariates."""
    np.random.seed(42)
    n = 200
    X = pd.DataFrame({
        "feature_1": np.random.uniform(0.0, 10.0, size=n),
        "feature_2": np.random.randn(n),
    })
    T = np.random.randn(n)
    Y = -1.2 * T + np.random.randn(n) * 0.5

    cate_est = CATEEstimator(
        dml_base=DoubleMachineLearning(model_y=Ridge(), model_t=Ridge(), n_folds=3),
        alpha=0.1,
    ).fit(X, T, Y)

    # Segment labels with one very small group (N=5 < 20)
    labels = ["Normal_Group"] * 195 + ["Sparse_Group"] * 5
    summary = cate_est.cate_by_segment(labels, X=X, min_sample_size=20)

    sparse_row = summary.loc[summary["segment"] == "Sparse_Group"].iloc[0]
    assert bool(sparse_row["is_unsupported_extrapolation"]) is True
    assert "Small sample size" in sparse_row["extrapolation_reasons"]

    # Query observations that drastically exceed training support bounds [0, 10]
    X_out_of_support = pd.DataFrame({
        "feature_1": [100.0, -50.0],  # Out of [0, 10]
        "feature_2": [0.0, 0.0],
    })
    support_check = cate_est.check_support_and_extrapolation(X_out_of_support)
    assert bool(support_check["is_unsupported_extrapolation"].all()) is True
    assert (support_check["bounds_violated_features"] > 0).all()


def test_subgroup_stability_kfold(synthetic_heterogeneous_data):
    """Test cross-fitting fold stability evaluation across segments."""
    X_df, T, Y, _ = synthetic_heterogeneous_data
    cate_est = CATEEstimator(
        moderator_cols=["feat_0"],
        alpha=0.1,
        random_state=42,
    )
    labels = np.where(X_df["feat_0"] > 0, "High_Segment", "Low_Segment")
    stab_df = cate_est.evaluate_subgroup_stability(
        X=X_df,
        T=T,
        Y=Y,
        segment_labels=labels,
        n_splits=3,
        random_state=42,
    )

    assert isinstance(stab_df, pd.DataFrame)
    assert len(stab_df) == 2
    for col in [
        "segment",
        "mean_across_folds",
        "std_across_folds",
        "cv_dispersion",
        "negative_elasticity_pct",
        "stability_rating",
    ]:
        assert col in stab_df.columns
    # Both segments should preserve negative elasticity across folds
    assert (stab_df["negative_elasticity_pct"] == 100.0).all()
    assert (stab_df["stability_rating"].isin(["HIGHLY_STABLE", "MODERATELY_STABLE"])).all()


def test_retail_customer_footfall_heterogeneity():
    """Test customer footfall heterogeneity on retail dataset: high traffic vs low traffic."""
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

    cat_cols = [c for c in X_train.columns if "product_category_name_" in c]
    mod_cols = cat_cols + ["customers"]

    dml_base = DoubleMachineLearning(
        n_folds=3,
        random_state=42,
    )
    cate_est = CATEEstimator(
        dml_base=dml_base,
        moderator_cols=mod_cols,
        alpha=1.0,
        random_state=42,
    ).fit(X_train, T_train, Y_train)

    train_df = df.loc[X_train.index]
    footfall_labels = np.where(train_df["customers"] > 100, "High_Footfall", "Low_Footfall")
    summary = cate_est.cate_by_segment(footfall_labels, X=X_train)

    high_e = summary.loc[summary["segment"] == "High_Footfall", "mean_elasticity"].iloc[0]
    low_e = summary.loc[summary["segment"] == "Low_Footfall", "mean_elasticity"].iloc[0]

    # High footfall (deal-seekers / high volume) is significantly more elastic (more negative)
    assert high_e < low_e
    assert high_e < -1.0  # High footfall is elastic


def test_retail_category_unit_elasticity_and_extrapolation():
    """Test category elasticities and extrapolation flags on retail dataset."""
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

    cat_cols = [c for c in X_train.columns if "product_category_name_" in c]
    dml_base = DoubleMachineLearning(
        n_folds=3,
        random_state=42,
    )
    cate_est = CATEEstimator(
        dml_base=dml_base,
        moderator_cols=cat_cols,
        alpha=1.0,
        random_state=42,
    ).fit(X_train, T_train, Y_train)

    train_cats = df.loc[X_train.index, "product_category_name"]
    summary = cate_est.cate_by_segment(train_cats, X=X_train, min_sample_size=20)

    # 1. Computers & accessories should be the most price-sensitive (most negative elasticity)
    comp_e = summary.loc[summary["segment"] == "computers_accessories", "mean_elasticity"].iloc[0]
    assert comp_e < -1.4, f"computers_accessories elasticity {comp_e} should be highly elastic"

    # 2. Sparse category (consoles_games with N=15 < 20) must be flagged as unsupported extrapolation
    sparse_cat = summary.loc[summary["segment"] == "consoles_games"].iloc[0]
    assert bool(sparse_cat["is_unsupported_extrapolation"]) is True
    assert "Small sample size" in sparse_cat["extrapolation_reasons"]

