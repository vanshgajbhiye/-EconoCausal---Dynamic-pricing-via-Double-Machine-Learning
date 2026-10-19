"""Unit tests for Double Machine Learning (DML) / Robinson's Residualization Engine.

Tests synthetic ground-truth recovery, cross-fitting integrity, orthogonality,
different nuisance regressors, and real dataset execution on retail_price.csv.
"""

import numpy as np
import pandas as pd
import pytest
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import Ridge

from src.data.load_data import load_retail_price_dataset
from src.models.train_dml import DoubleMachineLearning, train_dml_pipeline


@pytest.fixture
def synthetic_causal_data():
    """Generate synthetic data with known ground-truth causal parameter theta = -2.5.

    Structural Data Generating Process (DGP):
        X ~ N(0, I_5)
        T = 0.6*X_0 + 0.8*X_1 - 0.4*X_2 + eta_T,  eta_T ~ N(0, 1)
        Y = theta * T + 1.2*X_0 - 0.7*X_1 + 0.5*X_3 + eta_Y,  eta_Y ~ N(0, 1)
    """
    np.random.seed(42)
    n_samples = 1000
    n_features = 5
    true_theta = -2.5

    X = np.random.randn(n_samples, n_features)
    # Confounders drive treatment assignment T
    T = 0.6 * X[:, 0] + 0.8 * X[:, 1] - 0.4 * X[:, 2] + np.random.randn(n_samples)
    # Confounders and treatment drive outcome Y
    Y = true_theta * T + 1.2 * X[:, 0] - 0.7 * X[:, 1] + 0.5 * X[:, 3] + np.random.randn(n_samples)

    return X, T, Y, true_theta


def test_dml_synthetic_ground_truth_recovery(synthetic_causal_data):
    """Test that DML accurately recovers known ground truth causal effect theta."""
    X, T, Y, true_theta = synthetic_causal_data

    dml = DoubleMachineLearning(
        model_y=Ridge(alpha=1.0),
        model_t=Ridge(alpha=1.0),
        n_folds=5,
        random_state=42,
    )
    dml.fit(X, T, Y)

    # 1. Parameter estimate should be very close to true theta (-2.5)
    assert abs(dml.theta_ - true_theta) < 0.15, (
        f"Estimated theta ({dml.theta_:.4f}) deviates too far from true theta ({true_theta})"
    )

    # 2. 95% Confidence interval must cover true theta
    assert dml.ci_[0] <= true_theta <= dml.ci_[1], (
        f"True theta {true_theta} not in 95% CI [{dml.ci_[0]:.4f}, {dml.ci_[1]:.4f}]"
    )

    # 3. Statistical inference properties
    assert dml.std_error_ > 0
    assert dml.std_error_ < 0.1  # with N=1000, SE should be tight
    assert dml.p_value_ < 1e-10  # highly statistically significant
    assert dml.t_stat_ < -20.0   # strong negative test statistic


def test_dml_cross_fitting_and_residual_shapes(synthetic_causal_data):
    """Test that K-fold cross-fitting creates out-of-fold residuals of exact sample length."""
    X, T, Y, _ = synthetic_causal_data
    n_samples = len(X)
    n_folds = 5

    dml = DoubleMachineLearning(n_folds=n_folds, random_state=42)
    dml.fit(X, T, Y)

    # Verify lengths
    assert len(dml.residuals_y_) == n_samples
    assert len(dml.residuals_t_) == n_samples
    assert len(dml.pred_y_) == n_samples
    assert len(dml.pred_t_) == n_samples

    # Verify fold count
    assert len(dml.models_y_) == n_folds
    assert len(dml.models_t_) == n_folds

    # Verify nuisance scores
    assert "r2_y" in dml.nuisance_scores_
    assert "r2_t" in dml.nuisance_scores_
    assert dml.nuisance_scores_["r2_y"] > 0.0
    assert dml.nuisance_scores_["r2_t"] > 0.0


def test_dml_orthogonality_condition(synthetic_causal_data):
    """Test Neyman orthogonality: <T_tilde, epsilon_hat> must be algebraically zero."""
    X, T, Y, _ = synthetic_causal_data

    dml = DoubleMachineLearning(
        model_y=Ridge(alpha=1.0),
        model_t=Ridge(alpha=1.0),
        n_folds=5,
        random_state=42,
    )
    dml.fit(X, T, Y)

    # 1. Exact uncentered inner product / cosine orthogonality score
    ortho_score = dml.residual_diagnostics_["orthogonality_score"]
    assert ortho_score < 1e-10, f"Orthogonality score {ortho_score} is not near zero"

    # 2. Centered Pearson correlation
    ortho_corr = dml.residual_diagnostics_["orthogonality_corr"]
    assert abs(ortho_corr) < 1e-3, f"Orthogonality correlation {ortho_corr} is not small"


def test_dml_with_ensemble_estimators(synthetic_causal_data):
    """Test DML with non-linear tree-based nuisance estimators."""
    X, T, Y, true_theta = synthetic_causal_data

    dml_rf = DoubleMachineLearning(
        model_y=RandomForestRegressor(n_estimators=30, max_depth=4, random_state=42),
        model_t=RandomForestRegressor(n_estimators=30, max_depth=4, random_state=42),
        n_folds=3,
        random_state=42,
    )
    dml_rf.fit(X, T, Y)
    assert abs(dml_rf.theta_ - true_theta) < 0.25

    dml_gb = DoubleMachineLearning(
        model_y=GradientBoostingRegressor(n_estimators=30, max_depth=3, random_state=42),
        model_t=GradientBoostingRegressor(n_estimators=30, max_depth=3, random_state=42),
        n_folds=3,
        random_state=42,
    )
    dml_gb.fit(X, T, Y)
    assert abs(dml_gb.theta_ - true_theta) < 0.25


def test_dml_counterfactual_prediction(synthetic_causal_data):
    """Test counterfactual outcome prediction Delta Y = theta * Delta T."""
    X, T, Y, true_theta = synthetic_causal_data

    dml = DoubleMachineLearning(
        model_y=Ridge(alpha=1.0),
        model_t=Ridge(alpha=1.0),
        n_folds=5,
        random_state=42,
    ).fit(X, T, Y)

    # Price increase by 10 units -> demand change should be approx theta * 10
    delta_y = dml.predict_counterfactual(T_new=20.0, T_old=10.0)
    expected_delta_y = dml.theta_ * 10.0
    assert np.isclose(delta_y, expected_delta_y)
    assert delta_y < 0  # Demand decreases when price increases


def test_dml_validation_errors():
    """Test input validations and edge cases."""
    dml = DoubleMachineLearning(n_folds=5)

    # Calling summary before fit should raise ValueError
    with pytest.raises(ValueError, match="not fitted"):
        dml.summary()

    with pytest.raises(ValueError, match="not fitted"):
        dml.format_summary_table()

    with pytest.raises(ValueError, match="not fitted"):
        dml.predict_counterfactual(10.0, 5.0)

    # Mismatched sample sizes
    X = np.ones((50, 3))
    T = np.ones(50)
    Y_short = np.ones(40)
    with pytest.raises(ValueError, match="mismatch"):
        dml.fit(X, T, Y_short)

    # NaNs in input
    X_nan = X.copy()
    X_nan[0, 0] = np.nan
    with pytest.raises(ValueError, match="NaN"):
        dml.fit(X_nan, T, np.ones(50))

    # Invalid fold count
    with pytest.raises(ValueError, match="at least 2"):
        DoubleMachineLearning(n_folds=1)


def test_dml_format_summary_table(synthetic_causal_data):
    """Test summary and table formatting produce readable diagnostic strings."""
    X, T, Y, _ = synthetic_causal_data
    dml = DoubleMachineLearning(
        model_y=Ridge(),
        model_t=Ridge(),
        n_folds=3,
        random_state=42,
    ).fit(X, T, Y)

    summary = dml.summary()
    assert "theta" in summary
    assert "std_error" in summary
    assert "ci_lower" in summary
    assert "ci_upper" in summary

    table = dml.format_summary_table()
    assert "DOUBLE MACHINE LEARNING" in table
    assert "Causal Parameter" in table
    assert "NUISANCE ESTIMATOR PERFORMANCE" in table
    assert "RESIDUAL ORTHOGONALITY DIAGNOSTICS" in table


def test_train_dml_pipeline_real_retail_price_dataset():
    """Test full DML pipeline on the official Kaggle Retail Price Optimization dataset."""
    df = load_retail_price_dataset("data/raw/retail_price.csv")

    # 1. Elasticity (Log-Log) Specification
    results_log = train_dml_pipeline(
        df=df,
        treatment_col="unit_price",
        outcome_col="qty",
        categorical_confounders=["product_category_name"],
        log_transform_treatment=True,
        log_transform_outcome=True,
        n_folds=5,
        random_state=42,
    )

    model_log = results_log["model"]
    summary_log = results_log["summary"]

    assert model_log._is_fitted is True
    # In economic demand, price elasticity should be negative
    assert summary_log["theta"] < 0, f"Expected negative price elasticity, got {summary_log['theta']}"
    assert summary_log["std_error"] > 0
    assert summary_log["p_value"] < 0.05, f"Elasticity should be statistically significant, p={summary_log['p_value']}"

    # 2. Linear Demand (Marginal units per dollar) Specification
    results_linear = train_dml_pipeline(
        df=df,
        treatment_col="unit_price",
        outcome_col="qty",
        categorical_confounders=["product_category_name"],
        log_transform_treatment=False,
        log_transform_outcome=False,
        n_folds=5,
        random_state=42,
    )

    summary_linear = results_linear["summary"]
    assert summary_linear["theta"] < 0, f"Expected negative marginal demand slope, got {summary_linear['theta']}"
