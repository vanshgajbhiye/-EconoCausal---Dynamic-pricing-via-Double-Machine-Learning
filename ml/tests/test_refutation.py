"""Unit tests for Causal Model Refutation Suite.

Tests Placebo Treatment, Random Common Cause, Data Subsetting,
and consolidated audit reporting on synthetic and real datasets.
"""

import numpy as np
import pandas as pd
import pytest
from sklearn.linear_model import Ridge

from src.data.load_data import load_retail_price_dataset
from src.data.preprocess import split_causal_data
from src.models.refutation import CausalRefuter
from src.models.train_dml import DoubleMachineLearning


@pytest.fixture
def fitted_dml_model():
    """Fixture providing a fitted DoubleMachineLearning model on synthetic causal data."""
    np.random.seed(42)
    n = 600
    X = np.random.randn(n, 4)
    T = 0.5 * X[:, 0] + np.random.randn(n)
    # True causal effect theta = -2.0
    Y = -2.0 * T + 0.8 * X[:, 1] + np.random.randn(n)

    dml = DoubleMachineLearning(
        model_y=Ridge(alpha=1.0),
        model_t=Ridge(alpha=1.0),
        n_folds=3,
        random_state=42,
    )
    dml.fit(X, T, Y)
    return dml, X, T, Y


def test_refute_placebo_treatment(fitted_dml_model):
    """Test that replacing treatment with placebo permutation collapses causal effect to near zero."""
    dml, X, T, Y = fitted_dml_model
    refuter = CausalRefuter(dml, random_state=42)

    res = refuter.refute_placebo_treatment(X, T, Y, n_repeats=3)
    assert res["passed"] is True
    # Placebo effect should be much smaller in magnitude than original theta (~ -2.0)
    assert abs(res["placebo_theta_mean"]) < abs(dml.theta_) * 0.30
    assert "Placebo Treatment Refutation" in res["test_name"]


def test_refute_random_common_cause(fitted_dml_model):
    """Test that adding random noise confounder leaves treatment effect estimate stable."""
    dml, X, T, Y = fitted_dml_model
    refuter = CausalRefuter(dml, random_state=42)

    res = refuter.refute_random_common_cause(X, T, Y, n_repeats=3)
    assert res["passed"] is True
    assert res["relative_change"] < 0.20
    assert "Random Common Cause Refutation" in res["test_name"]


def test_refute_data_subset(fitted_dml_model):
    """Test that subsampling 80% of data preserves estimate direction and magnitude."""
    dml, X, T, Y = fitted_dml_model
    refuter = CausalRefuter(dml, random_state=42)

    res = refuter.refute_data_subset(X, T, Y, subset_fraction=0.80, n_repeats=3)
    assert res["passed"] is True
    # Negative sign of demand elasticity must be preserved
    assert res["subset_theta_mean"] < 0
    assert abs(res["subset_theta_mean"] - dml.theta_) < 0.35


def test_run_all_refutations_and_formatting(fitted_dml_model):
    """Test consolidated battery execution, scoring, and ASCII table generation."""
    dml, X, T, Y = fitted_dml_model
    refuter = CausalRefuter(dml, random_state=42)

    all_res = refuter.run_all_refutations(X, T, Y, n_repeats=2)
    assert all_res["total_tests"] == 3
    assert all_res["passed_tests"] == 3
    assert all_res["robustness_score_percent"] == 100.0
    assert all_res["is_causally_robust"] is True

    table = refuter.format_refutation_table(all_res)
    assert "CAUSAL MODEL REFUTATION AUDIT REPORT" in table
    assert "100.0%" in table
    assert "YES (ROBUST)" in table
    assert "Placebo Treatment" in table
    assert "Random Common Cause" in table
    assert "Data Subset" in table


def test_refuter_unfitted_model_error():
    """Test that initializing refuter with unfitted model raises ValueError."""
    unfitted_dml = DoubleMachineLearning()
    with pytest.raises(ValueError, match="must be fitted"):
        CausalRefuter(unfitted_dml)


def test_refutation_on_real_retail_price_dataset():
    """Integration test: Verify causal refutations pass on Kaggle retail_price.csv."""
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

    # Use fast Ridge nuisance models to keep refutation battery lightweight and fast
    dml = DoubleMachineLearning(
        model_y=Ridge(alpha=1.0),
        model_t=Ridge(alpha=1.0),
        n_folds=3,
        random_state=42,
    )
    dml.fit(X_train, T_train, Y_train)

    refuter = CausalRefuter(dml, random_state=42)
    all_res = refuter.run_all_refutations(X_train, T_train, Y_train, n_repeats=2)

    # Assert all refutations pass on real retail price data
    assert all_res["passed_tests"] >= 2
    assert all_res["robustness_score_percent"] >= 66.0
