"""Unit and integration tests for Causal Model Refutation and Sensitivity Suite.

Covers:
- Extended Placebo Treatment (permutation and random noise modes)
- Extended Random Common Cause (gaussian, uniform, and correlated pseudo-confounders)
- Extended Data Subsetting (subsampling stability, CV, sign preservation)
- Sensitivity to Plausible Unobserved Confounding (Cinelli-Hazlett Robustness Values and Benchmarks)
- Alternative Reasonable Specifications (Specification curve across learners and forms)
- Epistemic Validity Assessment (Statistical corroboration vs. DAG assumptions)
- Evidence-Based Governance Rules (ACCEPT, WARNING, REJECT)
- Consolidated Advanced Audit Reporting
- Integration on Kaggle retail price dataset
"""

import numpy as np
import pandas as pd
import pytest
from sklearn.linear_model import Ridge

from src.data.load_data import load_retail_price_dataset
from src.data.preprocess import split_causal_data
from src.models.refutation import CausalDecision, CausalRefuter
from src.models.train_dml import DoubleMachineLearning


@pytest.fixture
def fitted_dml_model():
    """Fixture providing a fitted DoubleMachineLearning model on synthetic causal data."""
    np.random.seed(42)
    n = 600
    # 4 covariates
    X = np.random.randn(n, 4)
    # Treatment influenced by X[:, 0]
    T = 0.5 * X[:, 0] + np.random.randn(n)
    # Outcome: true causal elasticity theta = -2.0, influenced by X[:, 1]
    Y = -2.0 * T + 0.8 * X[:, 1] + np.random.randn(n)

    dml = DoubleMachineLearning(
        model_y=Ridge(alpha=1.0),
        model_t=Ridge(alpha=1.0),
        n_folds=3,
        random_state=42,
    )
    dml.fit(X, T, Y)
    return dml, X, T, Y


@pytest.fixture
def synthetic_retail_df():
    """Fixture providing synthetic retail transactions with named columns for specification testing."""
    rng = np.random.RandomState(42)
    n = 300
    unit_price = rng.uniform(20.0, 150.0, size=n)
    comp_1 = unit_price * rng.uniform(0.9, 1.1, size=n)
    comp_2 = unit_price * rng.uniform(0.85, 1.15, size=n)
    customers = rng.randint(20, 300, size=n)
    product_score = rng.uniform(3.0, 5.0, size=n)
    s = np.sin(np.linspace(0, 4 * np.pi, n))
    weekday = rng.randint(15, 23, size=n)
    holiday = rng.randint(0, 4, size=n)
    freight_price = rng.uniform(10.0, 30.0, size=n)

    # True price elasticity theta = -1.50
    log_p = np.log(unit_price)
    log_q = 5.0 - 1.50 * log_p + 0.005 * customers + 0.3 * s + rng.normal(0, 0.2, size=n)
    qty = np.maximum(1, np.round(np.exp(log_q))).astype(int)

    return pd.DataFrame({
        "unit_price": unit_price,
        "qty": qty,
        "comp_1": comp_1,
        "comp_2": comp_2,
        "customers": customers,
        "product_score": product_score,
        "s": s,
        "weekday": weekday,
        "holiday": holiday,
        "freight_price": freight_price,
    })


# -----------------------------------------------------------------------------
# 1. Placebo Treatment Refutation Tests
# -----------------------------------------------------------------------------

def test_refute_placebo_treatment_permutation(fitted_dml_model):
    """Test that replacing treatment with placebo permutation collapses causal effect to near zero."""
    dml, X, T, Y = fitted_dml_model
    refuter = CausalRefuter(dml, random_state=42)

    res = refuter.refute_placebo_treatment(X, T, Y, n_repeats=3, mode="permutation")
    assert res["passed"] is True
    assert abs(res["placebo_theta_mean"]) < abs(dml.theta_) * 0.30
    assert res["collapse_ratio"] < 0.30
    assert "Placebo Treatment Refutation" in res["test_name"]
    assert len(res["placebo_thetas"]) == 3


def test_refute_placebo_treatment_random_noise_mode(fitted_dml_model):
    """Test placebo refutation under synthetic random noise mode."""
    dml, X, T, Y = fitted_dml_model
    refuter = CausalRefuter(dml, random_state=42)

    res = refuter.refute_placebo_treatment(X, T, Y, n_repeats=3, mode="random_noise")
    assert res["passed"] is True
    assert res["mode"] == "random_noise"
    assert abs(res["placebo_theta_mean"]) < 0.35 or res["placebo_p_value_mean"] > 0.05
    assert res["collapse_ratio"] < 0.30


# -----------------------------------------------------------------------------
# 2. Random Common Cause Refutation Tests
# -----------------------------------------------------------------------------

def test_refute_random_common_cause_gaussian(fitted_dml_model):
    """Test that adding gaussian random noise confounder leaves treatment effect estimate stable."""
    dml, X, T, Y = fitted_dml_model
    refuter = CausalRefuter(dml, random_state=42)

    res = refuter.refute_random_common_cause(X, T, Y, n_repeats=3, noise_distribution="gaussian")
    assert res["passed"] is True
    assert res["relative_change"] < 0.20
    assert "Random Common Cause Refutation (gaussian)" in res["test_name"]


def test_refute_random_common_cause_uniform_and_correlated(fitted_dml_model):
    """Test random common cause refutation with uniform and correlated noise distributions."""
    dml, X, T, Y = fitted_dml_model
    refuter = CausalRefuter(dml, random_state=42)

    res_uni = refuter.refute_random_common_cause(X, T, Y, n_repeats=2, noise_distribution="uniform", num_noise_vars=2)
    assert res_uni["passed"] is True
    assert res_uni["num_noise_vars"] == 2

    res_corr = refuter.refute_random_common_cause(X, T, Y, n_repeats=2, noise_distribution="correlated")
    assert res_corr["passed"] is True
    assert res_corr["noise_distribution"] == "correlated"


# -----------------------------------------------------------------------------
# 3. Data Subsetting Refutation Tests
# -----------------------------------------------------------------------------

def test_refute_data_subset(fitted_dml_model):
    """Test that subsampling 80% of data preserves estimate direction and magnitude."""
    dml, X, T, Y = fitted_dml_model
    refuter = CausalRefuter(dml, random_state=42)

    res = refuter.refute_data_subset(X, T, Y, subset_fraction=0.80, n_repeats=3)
    assert res["passed"] is True
    assert res["subset_theta_mean"] < 0
    assert res["sign_preserved"] is True
    assert res["coefficient_of_variation"] < 0.35
    assert abs(res["subset_theta_mean"] - dml.theta_) < 0.35


def test_refute_data_subset_fraction_variation(fitted_dml_model):
    """Test data subsetting under 70% fraction and check CV bounds."""
    dml, X, T, Y = fitted_dml_model
    refuter = CausalRefuter(dml, random_state=42)

    res = refuter.refute_data_subset(X, T, Y, subset_fraction=0.70, n_repeats=3)
    assert res["passed"] is True
    assert res["subset_fraction"] == 0.70
    assert len(res["subset_thetas"]) == 3


# -----------------------------------------------------------------------------
# 4. Sensitivity to Plausible Unobserved Confounding Tests
# -----------------------------------------------------------------------------

def test_refute_unobserved_confounder_analytical_and_benchmarks(fitted_dml_model):
    """Test Cinelli-Hazlett robustness value computation and covariate benchmarking."""
    dml, X, T, Y = fitted_dml_model
    X_df = pd.DataFrame(X, columns=["comp_1", "customers", "product_score", "s"])
    refuter = CausalRefuter(dml, random_state=42)

    res = refuter.refute_unobserved_confounder(
        X_df, T, Y,
        benchmark_feature_names=["comp_1", "customers", "product_score"],
        simulate_stress_grid=True,
    )

    assert res["passed"] is True
    assert "Unobserved Confounding Sensitivity" in res["test_name"]
    # True theta = -2.0 on N=600 is very strong, RV should comfortably exceed 12%
    assert res["robustness_value_point"] > 0.12
    assert res["robustness_value_alpha"] > 0.05
    assert res["partial_r2_treatment_outcome"] > 0.0
    assert res["benchmark_multiplier"] > 0.85
    assert res["strongest_benchmark_covariate"] in ["comp_1", "customers", "product_score"]
    assert "gamma_0.20" in res["simulated_grid"]


def test_unobserved_confounding_low_signal_sensitivity():
    """Verify that a model with high uncertainty / weak effect triggers sensitivity warning."""
    np.random.seed(99)
    n = 100
    X = np.random.randn(n, 2)
    T = np.random.randn(n)
    # Weak effect with heavy noise
    Y = -0.05 * T + 2.0 * np.random.randn(n)

    dml = DoubleMachineLearning(model_y=Ridge(), model_t=Ridge(), n_folds=2, random_state=99)
    dml.fit(X, T, Y)

    refuter = CausalRefuter(dml, random_state=99)
    res = refuter.refute_unobserved_confounder(X, T, Y, threshold_rv=0.30, simulate_stress_grid=False)
    # Should fail high threshold
    assert res["robustness_value_point"] < 0.30
    assert res["passed"] is False


# -----------------------------------------------------------------------------
# 5. Alternative Reasonable Specifications Tests
# -----------------------------------------------------------------------------

def test_evaluate_specification_sensitivity(synthetic_retail_df):
    """Test specification curve analysis across learners and functional forms."""
    df = synthetic_retail_df
    X = df[["comp_1", "comp_2", "customers", "product_score", "s", "freight_price"]].values
    T = np.log(df["unit_price"].values)
    Y = np.log(df["qty"].values)

    dml = DoubleMachineLearning(model_y=Ridge(alpha=1.0), model_t=Ridge(alpha=1.0), n_folds=3, random_state=42)
    dml.fit(X, T, Y)

    refuter = CausalRefuter(dml, random_state=42)
    spec_res = refuter.evaluate_specification_sensitivity(
        df=df,
        treatment_col="unit_price",
        outcome_col="qty",
    )

    assert spec_res["passed"] is True
    assert spec_res["total_specifications"] == 6
    assert spec_res["sign_consistency_pct"] >= 80.0
    # Median elasticity should be clearly negative (~ -1.50)
    assert spec_res["elasticity_median"] < -0.50
    assert spec_res["coefficient_of_variation"] < 0.75
    assert len(spec_res["specifications"]) == 6


# -----------------------------------------------------------------------------
# 6. Epistemic Validity Assessment Tests
# -----------------------------------------------------------------------------

def test_assess_causal_validity_epistemics(fitted_dml_model):
    """Test epistemic framework distinguishing statistical corroboration from causal proof."""
    dml, X, T, Y = fitted_dml_model
    refuter = CausalRefuter(dml, random_state=42)

    # Run refutations first
    consolidated = refuter.run_all_refutations(X, T, Y, n_repeats=2)
    epistemics = refuter.assess_causal_validity_epistemics(consolidated)

    assert epistemics["statistical_tests_passed"] is True
    assert "STATISTICALLY CORROBORATED" in epistemics["epistemic_verdict"]
    assert "Popperian Modus Tollens" in epistemics["falsification_paradigm"]

    # Verify structural assumptions checklist
    checklist = epistemics["assumptions_checklist"]
    assert "conditional_unconfoundedness" in checklist
    assert "positivity_common_support" in checklist
    assert "sutva_no_interference" in checklist
    assert "absence_of_bad_controls" in checklist
    assert "no_simultaneity_reverse_causality" in checklist
    assert checklist["conditional_unconfoundedness"]["epistemic_status"] == "CORROBORATED_BUT_UNPROVEN"


# -----------------------------------------------------------------------------
# 7. Evidence-Based Governance Rules (ACCEPT, WARNING, REJECT)
# -----------------------------------------------------------------------------

def test_model_governance_decision_accept(fitted_dml_model):
    """Test that a model passing all refutations cleanly receives ACCEPT verdict."""
    dml, X, T, Y = fitted_dml_model
    refuter = CausalRefuter(dml, random_state=42)

    all_res = refuter.run_all_refutations(X, T, Y, n_repeats=2)
    r_unobs = refuter.refute_unobserved_confounder(X, T, Y, simulate_stress_grid=False)
    all_res["results"]["unobserved_confounding"] = r_unobs

    gov = refuter.evaluate_model_evidence(all_res)
    assert gov["decision"] == CausalDecision.ACCEPT.value
    assert gov["evidence_score_percent"] >= 85.0
    assert len(gov["fatal_flaws"]) == 0
    assert "FULL PRODUCTION APPROVAL" in gov["operational_action"]


def test_model_governance_decision_warning():
    """Test that a marginal non-fatal warning triggers WARNING verdict with guardrails."""
    dml = DoubleMachineLearning()
    dml._is_fitted = True
    dml.theta_ = -1.20
    dml.std_error_ = 0.30

    refuter = CausalRefuter(dml, random_state=42)
    simulated_results = {
        "placebo_treatment": {"passed": True},
        "random_common_cause": {"passed": True, "relative_change": 0.08},
        "data_subset": {"passed": False, "sign_preserved": True, "coefficient_of_variation": 0.42},
        "unobserved_confounding": {"passed": True, "robustness_value_point": 0.16, "benchmark_multiplier": 1.4},
    }

    gov = refuter.evaluate_model_evidence(simulated_results)
    assert gov["decision"] == CausalDecision.WARNING.value
    assert len(gov["warning_flags"]) >= 1
    assert "CONDITIONAL APPROVAL WITH GUARDRAILS" in gov["operational_action"]


def test_model_governance_decision_reject():
    """Test that a fatal flaw (e.g. placebo failure due to leakage) triggers REJECT verdict."""
    dml = DoubleMachineLearning()
    dml._is_fitted = True
    dml.theta_ = -1.50
    dml.std_error_ = 0.40

    refuter = CausalRefuter(dml, random_state=42)
    simulated_results = {
        "placebo_treatment": {"passed": False, "placebo_theta_mean": -1.45},
        "random_common_cause": {"passed": True, "relative_change": 0.05},
        "data_subset": {"passed": True, "sign_preserved": True},
    }

    gov = refuter.evaluate_model_evidence(simulated_results)
    assert gov["decision"] == CausalDecision.REJECT.value
    assert len(gov["fatal_flaws"]) >= 1
    assert "PROHIBIT PRODUCTION DEPLOYMENT" in gov["operational_action"]


# -----------------------------------------------------------------------------
# 8. Advanced Battery & Report Formatting Tests
# -----------------------------------------------------------------------------

def test_run_advanced_refutations_and_formatting(synthetic_retail_df):
    """Test complete advanced battery orchestration and ASCII/Markdown report formatting."""
    df = synthetic_retail_df
    X = df[["comp_1", "comp_2", "customers", "product_score", "s"]].values
    T = np.log(df["unit_price"].values)
    Y = np.log(df["qty"].values)

    dml = DoubleMachineLearning(model_y=Ridge(alpha=1.0), model_t=Ridge(alpha=1.0), n_folds=3, random_state=42)
    dml.fit(X, T, Y)

    refuter = CausalRefuter(dml, random_state=42)
    adv_res = refuter.run_advanced_refutations(
        X=X, T=T, Y=Y,
        df=df,
        treatment_col="unit_price",
        outcome_col="qty",
        n_repeats=2,
    )

    assert adv_res["total_tests"] == 5
    assert adv_res["passed_tests"] >= 4
    assert adv_res["governance_decision"] in [CausalDecision.ACCEPT.value, CausalDecision.WARNING.value]

    report = refuter.format_advanced_report(adv_res)
    assert "CAUSAL MODEL REFUTATION AUDIT REPORT" in report
    assert "EVIDENCE-BASED CAUSAL MODEL GOVERNANCE VERDICT" in report
    assert "FINAL GOVERNANCE DECISION:" in report
    assert "EPISTEMIC VALIDITY & IDENTIFICATION STATUS:" in report
    assert "Unobserved Confounding" in report
    assert "Alternative Specifications" in report


def test_refuter_unfitted_model_error():
    """Test that initializing refuter with unfitted model raises ValueError."""
    unfitted_dml = DoubleMachineLearning()
    with pytest.raises(ValueError, match="must be fitted"):
        CausalRefuter(unfitted_dml)


# -----------------------------------------------------------------------------
# 9. Real Dataset Integration Test
# -----------------------------------------------------------------------------

def test_advanced_refutation_on_real_retail_price_dataset():
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
        engineer_features=True,
    )

    X_train = splits["X_train"]
    T_train = splits["T_train"]
    Y_train = splits["Y_train"]

    # Production-grade DML estimator
    dml = DoubleMachineLearning(
        n_folds=3,
        random_state=42,
    )
    dml.fit(X_train, T_train, Y_train)

    refuter = CausalRefuter(dml, random_state=42)
    adv_res = refuter.run_advanced_refutations(
        X=X_train,
        T=T_train,
        Y=Y_train,
        df=df,
        treatment_col="unit_price",
        outcome_col="qty",
        n_repeats=2,
        run_spec_curve=True,
    )

    # Real retail dataset must pass at least 4 out of 5 tests and receive ACCEPT or WARNING
    assert adv_res["passed_tests"] >= 4
    assert adv_res["robustness_score_percent"] >= 80.0
    assert adv_res["governance_decision"] in [CausalDecision.ACCEPT.value, CausalDecision.WARNING.value]
    assert adv_res["results"]["placebo_treatment"]["passed"] is True
    assert adv_res["results"]["random_common_cause"]["passed"] is True
    assert adv_res["results"]["data_subset"]["passed"] is True
    assert adv_res["results"]["unobserved_confounding"]["passed"] is True
