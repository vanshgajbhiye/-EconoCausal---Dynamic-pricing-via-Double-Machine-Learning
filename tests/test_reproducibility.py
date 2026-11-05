"""Unit and Integration Tests for Causal Model Reproducibility and Lineage Tracking.

Milestone: Day 18 — Reproducibility and model performance.
"""

import json
import os
import tempfile
import numpy as np
import pandas as pd
import pytest

from src.data.load_data import generate_synthetic_retail_data
from src.models.artifacts import (
    FeatureSchema,
    ModelMetadata,
    TrainingDataVersion,
    check_version_compatibility,
    load_versioned_bundle,
    save_versioned_bundle,
)
from src.pipeline import EconoCausalPipeline
from src.utils.reproducibility import (
    compute_array_hash,
    compute_data_hash,
    get_dependency_versions,
    get_git_info,
    get_system_info,
    set_seed,
)


@pytest.fixture
def synthetic_retail_df():
    """Deterministic synthetic retail dataset for fast testing."""
    return generate_synthetic_retail_data(n_samples=120, random_seed=42)


def test_fix_random_seeds_identical_training(synthetic_retail_df):
    """Verify that fixing random seed produces numerically identical causal models and predictions."""
    df = synthetic_retail_df

    # Train Model 1 with seed 42
    p1 = EconoCausalPipeline(n_splits=3, random_state=42, model_version="1.0.0")
    p1.fit(df, test_size=0.25, run_refutations=False)

    # Train Model 2 with seed 42
    p2 = EconoCausalPipeline(n_splits=3, random_state=42, model_version="1.0.0")
    p2.fit(df, test_size=0.25, run_refutations=False)

    # Verify Baseline ATE is identical
    assert p1.baseline_ate_ is not None
    assert p2.baseline_ate_ is not None
    np.testing.assert_allclose(p1.baseline_ate_, p2.baseline_ate_, rtol=1e-7, atol=1e-7)
    np.testing.assert_allclose(p1.baseline_ate_se_, p2.baseline_ate_se_, rtol=1e-7, atol=1e-7)

    # Verify CATE parameters are identical
    np.testing.assert_allclose(p1.cate_estimator.beta_, p2.cate_estimator.beta_, rtol=1e-7, atol=1e-7)
    assert p1.cate_estimator.intercept_ == pytest.approx(p2.cate_estimator.intercept_, rel=1e-7)

    # Verify batch recommendations are identical
    rec1 = p1.batch_recommend(df.head(20))
    rec2 = p2.batch_recommend(df.head(20))

    np.testing.assert_allclose(rec1["elasticity"].values, rec2["elasticity"].values, rtol=1e-7, atol=1e-7)
    np.testing.assert_allclose(rec1["optimal_price"].values, rec2["optimal_price"].values, rtol=1e-7, atol=1e-7)
    assert list(rec1["pricing_action"]) == list(rec2["pricing_action"])


def test_different_seeds_produce_variation(synthetic_retail_df):
    """Verify that varying random seed produces distinct cross-fitting splits."""
    df = synthetic_retail_df

    p_seed42 = EconoCausalPipeline(n_splits=3, random_state=42)
    p_seed42.fit(df, test_size=0.25, run_refutations=False)

    p_seed99 = EconoCausalPipeline(n_splits=3, random_state=99)
    p_seed99.fit(df, test_size=0.25, run_refutations=False)

    # Different seeds should produce slightly different nuisance model splits and predictions
    assert p_seed42.baseline_ate_ != p_seed99.baseline_ate_


def test_record_dependency_versions():
    """Verify dependency versions capture required libraries and Python version."""
    deps = get_dependency_versions()

    assert "python" in deps
    assert "numpy" in deps
    assert "pandas" in deps
    assert "scikit-learn" in deps
    assert "joblib" in deps
    assert "scipy" in deps

    # Verify version strings are non-empty
    for pkg, ver in deps.items():
        assert isinstance(ver, str)
        assert len(ver) > 0


def test_system_and_git_info_recording():
    """Verify system profiling and git metadata extraction."""
    sys_info = get_system_info()
    assert "platform" in sys_info
    assert "cpu_count" in sys_info
    assert sys_info["cpu_count"] >= 1
    assert "python_version" in sys_info

    git_info = get_git_info()
    assert "commit_sha" in git_info
    assert "branch" in git_info
    assert "is_dirty" in git_info


def test_cryptographic_data_hashing(synthetic_retail_df):
    """Verify SHA-256 data hashing determinism and sensitivity."""
    df = synthetic_retail_df

    hash1 = compute_data_hash(df)
    hash2 = compute_data_hash(df)
    assert hash1 == hash2
    assert len(hash1) == 64

    # Mutate a single value -> hash must change
    df_modified = df.copy()
    if "unit_price" in df_modified.columns:
        df_modified.loc[0, "unit_price"] = float(df_modified.loc[0, "unit_price"]) + 1.0
    else:
        df_modified.iloc[0, 0] = str(df_modified.iloc[0, 0]) + "_modified"
    hash_mod = compute_data_hash(df_modified)
    assert hash1 != hash_mod

    # Array hash test
    arr_hash1 = compute_array_hash([1, 2, 3, 4, 5])
    arr_hash2 = compute_array_hash([1, 2, 3, 4, 5])
    assert arr_hash1 == arr_hash2


def test_feature_schema_capture(synthetic_retail_df):
    """Verify that fitting pipeline generates comprehensive FeatureSchema."""
    pipeline = EconoCausalPipeline(n_splits=3, random_state=42)
    pipeline.fit(synthetic_retail_df, test_size=0.20, run_refutations=False)

    schema = pipeline.feature_schema_
    assert schema is not None
    assert isinstance(schema, FeatureSchema)
    assert schema.treatment_col == "unit_price"
    assert schema.outcome_col == "qty"
    assert len(schema.raw_confounder_cols) > 0
    assert len(schema.encoded_feature_names) > 0
    assert len(schema.categorical_levels) > 0
    assert len(schema.imputation_values) > 0

    # Verify statistical distribution properties
    for col, stats in schema.feature_statistics.items():
        assert "min" in stats
        assert "max" in stats
        assert "mean" in stats
        assert "median" in stats
        assert stats["min"] <= stats["max"]


def test_training_data_versioning(synthetic_retail_df):
    """Verify that training dataset provenance and split indices hash are captured."""
    pipeline = EconoCausalPipeline(n_splits=3, random_state=42)
    pipeline.fit(synthetic_retail_df, test_size=0.25, run_refutations=False)

    data_ver = pipeline.training_data_version_
    assert data_ver is not None
    assert isinstance(data_ver, TrainingDataVersion)
    assert data_ver.n_records == len(synthetic_retail_df)
    assert data_ver.train_records > 0
    assert data_ver.val_records > 0
    assert len(data_ver.data_sha256) == 64
    assert len(data_ver.train_indices_sha256) == 64
    assert data_ver.test_size == 0.25


def test_versioned_model_bundle_save_and_load(synthetic_retail_df):
    """Verify saving and loading versioned model artifact bundle with manifest and checksums."""
    pipeline = EconoCausalPipeline(n_splits=3, random_state=42, model_version="1.2.3")
    pipeline.fit(synthetic_retail_df, test_size=0.20, run_refutations=False)

    with tempfile.TemporaryDirectory() as tmp_dir:
        bundle_dir = os.path.join(tmp_dir, "v1.2.3")
        saved_dir = pipeline.save_versioned(bundle_dir)

        assert os.path.isdir(saved_dir)
        assert os.path.exists(os.path.join(saved_dir, "manifest.json"))
        assert os.path.exists(os.path.join(saved_dir, "model.joblib"))
        assert os.path.exists(os.path.join(saved_dir, "metadata.json"))
        assert os.path.exists(os.path.join(saved_dir, "feature_schema.json"))
        assert os.path.exists(os.path.join(saved_dir, "training_data_version.json"))
        assert os.path.exists(os.path.join(saved_dir, "evaluation_results.json"))

        # Load bundle back
        loaded_p = EconoCausalPipeline.load_versioned(saved_dir, verify_checksums=True)
        assert loaded_p.is_fitted_ is True
        assert loaded_p.model_metadata_.model_version == "1.2.3"
        assert loaded_p.baseline_ate_ == pipeline.baseline_ate_

        # Verify predictions match between original and loaded pipeline
        test_item = {"unit_price": 45.0, "qty": 12.0, "product_category_name": "health_beauty"}
        rec_orig = pipeline.predict_item(test_item)
        rec_loaded = loaded_p.predict_item(test_item)

        assert rec_orig["optimal_price"] == rec_loaded["optimal_price"]
        assert rec_orig["causal_elasticity"] == rec_loaded["causal_elasticity"]


def test_bundle_tamper_detection(synthetic_retail_df):
    """Verify that tampering with an artifact file triggers checksum failure."""
    pipeline = EconoCausalPipeline(n_splits=3, random_state=42)
    pipeline.fit(synthetic_retail_df, test_size=0.20, run_refutations=False)

    with tempfile.TemporaryDirectory() as tmp_dir:
        bundle_dir = os.path.join(tmp_dir, "v1.0.0")
        pipeline.save_versioned(bundle_dir)

        # Tamper with metadata.json
        meta_path = os.path.join(bundle_dir, "metadata.json")
        with open(meta_path, "a", encoding="utf-8") as f:
            f.write("\n// tampered")

        # Loading with verify_checksums=True must raise ValueError
        with pytest.raises(ValueError, match="Integrity check failed"):
            load_versioned_bundle(bundle_dir, verify_checksums=True)


def test_version_compatibility_auditing(synthetic_retail_df):
    """Verify environment compatibility checks for saved model metadata."""
    pipeline = EconoCausalPipeline(n_splits=3, random_state=42)
    pipeline.fit(synthetic_retail_df, test_size=0.20, run_refutations=False)

    compat = pipeline.check_version_compatibility()
    assert compat["status"] in ["COMPATIBLE", "WARNING"]
    assert compat["is_compatible"] is True

    # Test simulated major version incompatibility
    fake_meta = {
        "model_version": "0.1.0",
        "dependencies": {"python": "2.7.18", "scikit-learn": "0.20.0"},
        "schema_version": "99.0.0",
    }
    compat_fake = check_version_compatibility(fake_meta)
    assert compat_fake["status"] == "INCOMPATIBLE"
    assert len(compat_fake["errors"]) > 0
