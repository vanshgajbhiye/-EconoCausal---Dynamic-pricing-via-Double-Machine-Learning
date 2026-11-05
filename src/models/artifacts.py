"""Model Artifact Versioning, Metadata Tracking, and Schema Persistence.

Provides formal data structures and serialization protocols for versioned causal
dynamic pricing model artifacts, schemas, data provenance hashes, and diagnostic audits.
"""

from dataclasses import asdict, dataclass, field
import datetime
import json
import os
import uuid
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union
import joblib
import numpy as np
import pandas as pd

from src.utils.reproducibility import (
    compute_array_hash,
    compute_data_hash,
    get_dependency_versions,
    get_git_info,
    get_system_info,
)


MODEL_SCHEMA_VERSION = "1.0.0"
DEFAULT_MODEL_VERSION = "1.0.0"


@dataclass
class FeatureSchema:
    """Rigorous schema specification of training features and imputation baselines."""

    treatment_col: str
    outcome_col: str
    raw_confounder_cols: List[str]
    raw_numerical_cols: List[str]
    raw_categorical_cols: List[str]
    encoded_feature_names: List[str]
    column_dtypes: Dict[str, str]
    categorical_levels: Dict[str, List[str]]
    imputation_values: Dict[str, Any]
    feature_statistics: Dict[str, Dict[str, float]]
    schema_version: str = MODEL_SCHEMA_VERSION

    def to_dict(self) -> Dict[str, Any]:
        """Convert feature schema to dictionary."""
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "FeatureSchema":
        """Reconstruct feature schema from dictionary."""
        return cls(
            treatment_col=data["treatment_col"],
            outcome_col=data["outcome_col"],
            raw_confounder_cols=list(data.get("raw_confounder_cols", [])),
            raw_numerical_cols=list(data.get("raw_numerical_cols", [])),
            raw_categorical_cols=list(data.get("raw_categorical_cols", [])),
            encoded_feature_names=list(data.get("encoded_feature_names", [])),
            column_dtypes=dict(data.get("column_dtypes", {})),
            categorical_levels=dict(data.get("categorical_levels", {})),
            imputation_values=dict(data.get("imputation_values", {})),
            feature_statistics=dict(data.get("feature_statistics", {})),
            schema_version=data.get("schema_version", MODEL_SCHEMA_VERSION),
        )


@dataclass
class TrainingDataVersion:
    """Cryptographic provenance and split tracking for training datasets."""

    data_sha256: str
    n_records: int
    n_columns: int
    columns: List[str]
    train_records: int
    val_records: int
    train_indices_sha256: str
    test_size: float
    split_random_state: int
    dataset_path: Optional[str] = None
    recorded_at: str = field(
        default_factory=lambda: datetime.datetime.now(datetime.timezone.utc).isoformat()
    )

    def to_dict(self) -> Dict[str, Any]:
        """Convert training data provenance to dictionary."""
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "TrainingDataVersion":
        """Reconstruct training data provenance from dictionary."""
        return cls(
            data_sha256=data["data_sha256"],
            n_records=int(data["n_records"]),
            n_columns=int(data["n_columns"]),
            columns=list(data.get("columns", [])),
            train_records=int(data.get("train_records", 0)),
            val_records=int(data.get("val_records", 0)),
            train_indices_sha256=data.get("train_indices_sha256", ""),
            test_size=float(data.get("test_size", 0.2)),
            split_random_state=int(data.get("split_random_state", 42)),
            dataset_path=data.get("dataset_path"),
            recorded_at=data.get("recorded_at", ""),
        )


@dataclass
class EvaluationResults:
    """Evaluation metrics, confidence intervals, and refutation robustness scores."""

    baseline_ate: float
    baseline_ate_se: float
    baseline_ate_pvalue: float
    baseline_ate_ci: Tuple[float, float]
    cate_summary: Dict[str, float]
    nuisance_scores: Dict[str, float]
    refutation_pass_rate: Optional[float] = None
    refutation_results: Optional[Dict[str, Any]] = None
    recorded_at: str = field(
        default_factory=lambda: datetime.datetime.now(datetime.timezone.utc).isoformat()
    )

    def to_dict(self) -> Dict[str, Any]:
        """Convert evaluation results to dictionary."""
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "EvaluationResults":
        """Reconstruct evaluation results from dictionary."""
        ci_raw = data.get("baseline_ate_ci", [0.0, 0.0])
        ci_tuple = (float(ci_raw[0]), float(ci_raw[1])) if len(ci_raw) == 2 else (0.0, 0.0)
        return cls(
            baseline_ate=float(data["baseline_ate"]),
            baseline_ate_se=float(data.get("baseline_ate_se", 0.0)),
            baseline_ate_pvalue=float(data.get("baseline_ate_pvalue", 0.0)),
            baseline_ate_ci=ci_tuple,
            cate_summary=dict(data.get("cate_summary", {})),
            nuisance_scores=dict(data.get("nuisance_scores", {})),
            refutation_pass_rate=data.get("refutation_pass_rate"),
            refutation_results=data.get("refutation_results"),
            recorded_at=data.get("recorded_at", ""),
        )


@dataclass
class ModelMetadata:
    """Comprehensive top-level metadata and lineage container for trained models."""

    model_name: str
    model_version: str
    model_id: str
    created_at: str
    training_duration_sec: float
    random_state: int
    algorithm: str
    hyperparameters: Dict[str, Any]
    dependencies: Dict[str, str]
    system_info: Dict[str, Any]
    git_info: Dict[str, Any]
    schema_version: str = MODEL_SCHEMA_VERSION

    def to_dict(self) -> Dict[str, Any]:
        """Convert metadata to dictionary."""
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ModelMetadata":
        """Reconstruct metadata from dictionary."""
        return cls(
            model_name=data.get("model_name", "EconoCausalPipeline"),
            model_version=data.get("model_version", DEFAULT_MODEL_VERSION),
            model_id=data.get("model_id", str(uuid.uuid4())),
            created_at=data.get("created_at", ""),
            training_duration_sec=float(data.get("training_duration_sec", 0.0)),
            random_state=int(data.get("random_state", 42)),
            algorithm=data.get("algorithm", "PartiallyLinearDML_CATE"),
            hyperparameters=dict(data.get("hyperparameters", {})),
            dependencies=dict(data.get("dependencies", {})),
            system_info=dict(data.get("system_info", {})),
            git_info=dict(data.get("git_info", {})),
            schema_version=data.get("schema_version", MODEL_SCHEMA_VERSION),
        )


def extract_feature_schema(
    df: pd.DataFrame,
    treatment_col: str,
    outcome_col: str,
    raw_confounder_cols: List[str],
    raw_numerical_cols: List[str],
    raw_categorical_cols: List[str],
    encoded_feature_names: List[str],
    imputation_values: Dict[str, Any],
) -> FeatureSchema:
    """Build a rigorous FeatureSchema from training data and preprocessor specifications."""
    column_dtypes: Dict[str, str] = {}
    for col in [treatment_col, outcome_col] + raw_confounder_cols:
        if col in df.columns:
            column_dtypes[col] = str(df[col].dtype)

    categorical_levels: Dict[str, List[str]] = {}
    for col in raw_categorical_cols:
        if col in df.columns:
            cats = [str(x) for x in df[col].dropna().unique().tolist()]
            categorical_levels[col] = sorted(cats)

    feature_statistics: Dict[str, Dict[str, float]] = {}
    for col in raw_numerical_cols:
        if col in df.columns:
            s = df[col].dropna()
            if len(s) > 0:
                feature_statistics[col] = {
                    "min": float(s.min()),
                    "max": float(s.max()),
                    "mean": float(s.mean()),
                    "std": float(s.std()) if len(s) > 1 else 0.0,
                    "p25": float(s.quantile(0.25)),
                    "median": float(s.median()),
                    "p75": float(s.quantile(0.75)),
                }

    return FeatureSchema(
        treatment_col=treatment_col,
        outcome_col=outcome_col,
        raw_confounder_cols=list(raw_confounder_cols),
        raw_numerical_cols=list(raw_numerical_cols),
        raw_categorical_cols=list(raw_categorical_cols),
        encoded_feature_names=list(encoded_feature_names),
        column_dtypes=column_dtypes,
        categorical_levels=categorical_levels,
        imputation_values=imputation_values,
        feature_statistics=feature_statistics,
    )


def extract_training_data_version(
    df: pd.DataFrame,
    train_idx: Sequence[Any],
    test_idx: Sequence[Any],
    test_size: float,
    random_state: int,
    dataset_path: Optional[str] = None,
) -> TrainingDataVersion:
    """Build cryptographic dataset provenance and partition checksum."""
    data_sha = compute_data_hash(df)
    train_indices_sha = compute_array_hash(train_idx)

    return TrainingDataVersion(
        data_sha256=data_sha,
        n_records=len(df),
        n_columns=len(df.columns),
        columns=list(df.columns),
        train_records=len(train_idx),
        val_records=len(test_idx),
        train_indices_sha256=train_indices_sha,
        test_size=test_size,
        split_random_state=random_state,
        dataset_path=dataset_path,
    )


def check_version_compatibility(
    metadata: Union[ModelMetadata, Dict[str, Any]],
) -> Dict[str, Any]:
    """Verify runtime environment compatibility with saved model metadata.

    Parameters
    ----------
    metadata : Union[ModelMetadata, Dict[str, Any]]
        Persisted model metadata object or dictionary.

    Returns
    -------
    Dict[str, Any]
        Structured report indicating compatibility status ('COMPATIBLE', 'WARNING', 'INCOMPATIBLE'),
        along with detailed diffs on Python and library versions.
    """
    meta_dict = metadata.to_dict() if isinstance(metadata, ModelMetadata) else metadata
    current_deps = get_dependency_versions()
    target_deps = meta_dict.get("dependencies", {})

    warnings_list: List[str] = []
    errors_list: List[str] = []

    # Check Python Major.Minor version
    current_py = current_deps.get("python", "").split(".")
    target_py = target_deps.get("python", "").split(".")
    if len(current_py) >= 2 and len(target_py) >= 2:
        if current_py[0] != target_py[0]:
            errors_list.append(
                f"Major Python version mismatch: saved under {target_deps.get('python')}, "
                f"running under {current_deps.get('python')}"
            )
        elif current_py[1] != target_py[1]:
            warnings_list.append(
                f"Minor Python version discrepancy: saved under {target_deps.get('python')}, "
                f"running under {current_deps.get('python')}"
            )

    # Check key ML dependencies (sklearn, joblib, numpy)
    for pkg in ["scikit-learn", "joblib", "numpy", "pandas"]:
        t_ver = target_deps.get(pkg)
        c_ver = current_deps.get(pkg)
        if t_ver and c_ver and t_ver != "not_installed" and c_ver != "not_installed":
            t_parts = t_ver.split(".")
            c_parts = c_ver.split(".")
            if len(t_parts) >= 1 and len(c_parts) >= 1:
                if t_parts[0] != c_parts[0]:
                    warnings_list.append(
                        f"Major version difference for {pkg}: saved={t_ver}, current={c_ver}"
                    )

    # Check model schema version
    model_schema = meta_dict.get("schema_version", "1.0.0")
    if model_schema.split(".")[0] != MODEL_SCHEMA_VERSION.split(".")[0]:
        errors_list.append(
            f"Incompatible schema version {model_schema}. Expected {MODEL_SCHEMA_VERSION}"
        )

    if errors_list:
        status = "INCOMPATIBLE"
    elif warnings_list:
        status = "WARNING"
    else:
        status = "COMPATIBLE"

    return {
        "status": status,
        "is_compatible": status in ["COMPATIBLE", "WARNING"],
        "warnings": warnings_list,
        "errors": errors_list,
        "model_version": meta_dict.get("model_version", "unknown"),
        "model_id": meta_dict.get("model_id", "unknown"),
        "target_dependencies": target_deps,
        "current_dependencies": current_deps,
    }


def save_versioned_bundle(
    pipeline: Any,
    target_dir: str,
    version: str = DEFAULT_MODEL_VERSION,
) -> str:
    """Persist complete versioned model artifact bundle.

    Creates structured bundle layout:
        target_dir/
            manifest.json
            model.joblib
            metadata.json
            feature_schema.json
            training_data_version.json
            evaluation_results.json

    Parameters
    ----------
    pipeline : EconoCausalPipeline
        Fitted pipeline instance.
    target_dir : str
        Directory where artifact files will be persisted.
    version : str, default='1.0.0'
        Semantic version tag.

    Returns
    -------
    str
        Absolute path to the created artifact directory.
    """
    abs_dir = os.path.abspath(target_dir)
    os.makedirs(abs_dir, exist_ok=True)

    # 1. Save core joblib model
    model_path = os.path.join(abs_dir, "model.joblib")
    joblib.dump(pipeline, model_path, compress=3)

    # 2. Extract and save metadata
    metadata_obj = getattr(pipeline, "model_metadata_", None)
    if metadata_obj is None or isinstance(metadata_obj, dict):
        # Build metadata if not structured dataclass
        metadata_dict = getattr(pipeline, "metadata_", {})
        metadata_obj = ModelMetadata(
            model_name="EconoCausalPipeline",
            model_version=version,
            model_id=f"ec-{uuid.uuid4().hex[:12]}",
            created_at=datetime.datetime.now(datetime.timezone.utc).isoformat(),
            training_duration_sec=float(metadata_dict.get("training_duration_sec", 0.0)),
            random_state=getattr(pipeline, "random_state", 42),
            algorithm="PartiallyLinearDML_CATE_AmorosoRobinson",
            hyperparameters={
                "n_splits": getattr(pipeline, "n_splits", 5),
                "cate_alpha": getattr(pipeline, "cate_alpha", 1.0),
                "default_margin": getattr(pipeline, "default_margin", 0.30),
                "min_price_change_pct": getattr(pipeline, "min_price_change_pct", -0.30),
                "max_price_change_pct": getattr(pipeline, "max_price_change_pct", 0.20),
                "min_margin": getattr(pipeline, "min_margin", 0.05),
                "charm_pricing": getattr(pipeline, "charm_pricing", True),
                "log_transform_treatment": getattr(pipeline, "log_transform_treatment", True),
                "log_transform_outcome": getattr(pipeline, "log_transform_outcome", True),
            },
            dependencies=get_dependency_versions(),
            system_info=get_system_info(),
            git_info=get_git_info(),
        )

    metadata_path = os.path.join(abs_dir, "metadata.json")
    with open(metadata_path, "w", encoding="utf-8") as f:
        json.dump(metadata_obj.to_dict(), f, indent=2)

    # 3. Save feature schema
    schema_path = os.path.join(abs_dir, "feature_schema.json")
    schema_obj = getattr(pipeline, "feature_schema_", None)
    schema_dict = schema_obj.to_dict() if hasattr(schema_obj, "to_dict") else (schema_obj or {})
    with open(schema_path, "w", encoding="utf-8") as f:
        json.dump(schema_dict, f, indent=2)

    # 4. Save training data version
    data_ver_path = os.path.join(abs_dir, "training_data_version.json")
    data_ver_obj = getattr(pipeline, "training_data_version_", None)
    data_ver_dict = data_ver_obj.to_dict() if hasattr(data_ver_obj, "to_dict") else (data_ver_obj or {})
    with open(data_ver_path, "w", encoding="utf-8") as f:
        json.dump(data_ver_dict, f, indent=2)

    # 5. Save evaluation results
    eval_path = os.path.join(abs_dir, "evaluation_results.json")
    eval_obj = getattr(pipeline, "evaluation_results_", None)
    eval_dict = eval_obj.to_dict() if hasattr(eval_obj, "to_dict") else (eval_obj or {})
    with open(eval_path, "w", encoding="utf-8") as f:
        json.dump(eval_dict, f, indent=2)

    # 6. Calculate checksums and save manifest
    files_to_hash = [
        "model.joblib",
        "metadata.json",
        "feature_schema.json",
        "training_data_version.json",
        "evaluation_results.json",
    ]
    manifest_files: Dict[str, Dict[str, Any]] = {}
    for fname in files_to_hash:
        fpath = os.path.join(abs_dir, fname)
        if os.path.exists(fpath):
            manifest_files[fname] = {
                "sha256": compute_data_hash(fpath),
                "size_bytes": os.path.getsize(fpath),
            }

    manifest = {
        "manifest_version": "1.0.0",
        "model_version": version,
        "model_id": metadata_obj.model_id if hasattr(metadata_obj, "model_id") else "unknown",
        "created_at": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "files": manifest_files,
    }

    manifest_path = os.path.join(abs_dir, "manifest.json")
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)

    return abs_dir


def load_versioned_bundle(
    target_dir: str,
    verify_checksums: bool = True,
    strict_compatibility: bool = False,
) -> Any:
    """Load a versioned model bundle, verify cryptographic checksums, and check compatibility.

    Parameters
    ----------
    target_dir : str
        Directory containing the versioned bundle files.
    verify_checksums : bool, default=True
        Whether to verify SHA-256 hashes against manifest.json.
    strict_compatibility : bool, default=False
        Whether to raise an exception if compatibility status is 'WARNING' or 'INCOMPATIBLE'.

    Returns
    -------
    EconoCausalPipeline
        Restored pipeline with metadata attributes populated.
    """
    abs_dir = os.path.abspath(target_dir)
    if not os.path.isdir(abs_dir):
        raise FileNotFoundError(f"Versioned bundle directory not found at: {abs_dir}")

    # 1. Verify Manifest and Checksums if available
    manifest_path = os.path.join(abs_dir, "manifest.json")
    if os.path.exists(manifest_path) and verify_checksums:
        with open(manifest_path, "r", encoding="utf-8") as f:
            manifest = json.load(f)
        for fname, file_info in manifest.get("files", {}).items():
            fpath = os.path.join(abs_dir, fname)
            if not os.path.exists(fpath):
                raise FileNotFoundError(f"Bundle file missing according to manifest: {fpath}")
            current_sha = compute_data_hash(fpath)
            expected_sha = file_info.get("sha256")
            if current_sha != expected_sha:
                raise ValueError(
                    f"Integrity check failed for {fname}! Expected SHA: {expected_sha}, got {current_sha}"
                )

    # 2. Check metadata compatibility
    metadata_path = os.path.join(abs_dir, "metadata.json")
    metadata_dict = {}
    if os.path.exists(metadata_path):
        with open(metadata_path, "r", encoding="utf-8") as f:
            metadata_dict = json.load(f)
        compat = check_version_compatibility(metadata_dict)
        if strict_compatibility and not compat["is_compatible"]:
            raise RuntimeError(
                f"Model bundle is incompatible with current environment: {compat['errors']}"
            )

    # 3. Load Joblib Model
    model_path = os.path.join(abs_dir, "model.joblib")
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Core model.joblib not found in bundle at: {model_path}")

    pipeline = joblib.load(model_path)

    # 4. Attach structured companion objects if available
    if metadata_dict:
        pipeline.model_metadata_ = ModelMetadata.from_dict(metadata_dict)
        pipeline.metadata_.update(metadata_dict)

    schema_path = os.path.join(abs_dir, "feature_schema.json")
    if os.path.exists(schema_path):
        with open(schema_path, "r", encoding="utf-8") as f:
            pipeline.feature_schema_ = FeatureSchema.from_dict(json.load(f))

    data_ver_path = os.path.join(abs_dir, "training_data_version.json")
    if os.path.exists(data_ver_path):
        with open(data_ver_path, "r", encoding="utf-8") as f:
            pipeline.training_data_version_ = TrainingDataVersion.from_dict(json.load(f))

    eval_path = os.path.join(abs_dir, "evaluation_results.json")
    if os.path.exists(eval_path):
        with open(eval_path, "r", encoding="utf-8") as f:
            pipeline.evaluation_results_ = EvaluationResults.from_dict(json.load(f))

    return pipeline
