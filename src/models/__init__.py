"""Models module package for EconoCausal."""

from src.models.artifacts import (
    EvaluationResults,
    FeatureSchema,
    ModelMetadata,
    TrainingDataVersion,
    check_version_compatibility,
    extract_feature_schema,
    extract_training_data_version,
    load_versioned_bundle,
    save_versioned_bundle,
)
from src.models.benchmark import (
    BatchBenchmarkMetrics,
    ModelBenchmarkSummary,
    ReproducibilityMetrics,
    SingleItemBenchmarkMetrics,
    run_inference_benchmarks,
)
from src.models.cate import CATEEstimator
from src.models.evaluator import EvaluationAuditReport, ModelEvaluator
from src.models.optimizer import (
    DynamicPricingOptimizer,
    ItemOptimizationResult,
    ProductConstraints,
)
from src.models.refutation import CausalRefuter
from src.models.train_dml import DoubleMachineLearning, train_dml_pipeline
from src.utils.reproducibility import (
    compute_array_hash,
    compute_data_hash,
    get_dependency_versions,
    get_git_info,
    get_system_info,
    set_seed,
)

__all__ = [
    "DoubleMachineLearning",
    "train_dml_pipeline",
    "CATEEstimator",
    "CausalRefuter",
    "DynamicPricingOptimizer",
    "ItemOptimizationResult",
    "ProductConstraints",
    "FeatureSchema",
    "TrainingDataVersion",
    "EvaluationResults",
    "ModelMetadata",
    "save_versioned_bundle",
    "load_versioned_bundle",
    "check_version_compatibility",
    "extract_feature_schema",
    "extract_training_data_version",
    "BatchBenchmarkMetrics",
    "ModelBenchmarkSummary",
    "ReproducibilityMetrics",
    "SingleItemBenchmarkMetrics",
    "run_inference_benchmarks",
    "EvaluationAuditReport",
    "ModelEvaluator",
    "compute_array_hash",
    "compute_data_hash",
    "get_dependency_versions",
    "get_git_info",
    "get_system_info",
    "set_seed",
]
