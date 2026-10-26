"""EconoCausal Continuous Causal MLOps and Drift Monitoring Package.

Provides confounder distribution drift detection (PSI, KS test, Wasserstein distance),
heterogeneous elasticity (CATE) shift tracking, automated causal refutation re-auditing,
and automated retraining trigger decision logic.
"""

from src.monitoring.drift import (
    CausalDriftDetector,
    CausalDriftReport,
    FeatureDriftResult,
    calculate_population_stability_index,
)

__all__ = [
    "CausalDriftDetector",
    "CausalDriftReport",
    "FeatureDriftResult",
    "calculate_population_stability_index",
]
