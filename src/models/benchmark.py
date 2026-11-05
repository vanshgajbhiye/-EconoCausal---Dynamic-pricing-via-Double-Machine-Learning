"""Inference Performance, Latency, and Memory Benchmarking Engine.

Measures single-item and representative batch inference latency distributions
(mean, min, max, P50, P90, P95, P99), throughput, and memory consumption (peak & delta via tracemalloc).
Validates inference determinism and reproducibility across repeated calls.
"""

from dataclasses import asdict, dataclass, field
import datetime
import json
import os
import time
import tracemalloc
from typing import Any, Dict, List, Optional, Sequence, Tuple
import numpy as np
import pandas as pd

from src.utils.reproducibility import get_system_info


@dataclass
class SingleItemBenchmarkMetrics:
    """Latency distribution and memory metrics for single-SKU real-time inference."""

    iterations: int
    mean_ms: float
    std_ms: float
    min_ms: float
    max_ms: float
    p50_ms: float
    p90_ms: float
    p95_ms: float
    p99_ms: float
    throughput_items_per_sec: float
    memory_peak_mb: float
    memory_delta_mb: float

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class BatchBenchmarkMetrics:
    """Latency and memory profile for a specific batch size."""

    batch_size: int
    iterations: int
    mean_batch_latency_ms: float
    mean_per_item_latency_ms: float
    throughput_items_per_sec: float
    memory_peak_mb: float
    memory_delta_mb: float

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class ReproducibilityMetrics:
    """Numerical reproducibility and determinism verification."""

    is_reproducible: bool
    max_elasticity_abs_diff: float
    max_optimal_price_abs_diff: float
    identical_actions: bool
    sample_size: int

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class ModelBenchmarkSummary:
    """Comprehensive benchmark results container."""

    single_item: SingleItemBenchmarkMetrics
    batches: List[BatchBenchmarkMetrics]
    reproducibility: ReproducibilityMetrics
    model_version: str
    model_id: str
    system_info: Dict[str, Any]
    timestamp: str = field(
        default_factory=lambda: datetime.datetime.now(datetime.timezone.utc).isoformat()
    )

    def to_dict(self) -> Dict[str, Any]:
        return {
            "timestamp": self.timestamp,
            "model_version": self.model_version,
            "model_id": self.model_id,
            "single_item": self.single_item.to_dict(),
            "batches": [b.to_dict() for b in self.batches],
            "reproducibility": self.reproducibility.to_dict(),
            "system_info": self.system_info,
        }

    def save_json(self, filepath: str) -> None:
        """Export benchmark summary to JSON file."""
        abs_path = os.path.abspath(filepath)
        os.makedirs(os.path.dirname(abs_path), exist_ok=True)
        with open(abs_path, "w", encoding="utf-8") as f:
            json.dump(self.to_dict(), f, indent=2)

    def format_report(self) -> str:
        """Render readable ASCII audit table of performance metrics."""
        lines = [
            "=" * 78,
            "          ECONOCAUSAL INFERENCE LATENCY & MEMORY BENCHMARK REPORT",
            "=" * 78,
            f"Timestamp:       {self.timestamp}",
            f"Model Version:   {self.model_version} (ID: {self.model_id})",
            f"Platform:        {self.system_info.get('platform', 'unknown')} | CPUs: {self.system_info.get('cpu_count', 1)}",
            "-" * 78,
            "1. SINGLE-ITEM REAL-TIME INFERENCE (predict_item):",
            f"   - Iterations:         {self.single_item.iterations}",
            f"   - Mean Latency:       {self.single_item.mean_ms:.3f} ms (±{self.single_item.std_ms:.3f} ms)",
            f"   - Median (P50):       {self.single_item.p50_ms:.3f} ms",
            f"   - P90 / P95 / P99:    {self.single_item.p90_ms:.3f} ms / {self.single_item.p95_ms:.3f} ms / {self.single_item.p99_ms:.3f} ms",
            f"   - Min / Max:          {self.single_item.min_ms:.3f} ms / {self.single_item.max_ms:.3f} ms",
            f"   - Throughput:         {self.single_item.throughput_items_per_sec:.1f} items/sec",
            f"   - Memory Peak / Delta:{self.single_item.memory_peak_mb:.3f} MiB / {self.single_item.memory_delta_mb:+.3f} MiB",
            "-" * 78,
            "2. REPRESENTATIVE BATCH INFERENCE (batch_recommend):",
            f"   {'Batch Size':<12} {'Batch (ms)':<14} {'Per Item (ms)':<16} {'Throughput (it/s)':<20} {'Peak RAM (MiB)':<14}",
            f"   {'-'*10:<12} {'-'*12:<14} {'-'*14:<16} {'-'*18:<20} {'-'*12:<14}",
        ]

        for b in self.batches:
            lines.append(
                f"   {b.batch_size:<12} {b.mean_batch_latency_ms:<14.2f} "
                f"{b.mean_per_item_latency_ms:<16.3f} {b.throughput_items_per_sec:<20.1f} "
                f"{b.memory_peak_mb:<14.3f}"
            )

        lines.extend([
            "-" * 78,
            "3. INFERENCE REPRODUCIBILITY & DETERMINISM AUDIT:",
            f"   - Determinism Status: {'PASSED (Deterministic)' if self.reproducibility.is_reproducible else 'FAILED (Non-deterministic)'}",
            f"   - Max Elasticity Diff:{self.reproducibility.max_elasticity_abs_diff:.2e}",
            f"   - Max Price Diff:     {self.reproducibility.max_optimal_price_abs_diff:.2e}",
            f"   - Identical Actions:  {'100%' if self.reproducibility.identical_actions else 'MISMATCH'}",
            "=" * 78,
        ])

        return "\n".join(lines)


def run_inference_benchmarks(
    pipeline: Any,
    df: pd.DataFrame,
    batch_sizes: Sequence[int] = (1, 10, 50, 100),
    n_single_runs: int = 100,
    measure_memory: bool = True,
) -> ModelBenchmarkSummary:
    """Execute end-to-end latency, throughput, and memory benchmarking on fitted pipeline.

    Parameters
    ----------
    pipeline : EconoCausalPipeline
        Fitted pipeline instance.
    df : pd.DataFrame
        Representative catalog or transaction dataset.
    batch_sizes : Sequence[int], default=(1, 10, 50, 100)
        Representative batch sizes to benchmark.
    n_single_runs : int, default=100
        Number of single-SKU inference iterations for latency percentile calculation.
    measure_memory : bool, default=True
        Whether to profile memory allocation using tracemalloc.

    Returns
    -------
    ModelBenchmarkSummary
        Comprehensive benchmark report container.
    """
    if not getattr(pipeline, "is_fitted_", False):
        raise ValueError("Pipeline must be fitted before running inference benchmarks.")

    if len(df) == 0:
        raise ValueError("Cannot benchmark on empty DataFrame.")

    # 1. Warmup execution
    warmup_item = {
        "unit_price": 50.0,
        "qty": 10.0,
        "product_category_name": str(df["product_category_name"].iloc[0])
        if "product_category_name" in df.columns
        else "health_beauty",
    }
    for _ in range(5):
        pipeline.predict_item(warmup_item)

    # 2. Single-SKU Latency Benchmarking
    sample_items = []
    n_samples_available = len(df)
    for i in range(min(n_single_runs, n_samples_available)):
        row = df.iloc[i].to_dict()
        sample_items.append(row)

    latencies_ms: List[float] = []

    # Memory measurement for single-item
    if measure_memory:
        tracemalloc.start()
        snap_before = tracemalloc.take_snapshot()

    for i in range(n_single_runs):
        item = sample_items[i % len(sample_items)]
        t0 = time.perf_counter()
        pipeline.predict_item(item, objective="profit")
        t1 = time.perf_counter()
        latencies_ms.append((t1 - t0) * 1000.0)

    if measure_memory:
        snap_after = tracemalloc.take_snapshot()
        current_mem, peak_mem = tracemalloc.get_traced_memory()
        tracemalloc.stop()
        single_peak_mb = peak_mem / (1024.0 * 1024.0)
        single_delta_mb = (current_mem) / (1024.0 * 1024.0)
    else:
        single_peak_mb = 0.0
        single_delta_mb = 0.0

    lat_arr = np.array(latencies_ms)
    total_time_sec = np.sum(lat_arr) / 1000.0
    throughput = (n_single_runs / total_time_sec) if total_time_sec > 0 else 0.0

    single_metrics = SingleItemBenchmarkMetrics(
        iterations=n_single_runs,
        mean_ms=float(np.mean(lat_arr)),
        std_ms=float(np.std(lat_arr)),
        min_ms=float(np.min(lat_arr)),
        max_ms=float(np.max(lat_arr)),
        p50_ms=float(np.percentile(lat_arr, 50)),
        p90_ms=float(np.percentile(lat_arr, 90)),
        p95_ms=float(np.percentile(lat_arr, 95)),
        p99_ms=float(np.percentile(lat_arr, 99)),
        throughput_items_per_sec=throughput,
        memory_peak_mb=single_peak_mb,
        memory_delta_mb=single_delta_mb,
    )

    # 3. Representative Batch Inference Benchmarking
    batch_metrics_list: List[BatchBenchmarkMetrics] = []

    # Include full catalog if not already in batch_sizes
    valid_batch_sizes = sorted(list(set([b for b in batch_sizes if b > 0])))
    if len(df) not in valid_batch_sizes and len(df) <= 1000:
        valid_batch_sizes.append(len(df))

    for b_size in valid_batch_sizes:
        # Construct batch slice (looping around df if b_size > len(df))
        if b_size <= len(df):
            batch_slice = df.iloc[:b_size].copy()
        else:
            repeats = int(np.ceil(b_size / len(df)))
            batch_slice = pd.concat([df] * repeats, ignore_index=True).iloc[:b_size]

        n_repeats = 3 if b_size >= 100 else 5
        batch_latencies: List[float] = []

        if measure_memory:
            tracemalloc.start()

        for _ in range(n_repeats):
            t0 = time.perf_counter()
            pipeline.batch_recommend(batch_slice, objective="profit")
            t1 = time.perf_counter()
            batch_latencies.append((t1 - t0) * 1000.0)

        if measure_memory:
            current_mem, peak_mem = tracemalloc.get_traced_memory()
            tracemalloc.stop()
            batch_peak_mb = peak_mem / (1024.0 * 1024.0)
            batch_delta_mb = current_mem / (1024.0 * 1024.0)
        else:
            batch_peak_mb = 0.0
            batch_delta_mb = 0.0

        mean_batch_lat = float(np.mean(batch_latencies))
        mean_item_lat = mean_batch_lat / b_size
        batch_throughput = (b_size / (mean_batch_lat / 1000.0)) if mean_batch_lat > 0 else 0.0

        batch_metrics_list.append(
            BatchBenchmarkMetrics(
                batch_size=b_size,
                iterations=n_repeats,
                mean_batch_latency_ms=mean_batch_lat,
                mean_per_item_latency_ms=mean_item_lat,
                throughput_items_per_sec=batch_throughput,
                memory_peak_mb=batch_peak_mb,
                memory_delta_mb=batch_delta_mb,
            )
        )

    # 4. Reproducibility & Determinism Verification
    eval_slice = df.head(min(50, len(df))).copy()
    rec1 = pipeline.batch_recommend(eval_slice, objective="profit")
    rec2 = pipeline.batch_recommend(eval_slice, objective="profit")

    diff_elasticity = np.max(np.abs(rec1["elasticity"].values - rec2["elasticity"].values))
    diff_price = np.max(np.abs(rec1["optimal_price"].values - rec2["optimal_price"].values))
    actions_match = bool((rec1["pricing_action"].values == rec2["pricing_action"].values).all())

    repro_metrics = ReproducibilityMetrics(
        is_reproducible=(diff_elasticity < 1e-7 and diff_price < 1e-7 and actions_match),
        max_elasticity_abs_diff=float(diff_elasticity),
        max_optimal_price_abs_diff=float(diff_price),
        identical_actions=actions_match,
        sample_size=len(eval_slice),
    )

    metadata = getattr(pipeline, "metadata_", {})
    model_version = str(metadata.get("model_version", "1.0.0"))
    model_id = str(metadata.get("model_id", "unknown"))

    return ModelBenchmarkSummary(
        single_item=single_metrics,
        batches=batch_metrics_list,
        reproducibility=repro_metrics,
        model_version=model_version,
        model_id=model_id,
        system_info=get_system_info(),
    )
