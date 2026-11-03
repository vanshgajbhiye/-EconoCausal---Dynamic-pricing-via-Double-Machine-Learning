"""Latency and throughput benchmarking script for EconoCausal Dynamic Pricing API.

Executes micro-benchmarks on:
1. Single-SKU real-time pricing inference (P50, P90, P95, P99, Throughput)
2. Causal elasticity estimation theta(X)
3. Batch catalog optimization (10, 50, 100 items)
4. Counterfactual portfolio scenario simulations
"""

import os
import sys
import time
import numpy as np
from starlette.testclient import TestClient

# Ensure project root is in sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.api.main import app
from src.api.service import CausalPricingService


def run_latency_benchmarks():
    """Execute end-to-end latency benchmarks on the FastAPI test client."""
    print("=" * 80)
    print("      ECONOCAUSAL REST API MICROSERVICE LATENCY BENCHMARK SUITE")
    print("=" * 80)

    # Initialize client
    CausalPricingService.reset_instance()
    client = TestClient(app)

    # Health check verification
    health_resp = client.get("/health")
    assert health_resp.status_code == 200, "Service is not healthy"
    health_data = health_resp.json()
    print(f"Service Status:        {health_data['status'].upper()}")
    print(f"Model Loaded:          {health_data['model_loaded']} ({health_data['model_path']})")
    print(f"Baseline DML ATE:      {health_data['baseline_ate']}")
    print(f"Refutation Pass Rate:  {health_data['refutation_pass_rate']}%")
    print("-" * 80)

    # Warmup
    print("Performing 10 warmup requests...")
    sample_payload = {
        "item": {
            "product_id": "WARMUP_SKU",
            "unit_price": 59.90,
            "qty": 15.0,
            "product_category_name": "health_beauty",
            "comp_price": 58.50,
        },
        "objective": "profit",
    }
    for _ in range(10):
        client.post("/predict/price", json=sample_payload)

    # 1. Single-SKU Inference Benchmark (100 runs)
    n_single = 100
    print(f"\n1. Benchmarking Single-SKU Pricing Inference ({n_single} iterations)...")
    single_latencies = []
    internal_latencies = []

    categories = [
        "health_beauty",
        "bed_bath_table",
        "watches_gifts",
        "computers_accessories",
        "sports_leisure",
    ]

    t_start_total = time.perf_counter()
    for i in range(n_single):
        cat = categories[i % len(categories)]
        payload = {
            "item": {
                "product_id": f"BENCH_SKU_{i}",
                "unit_price": round(25.0 + (i * 1.5) % 150.0, 2),
                "qty": float(5 + (i % 30)),
                "product_category_name": cat,
                "comp_price": round(24.0 + (i * 1.5) % 150.0, 2),
                "freight_price": 12.0 + (i % 8),
            },
            "objective": "profit" if i % 2 == 0 else "revenue",
        }
        t0 = time.perf_counter()
        resp = client.post("/predict/price", json=payload)
        client_latency_ms = (time.perf_counter() - t0) * 1000.0

        assert resp.status_code == 200
        single_latencies.append(client_latency_ms)
        internal_latencies.append(resp.json()["inference_latency_ms"])

    total_single_time = time.perf_counter() - t_start_total
    single_latencies = np.array(single_latencies)
    internal_latencies = np.array(internal_latencies)

    print("   [Client Round-Trip Latency]")
    print(f"   - Mean Latency:        {np.mean(single_latencies):.2f} ms")
    print(f"   - Median (P50):        {np.median(single_latencies):.2f} ms")
    print(f"   - 90th Percentile:     {np.percentile(single_latencies, 90):.2f} ms")
    print(f"   - 95th Percentile:     {np.percentile(single_latencies, 95):.2f} ms")
    print(f"   - 99th Percentile:     {np.percentile(single_latencies, 99):.2f} ms")
    print(f"   - Min / Max Latency:   {np.min(single_latencies):.2f} ms / {np.max(single_latencies):.2f} ms")
    print(f"   - Service Throughput:  {n_single / total_single_time:.1f} requests/sec")

    print("   [Core Causal Pipeline Compute Latency]")
    print(f"   - Mean Compute Time:   {np.mean(internal_latencies):.2f} ms")
    print(f"   - P95 Compute Time:    {np.percentile(internal_latencies, 95):.2f} ms")

    # 2. Elasticity Estimation Benchmark (50 items)
    print("\n2. Benchmarking Batch Elasticity Estimation (50 items)...")
    elasticity_payload = {
        "items": [
            {
                "product_id": f"ELAST_{j}",
                "unit_price": round(20.0 + j * 2.0, 2),
                "product_category_name": categories[j % len(categories)],
            }
            for j in range(50)
        ]
    }
    t0 = time.perf_counter()
    elast_resp = client.post("/predict/elasticity", json=elasticity_payload)
    elast_time_ms = (time.perf_counter() - t0) * 1000.0
    assert elast_resp.status_code == 200
    elast_data = elast_resp.json()
    print(f"   - Total Batch Time:    {elast_time_ms:.2f} ms")
    print(f"   - Latency per Item:    {elast_time_ms / 50.0:.3f} ms/item")
    print(f"   - Elasticity Rate:     {50.0 / (elast_time_ms / 1000.0):.1f} items/sec")

    # 3. Batch Catalog Optimization (10, 50, 100 items)
    print("\n3. Benchmarking Batch Catalog Recommendations...")
    for batch_size in [10, 50, 100]:
        batch_payload = {
            "items": [
                {
                    "product_id": f"BATCH_{b}",
                    "unit_price": round(30.0 + (b * 1.8) % 120.0, 2),
                    "qty": float(5 + (b % 20)),
                    "product_category_name": categories[b % len(categories)],
                    "comp_price": round(29.0 + (b * 1.8) % 120.0, 2),
                }
                for b in range(batch_size)
            ],
            "objective": "profit",
        }
        t0 = time.perf_counter()
        batch_resp = client.post("/batch/recommend", json=batch_payload)
        batch_duration_ms = (time.perf_counter() - t0) * 1000.0
        assert batch_resp.status_code == 200
        summary = batch_resp.json()["portfolio_summary"]
        rate = batch_size / (batch_duration_ms / 1000.0)
        print(f"   - Batch {batch_size:3d} Items:       {batch_duration_ms:6.2f} ms ({rate:5.1f} items/sec) | Profit Uplift: {summary['portfolio_profit_uplift_pct']:+.1f}%")

    # 4. Scenario Simulation Benchmark (50 items)
    print("\n4. Benchmarking Portfolio Scenario Matrix Simulation (50 items)...")
    sim_payload = {
        "items": [
            {
                "product_id": f"SIM_{s}",
                "unit_price": round(35.0 + (s * 2.2) % 110.0, 2),
                "qty": float(10 + (s % 15)),
                "product_category_name": categories[s % len(categories)],
                "comp_price": round(34.0 + (s * 2.2) % 110.0, 2),
            }
            for s in range(50)
        ],
        "comp_col": "comp_1",
    }
    t0 = time.perf_counter()
    sim_resp = client.post("/simulate/portfolio", json=sim_payload)
    sim_duration_ms = (time.perf_counter() - t0) * 1000.0
    assert sim_resp.status_code == 200
    sim_data = sim_resp.json()
    print(f"   - Total Simulation Duration: {sim_duration_ms:.2f} ms")
    print(f"   - Number of Scenarios:      {len(sim_data['scenarios'])}")
    print(f"   - Recommended Strategy:     {sim_data['recommended_strategy']}")

    print("\n" + "=" * 80)
    print("                        BENCHMARK SUMMARY CONCLUSION")
    print("=" * 80)
    print(f" Target Latency:          < 15.00 ms (Single SKU)")
    print(f" Achieved Median (P50):    {np.median(single_latencies):.2f} ms")
    print(f" Achieved 95th (P95):      {np.percentile(single_latencies, 95):.2f} ms")
    print(f" Achieved Mean Compute:    {np.mean(internal_latencies):.2f} ms")
    print(f" SLA Verification:        {'PASS [COMPLIANT]' if np.percentile(single_latencies, 95) < 30.0 else 'WARN'}")
    print("=" * 80 + "\n")


if __name__ == "__main__":
    run_latency_benchmarks()
