"""Unit and integration tests for EconoCausal Dynamic Pricing REST API Microservice."""

import pytest
from starlette.testclient import TestClient

from src.api.main import app, create_app
from src.api.service import CausalPricingService


@pytest.fixture(scope="module")
def client():
    """Shared TestClient fixture against the default loaded API application."""
    # Reset singleton to ensure it picks up the default trained pipeline
    CausalPricingService.reset_instance()
    with TestClient(app) as test_client:
        yield test_client


def test_api_root(client):
    """Test API root information endpoint."""
    response = client.get("/")
    assert response.status_code == 200
    data = response.json()
    assert "version" in data
    assert data["version"] == "1.0.0"
    assert "/docs" in data["docs_url"]


def test_health_check_healthy(client):
    """Test /health endpoint returns healthy when model is present."""
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "healthy"
    assert data["service"] == "econocausal-pricing-service"
    assert data["model_loaded"] is True
    assert data["baseline_ate"] is not None
    assert round(data["baseline_ate"], 2) == -1.27
    assert data["refutation_pass_rate"] == 100.0
    assert data["uptime_seconds"] >= 0.0


def test_model_metadata(client):
    """Test /model/metadata returns full training diagnostics."""
    response = client.get("/model/metadata")
    assert response.status_code == 200
    data = response.json()
    assert data["is_fitted"] is True
    assert data["baseline_ate"] is not None
    assert data["refutation_pass_rate"] == 100.0
    assert "optimizer_bounds" in data
    assert data["optimizer_bounds"]["min_price_change_pct"] == -0.30
    assert data["optimizer_bounds"]["max_price_change_pct"] == 0.20
    assert data["optimizer_bounds"]["charm_pricing"] is True
    assert "feature_imputation_medians" in data


def test_predict_price_profit_objective(client):
    """Test real-time single-SKU price prediction with profit objective."""
    payload = {
        "item": {
            "product_id": "SKU_HEALTH_001",
            "unit_price": 59.90,
            "qty": 15.0,
            "product_category_name": "health_beauty",
            "comp_price": 58.50,
            "freight_price": 12.50,
        },
        "objective": "profit",
    }
    response = client.post("/predict/price", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "SUCCESS"
    assert data["product_id"] == "SKU_HEALTH_001"
    assert data["action"] in ["MARKUP", "DISCOUNT", "HOLD"]
    assert data["baseline_price"] == 59.90
    assert data["optimal_price"] > 0
    assert data["causal_elasticity"] < 0
    assert data["projected_margin_pct"] >= 5.0
    assert data["objective"] == "profit"
    assert data["inference_latency_ms"] is not None
    assert data["inference_latency_ms"] >= 0


def test_predict_price_revenue_objective(client):
    """Test real-time single-SKU price prediction with revenue objective."""
    payload = {
        "item": {
            "product_id": "SKU_BED_002",
            "unit_price": 120.00,
            "qty": 5.0,
            "product_category_name": "bed_bath_table",
        },
        "objective": "revenue",
    }
    response = client.post("/predict/price", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "SUCCESS"
    assert data["objective"] == "revenue"
    assert data["optimal_price"] > 0


def test_predict_price_validation_invalid_price(client):
    """Test validation reject when unit_price <= 0."""
    payload = {
        "item": {
            "unit_price": -10.0,
            "qty": 5.0,
        },
        "objective": "profit",
    }
    response = client.post("/predict/price", json=payload)
    assert response.status_code == 422  # Pydantic validation error


def test_predict_price_validation_missing_unit_price(client):
    """Test validation reject when unit_price is missing."""
    payload = {
        "item": {
            "qty": 5.0,
        },
    }
    response = client.post("/predict/price", json=payload)
    assert response.status_code == 422


def test_predict_elasticity(client):
    """Test heterogeneous elasticity estimation endpoint."""
    payload = {
        "items": [
            {
                "product_id": "SKU_01",
                "unit_price": 59.90,
                "product_category_name": "health_beauty",
            },
            {
                "product_id": "SKU_02",
                "unit_price": 149.99,
                "product_category_name": "computers_accessories",
            },
        ]
    }
    response = client.post("/predict/elasticity", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert data["total_items"] == 2
    assert len(data["predictions"]) == 2
    for pred in data["predictions"]:
        assert pred["predicted_elasticity"] < 0
        assert pred["elasticity_classification"] in ["Elastic", "Highly Elastic", "Inelastic"]


def test_batch_recommend(client):
    """Test batch catalog recommendation endpoint."""
    payload = {
        "items": [
            {
                "product_id": "BATCH_01",
                "unit_price": 59.90,
                "qty": 15.0,
                "product_category_name": "health_beauty",
            },
            {
                "product_id": "BATCH_02",
                "unit_price": 120.00,
                "qty": 5.0,
                "product_category_name": "bed_bath_table",
            },
            {
                "product_id": "BATCH_03",
                "unit_price": 35.00,
                "qty": 25.0,
                "product_category_name": "watches_gifts",
            },
        ],
        "objective": "profit",
    }
    response = client.post("/batch/recommend", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert data["total_items"] == 3
    assert len(data["recommendations"]) == 3
    assert "portfolio_summary" in data
    summary = data["portfolio_summary"]
    assert summary["total_baseline_revenue"] > 0
    assert summary["total_projected_revenue"] > 0
    assert summary["total_baseline_profit"] > 0
    assert summary["total_projected_profit"] > 0
    assert (summary["markup_count"] + summary["discount_count"] + summary["hold_count"]) == 3
    assert data["batch_latency_ms"] >= 0


def test_batch_recommend_empty_list_rejected(client):
    """Test batch recommendation rejects empty list."""
    payload = {"items": []}
    response = client.post("/batch/recommend", json=payload)
    assert response.status_code == 422


def test_simulate_portfolio_scenarios(client):
    """Test counterfactual scenario comparison matrix endpoint."""
    payload = {
        "items": [
            {
                "product_id": "SIM_01",
                "unit_price": 59.90,
                "qty": 15.0,
                "product_category_name": "health_beauty",
            },
            {
                "product_id": "SIM_02",
                "unit_price": 120.00,
                "qty": 5.0,
                "product_category_name": "bed_bath_table",
            },
        ],
        "comp_col": "comp_1",
    }
    response = client.post("/simulate/portfolio", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert data["total_items"] == 2
    assert len(data["scenarios"]) == 6
    scenario_names = [s["strategy"] for s in data["scenarios"]]
    assert "Status Quo (Baseline)" in scenario_names
    assert "Flat 10% Discount" in scenario_names
    assert "Flat 10% Markup" in scenario_names
    assert "Competitor Benchmark" in scenario_names
    assert "Causal Revenue Maximizer" in scenario_names
    assert "Causal Profit Maximizer" in scenario_names
    assert data["recommended_strategy"] == "Causal Profit Maximizer"


def test_process_time_header(client):
    """Test that X-Process-Time-Ms header is injected into responses."""
    response = client.get("/health")
    assert "X-Process-Time-Ms" in response.headers
    val = float(response.headers["X-Process-Time-Ms"])
    assert val >= 0.0


def test_cors_headers(client):
    """Test CORS preflight or origin header reflections."""
    response = client.options(
        "/predict/price",
        headers={
            "Origin": "http://localhost:3000",
            "Access-Control-Request-Method": "POST",
        },
    )
    assert response.headers.get("access-control-allow-origin") == "http://localhost:3000"


def test_degraded_service_when_model_missing():
    """Test graceful degradation when model file does not exist."""
    custom_app = create_app(model_path="non_existent_model_file.joblib")
    with TestClient(custom_app) as test_client:
        # Reset singleton with invalid path
        CausalPricingService.reset_instance(model_path="non_existent_model_file.joblib")
        health = test_client.get("/health").json()
        assert health["status"] == "degraded"
        assert health["model_loaded"] is False

        # Metadata should return 503
        meta_resp = test_client.get("/model/metadata")
        assert meta_resp.status_code == 503

        # Predict price should return 400 Bad Request
        pred_resp = test_client.post("/predict/price", json={"item": {"unit_price": 50.0}})
        assert pred_resp.status_code == 400

    # Restore singleton for subsequent tests
    CausalPricingService.reset_instance()
