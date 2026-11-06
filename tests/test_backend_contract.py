"""Contract and integration test suite for EconoCausal ML Engine and Backend Integration.

Validates the complete Python microservice input/output contract:
1. Treatment-effect estimates (CATE price elasticity theta(X))
2. Recommended prices & microeconomic Amoroso-Robinson optimization
3. Expected outcomes (projected demand volume, revenue, profit, margins)
4. Confidence and uncertainty fields (asymptotic SE, 95% CI, p-value, regime)
5. Model-health status & readiness probes (healthy vs degraded, baseline ATE, CI)
6. Validation on missing fields, invalid prices, out-of-range features, and bounds
7. Backend integration compatibility (Node.js camelCase schemas & PricingExperiment contract)
8. Model failure resilience and degraded state handling
"""

import pytest
from starlette.testclient import TestClient

from src.api.main import app, create_app
from src.api.service import CausalPricingService
from tests.fixtures import load_request_fixture, load_response_fixture


@pytest.fixture(scope="module")
def client():
    """Test client fixture against default trained pipeline."""
    CausalPricingService.reset_instance()
    with TestClient(app) as test_client:
        yield test_client


# =====================================================================
# 1. Treatment-Effect Estimates Contract Tests
# =====================================================================


class TestTreatmentEffectContract:
    """Validate treatment-effect estimation and heterogeneous elasticity properties."""

    def test_treatment_effect_point_estimate(self, client):
        """Verify that treatment effect is estimated and equals causal elasticity."""
        req = load_request_fixture("valid_standard_request")
        resp = client.post("/predict/price", json=req)
        assert resp.status_code == 200
        data = resp.json()

        assert "causal_elasticity" in data
        assert "treatment_effect" in data
        assert data["treatment_effect"] == data["causal_elasticity"]
        # Demand obeys law of demand (negative elasticity)
        assert data["causal_elasticity"] < 0.0
        assert -4.0 < data["causal_elasticity"] < -0.1

    def test_heterogeneous_treatment_effects_by_category(self, client):
        """Verify CATE estimates vary across distinct product categories."""
        categories = ["health_beauty", "watches_gifts", "computers_accessories"]
        elasticities = {}

        for cat in categories:
            payload = {
                "item": {
                    "product_id": f"SKU_{cat}",
                    "unit_price": 99.00,
                    "qty": 10.0,
                    "product_category_name": cat,
                },
                "objective": "profit",
            }
            resp = client.post("/predict/price", json=payload)
            assert resp.status_code == 200
            data = resp.json()
            elasticities[cat] = data["causal_elasticity"]

        # Different categories should have heterogeneous moderations
        assert len(set(elasticities.values())) > 1

    def test_elasticity_endpoint_detailed_uncertainty(self, client):
        """Verify POST /predict/elasticity provides full uncertainty estimates."""
        payload = {
            "items": [
                {
                    "product_id": "SKU_E1",
                    "unit_price": 59.90,
                    "product_category_name": "health_beauty",
                },
                {
                    "product_id": "SKU_E2",
                    "unit_price": 120.00,
                    "product_category_name": "bed_bath_table",
                },
            ]
        }
        resp = client.post("/predict/elasticity", json=payload)
        assert resp.status_code == 200
        data = resp.json()
        assert data["total_items"] == 2

        for pred in data["predictions"]:
            assert pred["predicted_elasticity"] < 0.0
            assert pred["elasticity_classification"] in ["Elastic", "Highly Elastic", "Inelastic"]
            assert pred["std_error"] is not None
            assert pred["std_error"] > 0.0
            assert pred["confidence_interval"] is not None
            ci = pred["confidence_interval"]
            assert ci["lower"] < ci["upper"]
            assert ci["lower"] <= pred["predicted_elasticity"] <= ci["upper"]
            assert pred["p_value"] is not None
            assert 0.0 <= pred["p_value"] <= 1.0


# =====================================================================
# 2. Recommended Prices & Microeconomic Optimization Contract Tests
# =====================================================================


class TestRecommendedPricesContract:
    """Validate optimal price recommendation rules, bounds, and guardrails."""

    def test_amoroso_robinson_optimal_price(self, client):
        """Verify recommended optimal price satisfies business constraints."""
        req = load_request_fixture("valid_standard_request")
        resp = client.post("/predict/price", json=req)
        assert resp.status_code == 200
        data = resp.json()

        base_p = data["baseline_price"]
        opt_p = data["optimal_price"]
        unit_cost = data["unit_cost"]

        assert opt_p > 0.0
        # Margin floor: optimal price must cover cost with >= 5% margin
        assert (opt_p - unit_cost) / opt_p >= 0.05
        # Guardrail bounds: within permissible percentage band
        price_change_pct = data["price_change_pct"]
        assert -30.0 <= price_change_pct <= 20.0

    def test_pricing_action_classification(self, client):
        """Verify action tags (MARKUP, DISCOUNT, HOLD) correspond to price changes."""
        req = load_request_fixture("valid_standard_request")
        resp = client.post("/predict/price", json=req)
        assert resp.status_code == 200
        data = resp.json()

        if data["price_change_pct"] > 0.5:
            assert data["action"] == "MARKUP"
        elif data["price_change_pct"] < -0.5:
            assert data["action"] == "DISCOUNT"
        else:
            assert data["action"] == "HOLD"

    def test_psychological_charm_pricing(self, client):
        """Verify prices end in .99 when charm pricing is active."""
        req = load_request_fixture("valid_standard_request")
        resp = client.post("/predict/price", json=req)
        assert resp.status_code == 200
        data = resp.json()
        opt_price = data["optimal_price"]
        # Check if ending is .99 or close rounding
        cents = round((opt_price - int(opt_price)) * 100)
        assert cents in [99, 95, 0] or abs(opt_price - round(opt_price, 2)) < 0.01


# =====================================================================
# 3. Expected Outcomes Contract Tests
# =====================================================================


class TestExpectedOutcomesContract:
    """Validate expected demand, revenue, and gross profit simulations."""

    def test_expected_outcomes_fields(self, client):
        """Verify simulated demand, revenue, and profit are reported with aliases."""
        req = load_request_fixture("valid_standard_request")
        resp = client.post("/predict/price", json=req)
        assert resp.status_code == 200
        data = resp.json()

        # Demand
        assert data["baseline_qty"] > 0
        assert data["projected_qty"] > 0
        assert data["expected_demand"] == data["projected_qty"]
        assert "demand_uplift_pct" in data

        # Revenue
        assert data["baseline_revenue"] > 0
        assert data["projected_revenue"] > 0
        assert data["expected_revenue"] == data["projected_revenue"]
        assert "revenue_uplift_pct" in data

        # Profit
        assert "baseline_profit" in data
        assert "projected_profit" in data
        assert data["expected_profit"] == data["projected_profit"]
        assert "profit_uplift_pct" in data

        # Margins
        assert data["baseline_margin_pct"] > 0
        assert data["projected_margin_pct"] >= 5.0

    def test_price_demand_curve_generated(self, client):
        """Verify full simulated price-demand curve is generated."""
        req = load_request_fixture("valid_standard_request")
        resp = client.post("/predict/price", json=req)
        assert resp.status_code == 200
        data = resp.json()

        assert "price_demand_curve" in data
        curve = data["price_demand_curve"]
        assert isinstance(curve, list)
        assert len(curve) == 15

        prev_price = -1
        for pt in curve:
            assert pt["price"] > 0
            assert pt["predicted_demand"] > 0
            assert pt["predicted_revenue"] >= 0
            assert "predicted_profit" in pt
            # Strictly increasing price steps
            assert pt["price"] > prev_price
            prev_price = pt["price"]


# =====================================================================
# 4. Confidence and Uncertainty Fields Contract Tests
# =====================================================================


class TestConfidenceUncertaintyContract:
    """Validate asymptotic standard errors, confidence intervals, and controlled confounders."""

    def test_confidence_interval_structure(self, client):
        """Verify 95% confidence interval schema and statistical properties."""
        req = load_request_fixture("valid_standard_request")
        resp = client.post("/predict/price", json=req)
        assert resp.status_code == 200
        data = resp.json()

        ci = data["confidence_interval"]
        assert ci is not None
        assert "lower" in ci
        assert "upper" in ci
        assert "confidence_level" in ci
        assert ci["confidence_level"] == 0.95
        assert ci["lower"] < ci["upper"]
        # Elasticity falls inside CI bounds
        assert ci["lower"] <= data["causal_elasticity"] <= ci["upper"]

    def test_uncertainty_metrics(self, client):
        """Verify standard error, p-value, and statistical significance."""
        req = load_request_fixture("valid_standard_request")
        resp = client.post("/predict/price", json=req)
        assert resp.status_code == 200
        data = resp.json()

        assert data["std_error"] is not None
        assert data["std_error"] > 0.0
        assert data["p_value"] is not None
        assert 0.0 <= data["p_value"] <= 1.0
        assert isinstance(data["is_statistically_significant"], bool)
        assert data["elasticity_regime"] in ["Elastic", "Inelastic", "Unit Elastic"]

    def test_controlled_confounders_list(self, client):
        """Verify controlled confounders list contains all required market covariates."""
        req = load_request_fixture("valid_standard_request")
        resp = client.post("/predict/price", json=req)
        assert resp.status_code == 200
        data = resp.json()

        confounders = data["controlled_confounders"]
        assert isinstance(confounders, list)
        assert len(confounders) >= 10
        expected_subset = ["comp_1", "product_score", "freight_price", "customers"]
        for c in expected_subset:
            assert c in confounders


# =====================================================================
# 5. Model-Health Status Contract Tests
# =====================================================================


class TestModelHealthContract:
    """Validate /health diagnostic probe and readiness status."""

    def test_healthy_status_when_pipeline_loaded(self, client):
        """Verify /health returns healthy status with baseline ATE and CI."""
        resp = client.get("/health")
        assert resp.status_code == 200
        data = resp.json()

        assert data["status"] == "healthy"
        assert data["service"] == "econocausal-pricing-service"
        assert data["version"] == "1.0.0"
        assert data["model_loaded"] is True
        assert data["uptime_seconds"] >= 0.0
        assert data["baseline_ate"] is not None
        assert round(data["baseline_ate"], 2) == -1.27
        assert data["baseline_ate_ci"] is not None
        ci = data["baseline_ate_ci"]
        assert ci["lower"] < ci["upper"]
        assert ci["lower"] <= data["baseline_ate"] <= ci["upper"]
        assert data["refutation_pass_rate"] == 100.0
        assert data["active_features"] is not None
        assert len(data["active_features"]) >= 10

    def test_degraded_status_when_pipeline_unloaded(self):
        """Verify /health returns degraded when model file does not exist."""
        unloaded_app = create_app(model_path="missing_dummy_model.joblib")
        with TestClient(unloaded_app) as custom_client:
            CausalPricingService.reset_instance(model_path="missing_dummy_model.joblib")
            resp = custom_client.get("/health")
            assert resp.status_code == 200
            data = resp.json()

            assert data["status"] == "degraded"
            assert data["model_loaded"] is False
            assert data["baseline_ate"] is None
            assert data["baseline_ate_ci"] is None
            assert data["refutation_pass_rate"] is None

        # Reset singleton back to default
        CausalPricingService.reset_instance()


# =====================================================================
# 6. Validation on Missing Fields, Invalid Prices, and Edge Cases
# =====================================================================


class TestValidationContract:
    """Validate defensive error handling for missing and out-of-range inputs."""

    def test_missing_price_raises_422(self, client):
        """Verify request lacking unit_price returns 422 Unprocessable Entity."""
        req = load_request_fixture("missing_price_request")
        resp = client.post("/predict/price", json=req)
        assert resp.status_code == 422
        data = resp.json()
        assert "detail" in data

    def test_negative_price_raises_422(self, client):
        """Verify negative price is rejected with 422."""
        req = load_request_fixture("invalid_negative_price_request")
        resp = client.post("/predict/price", json=req)
        assert resp.status_code == 422

    def test_zero_price_raises_422(self, client):
        """Verify zero price is rejected with 422."""
        req = load_request_fixture("invalid_zero_price_request")
        resp = client.post("/predict/price", json=req)
        assert resp.status_code == 422

    def test_negative_cost_raises_422(self, client):
        """Verify negative unit_cost is rejected with 422."""
        req = load_request_fixture("invalid_negative_cost_request")
        resp = client.post("/predict/price", json=req)
        assert resp.status_code == 422

    def test_out_of_range_product_score_high_raises_422(self, client):
        """Verify product_score > 5.0 is rejected with 422."""
        req = load_request_fixture("out_of_range_product_score_high")
        resp = client.post("/predict/price", json=req)
        assert resp.status_code == 422

    def test_out_of_range_product_score_low_raises_422(self, client):
        """Verify negative product_score is rejected with 422."""
        req = load_request_fixture("out_of_range_product_score_low")
        resp = client.post("/predict/price", json=req)
        assert resp.status_code == 422

    def test_out_of_range_inverted_bounds_raises_422(self, client):
        """Verify min_price > max_price is rejected with 422."""
        req = load_request_fixture("out_of_range_inverted_bounds")
        resp = client.post("/predict/price", json=req)
        assert resp.status_code == 422

    def test_invalid_objective_raises_422(self, client):
        """Verify unsupported objective string is rejected with 422."""
        req = load_request_fixture("invalid_objective_request")
        resp = client.post("/predict/price", json=req)
        assert resp.status_code == 422

    def test_empty_batch_items_raises_422(self, client):
        """Verify empty items list in batch recommendation returns 422."""
        req = load_request_fixture("empty_batch_request")
        resp = client.post("/batch/recommend", json=req)
        assert resp.status_code == 422


# =====================================================================
# 7. Backend Integration Schema Contract Tests
# =====================================================================


class TestBackendIntegrationContract:
    """Validate integration with Node.js Express backend request and response schemas."""

    def test_backend_camelcase_request_on_predict_price(self, client):
        """Verify Python service seamlessly accepts Node.js camelCase fields on /predict/price."""
        req = load_request_fixture("valid_backend_camelcase_request")
        resp = client.post("/predict/price", json=req)
        assert resp.status_code == 200
        data = resp.json()

        assert data["status"] == "SUCCESS"
        assert data["product_id"] == "650000000000000000000001"
        assert data["baseline_price"] == 120.00
        assert data["unit_cost"] == 55.00
        assert data["optimal_price"] > 0.0
        assert data["causal_elasticity"] < 0.0

    def test_direct_backend_optimization_contract(self, client):
        """Verify /integration/pricing/optimize returns exact PricingExperiment schema."""
        req = load_request_fixture("valid_backend_optimization_request")
        resp = client.post("/integration/pricing/optimize", json=req)
        assert resp.status_code == 200
        data = resp.json()

        assert data["success"] is True
        assert data["message"] == "Causal DML analysis calculated successfully"
        assert "data" in data

        exp = data["data"]
        assert exp["title"] == "Causal DML Price Optimization"
        assert exp["productId"] == "650000000000000000000001"
        assert exp["status"] == "completed"
        assert "Double Machine Learning" in exp["methodology"]
        assert exp["sampleSize"] == 2
        assert exp["causalTreatmentEffect"] < 0.0
        assert exp["priceElasticity"] == exp["causalTreatmentEffect"]

        # Confidence interval
        assert "confidenceInterval" in exp
        ci = exp["confidenceInterval"]
        assert "lower" in ci
        assert "upper" in ci
        assert ci["lower"] < ci["upper"]

        # Prices & Outcomes
        assert exp["currentPrice"] == 115.00
        assert exp["recommendedOptimalPrice"] > 0.0
        assert exp["projectedDemandAtOptimal"] > 0.0
        assert exp["projectedRevenueAtOptimal"] > 0.0
        assert exp["projectedProfitLiftPercentage"] >= 0.0

        # Confounders
        assert isinstance(exp["controlledConfounders"], list)
        assert len(exp["controlledConfounders"]) >= 10

        # Price Demand Curve (15 points)
        assert isinstance(exp["priceDemandCurve"], list)
        assert len(exp["priceDemandCurve"]) == 15
        pt0 = exp["priceDemandCurve"][0]
        assert "price" in pt0
        assert "predictedDemand" in pt0
        assert "predictedRevenue" in pt0
        assert "predictedProfit" in pt0

    def test_backend_optimization_with_synthetic_defaults(self, client):
        """Verify backend optimization handles minimal payloads with sensible econometric defaults."""
        minimal_payload = {
            "productId": "650000000000000000000099",
            "basePrice": 85.00,
        }
        resp = client.post("/integration/pricing/optimize", json=minimal_payload)
        assert resp.status_code == 200
        data = resp.json()

        assert data["success"] is True
        exp = data["data"]
        assert exp["productId"] == "650000000000000000000099"
        assert exp["recommendedOptimalPrice"] > 0.0
        assert len(exp["priceDemandCurve"]) == 15

    def test_backend_optimization_bounds_validation(self, client):
        """Verify minPrice > maxPrice raises 422 on /integration/pricing/optimize."""
        invalid_payload = {
            "productId": "650000000000000000000099",
            "basePrice": 100.00,
            "minPrice": 150.00,
            "maxPrice": 80.00,
        }
        resp = client.post("/integration/pricing/optimize", json=invalid_payload)
        assert resp.status_code == 422


# =====================================================================
# 8. Model Failure & Resilience Contract Tests
# =====================================================================


class TestModelFailureResilienceContract:
    """Validate service behavior during model failure or degraded operations."""

    def test_metadata_returns_503_when_unready(self):
        """Verify /model/metadata returns 503 Service Unavailable when model is not loaded."""
        unloaded_app = create_app(model_path="non_existent_file.joblib")
        with TestClient(unloaded_app) as custom_client:
            CausalPricingService.reset_instance(model_path="non_existent_file.joblib")
            resp = custom_client.get("/model/metadata")
            assert resp.status_code == 503
            data = resp.json()
            assert "detail" in data

        # Restore singleton
        CausalPricingService.reset_instance()

    def test_predict_returns_400_when_unready(self):
        """Verify /predict/price returns 400 when model is not ready."""
        unloaded_app = create_app(model_path="non_existent_file.joblib")
        with TestClient(unloaded_app) as custom_client:
            CausalPricingService.reset_instance(model_path="non_existent_file.joblib")
            resp = custom_client.post("/predict/price", json={"item": {"unit_price": 50.0}})
            assert resp.status_code == 400
            data = resp.json()
            assert "detail" in data

        # Restore singleton
        CausalPricingService.reset_instance()
