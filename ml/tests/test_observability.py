"""Unit and integration tests for Day 8 Observability, Dockerization, and Deployment Manifests."""

import os
from pathlib import Path
import pytest
from starlette.testclient import TestClient
import yaml

from src.api.main import create_app
from src.api.service import CausalPricingService


@pytest.fixture(scope="module")
def api_client():
    """Create a test client with loaded model pipeline."""
    model_path = "models/econocausal_pipeline.joblib"
    if not os.path.exists(model_path):
        pytest.skip(f"Model file {model_path} not found. Skipping API integration tests.")

    # Reset service singleton to ensure clean test state
    CausalPricingService.reset_instance()
    app = create_app(model_path=model_path)
    with TestClient(app) as client:
        yield client


class TestPrometheusMetrics:
    """Test suite for Prometheus observability and telemetry export."""

    def test_metrics_endpoint_status_and_content_type(self, api_client):
        """Verify /metrics endpoint returns HTTP 200 and Prometheus exposition format."""
        response = api_client.get("/metrics")
        assert response.status_code == 200
        content_type = response.headers.get("content-type", "")
        assert "text/plain" in content_type

    def test_metrics_contain_core_metric_definitions(self, api_client):
        """Verify core metric names are exported in /metrics output."""
        response = api_client.get("/metrics")
        body = response.text
        assert "econocausal_http_requests_total" in body
        assert "econocausal_http_request_duration_seconds" in body
        assert "econocausal_http_requests_active" in body
        assert "econocausal_model_info" in body

    def test_metrics_track_http_requests(self, api_client):
        """Verify HTTP requests increment the request counter."""
        # Query /health to generate request traffic
        api_client.get("/health")
        response = api_client.get("/metrics")
        assert 'endpoint="/health"' in response.text

    def test_metrics_track_pricing_recommendations(self, api_client):
        """Verify dynamic pricing inference records recommendation metrics."""
        payload = {
            "item": {
                "product_id": "OBS_TEST_SKU_1",
                "product_category_name": "health_beauty",
                "unit_price": 105.00,
                "comp_price": 108.00,
                "freight_price": 12.50,
            },
            "objective": "profit",
            "cost": 70.00,
        }
        pred_resp = api_client.post("/predict/price", json=payload)
        assert pred_resp.status_code == 200

        metrics_resp = api_client.get("/metrics")
        body = metrics_resp.text
        assert "econocausal_pricing_recommendations_total" in body
        assert 'product_category="health_beauty"' in body
        assert "econocausal_cate_elasticity_latest" in body

    def test_metrics_track_batch_recommendations(self, api_client):
        """Verify batch catalog recommendations update telemetry for multiple items."""
        payload = {
            "items": [
                {
                    "product_id": "BATCH_OBS_1",
                    "product_category_name": "computers_accessories",
                    "unit_price": 140.00,
                    "qty": 5.0,
                },
                {
                    "product_id": "BATCH_OBS_2",
                    "product_category_name": "cool_stuff",
                    "unit_price": 115.00,
                    "qty": 10.0,
                },
            ],
            "objective": "profit",
        }
        batch_resp = api_client.post("/batch/recommend", json=payload)
        assert batch_resp.status_code == 200

        metrics_resp = api_client.get("/metrics")
        body = metrics_resp.text
        assert 'product_category="computers_accessories"' in body
        assert 'product_category="cool_stuff"' in body


class TestDockerAndDeploymentArtifacts:
    """Test suite validating Dockerfile, docker-compose, and Kubernetes manifests."""

    @pytest.fixture(autouse=True)
    def setup_paths(self):
        """Locate root repository directory."""
        self.repo_root = Path(__file__).resolve().parent.parent

    def test_dockerfile_structure_and_hardening(self):
        """Verify Dockerfile uses multi-stage build, non-root user, and healthcheck."""
        dockerfile_path = self.repo_root / "Dockerfile"
        assert dockerfile_path.exists(), "Dockerfile must exist at repository root"

        content = dockerfile_path.read_text(encoding="utf-8")
        assert "FROM python:3.11-slim AS builder" in content
        assert "FROM python:3.11-slim AS runner" in content
        assert "useradd" in content or "adduser" in content
        assert "USER appuser" in content
        assert "EXPOSE 8000" in content
        assert "HEALTHCHECK" in content
        assert "/health" in content

    def test_dockerignore_configuration(self):
        """Verify .dockerignore excludes cache, git, and testing files."""
        dockerignore_path = self.repo_root / ".dockerignore"
        assert dockerignore_path.exists(), ".dockerignore must exist"

        content = dockerignore_path.read_text(encoding="utf-8")
        assert ".git" in content
        assert "__pycache__" in content
        assert ".pytest_cache" in content

    def test_docker_compose_validity(self):
        """Verify docker-compose.yml parses correctly and configures core services."""
        compose_path = self.repo_root / "docker-compose.yml"
        assert compose_path.exists(), "docker-compose.yml must exist"

        with open(compose_path, "r", encoding="utf-8") as f:
            compose_cfg = yaml.safe_load(f)

        assert "services" in compose_cfg
        services = compose_cfg["services"]
        assert "econocausal-api" in services
        assert "prometheus" in services
        assert "grafana" in services
        assert "8000:8000" in services["econocausal-api"]["ports"]
        assert "healthcheck" in services["econocausal-api"]

    def test_prometheus_scrape_config_validity(self):
        """Verify Prometheus configuration targets /metrics."""
        prom_path = self.repo_root / "deploy" / "prometheus" / "prometheus.yml"
        assert prom_path.exists(), "prometheus.yml must exist"

        with open(prom_path, "r", encoding="utf-8") as f:
            prom_cfg = yaml.safe_load(f)

        scrape_configs = prom_cfg.get("scrape_configs", [])
        job_names = [sc.get("job_name") for sc in scrape_configs]
        assert "econocausal-pricing-api" in job_names

    def test_kubernetes_manifests_validity(self):
        """Verify all Kubernetes manifests exist and parse as valid YAML."""
        k8s_dir = self.repo_root / "deploy" / "k8s"
        manifest_files = [
            "namespace.yaml",
            "configmap.yaml",
            "deployment.yaml",
            "service.yaml",
            "hpa.yaml",
            "ingress.yaml",
        ]

        for fname in manifest_files:
            file_path = k8s_dir / fname
            assert file_path.exists(), f"Kubernetes manifest {fname} missing"
            with open(file_path, "r", encoding="utf-8") as f:
                docs = list(yaml.safe_load_all(f))
                assert len(docs) > 0, f"Manifest {fname} produced no YAML documents"

        # Check Deployment specifics
        dep_path = k8s_dir / "deployment.yaml"
        with open(dep_path, "r", encoding="utf-8") as f:
            dep_doc = yaml.safe_load(f)

        assert dep_doc["kind"] == "Deployment"
        assert dep_doc["spec"]["replicas"] == 3
        container = dep_doc["spec"]["template"]["spec"]["containers"][0]
        assert container["name"] == "econocausal-api"
        assert "livenessProbe" in container
        assert "readinessProbe" in container
        assert container["livenessProbe"]["httpGet"]["path"] == "/health"
        assert container["readinessProbe"]["httpGet"]["path"] == "/health"

    def test_github_actions_workflow_validity(self):
        """Verify GitHub Actions CI/CD workflow YAML structure."""
        ci_path = self.repo_root / ".github" / "workflows" / "ci.yml"
        assert ci_path.exists(), "GitHub Actions workflow ci.yml must exist"

        with open(ci_path, "r", encoding="utf-8") as f:
            ci_cfg = yaml.safe_load(f)

        assert "jobs" in ci_cfg
        jobs = ci_cfg["jobs"]
        assert "lint-and-validate" in jobs
        assert "test-and-verify" in jobs
        assert "docker-build-and-smoke-test" in jobs
