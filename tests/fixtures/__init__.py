"""Integration fixtures loader for EconoCausal ML engine and backend integration tests."""

import json
import os
from typing import Any, Dict


FIXTURES_DIR = os.path.dirname(__file__)
REQUESTS_PATH = os.path.join(FIXTURES_DIR, "backend_requests.json")
RESPONSES_PATH = os.path.join(FIXTURES_DIR, "backend_responses.json")


def load_request_fixture(name: str) -> Dict[str, Any]:
    """Load a specific request fixture payload by key name."""
    with open(REQUESTS_PATH, "r", encoding="utf-8") as f:
        data = json.load(f)
    if name not in data:
        raise KeyError(f"Request fixture '{name}' not found. Available: {list(data.keys())}")
    return data[name]


def load_response_fixture(name: str) -> Dict[str, Any]:
    """Load a specific response fixture schema by key name."""
    with open(RESPONSES_PATH, "r", encoding="utf-8") as f:
        data = json.load(f)
    if name not in data:
        raise KeyError(f"Response fixture '{name}' not found. Available: {list(data.keys())}")
    return data[name]
