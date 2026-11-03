"""Launcher script for EconoCausal Streamlit Dynamic Pricing Dashboard.

Usage:
    python scripts/run_dashboard.py [--port 8501] [--host 0.0.0.0]
"""

import argparse
import os
import subprocess
import sys

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
DASHBOARD_APP = os.path.join(PROJECT_ROOT, "src", "dashboard", "app.py")
DEFAULT_MODEL = os.path.join(PROJECT_ROOT, "models", "econocausal_pipeline.joblib")
DEFAULT_DATA = os.path.join(PROJECT_ROOT, "data", "raw", "retail_price.csv")


def main():
    parser = argparse.ArgumentParser(description="Launch EconoCausal Streamlit Dashboard")
    parser.add_argument("--port", type=int, default=8501, help="Port to run Streamlit server on")
    parser.add_argument("--host", type=str, default="localhost", help="Host address")
    parser.add_argument("--browser", action="store_true", help="Automatically open browser window")
    args = parser.parse_args()

    # Verify model artifact
    if not os.path.exists(DEFAULT_MODEL):
        print(f"Warning: Model artifact not found at {DEFAULT_MODEL}")
        print("Training model pipeline first...")
        cmd_train = [
            sys.executable,
            "-m",
            "src.cli",
            "train",
            "--data",
            DEFAULT_DATA,
            "--save-model",
            DEFAULT_MODEL,
        ]
        subprocess.check_call(cmd_train, cwd=PROJECT_ROOT)

    cmd = [
        sys.executable,
        "-m",
        "streamlit",
        "run",
        DASHBOARD_APP,
        "--server.port",
        str(args.port),
        "--server.address",
        args.host,
    ]

    if not args.browser:
        cmd.extend(["--server.headless", "true"])

    print(f"Starting EconoCausal Streamlit Dashboard on http://{args.host}:{args.port}")
    try:
        subprocess.run(cmd, cwd=PROJECT_ROOT)
    except KeyboardInterrupt:
        print("\nStopping dashboard server.")


if __name__ == "__main__":
    main()
