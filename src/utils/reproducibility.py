"""Reproducibility and Dependency Version Management.

Provides deterministic random seed configuration, system environment recording,
cryptographic dataset hashing, and dependency version auditing for EconoCausal models.
"""

import hashlib
import importlib
import os
import platform
import random
import subprocess
import sys
from typing import Any, Dict, List, Optional, Union
import numpy as np
import pandas as pd


# Critical libraries tracked for causal model training and inference reproducibility
CRITICAL_DEPENDENCIES: List[str] = [
    "numpy",
    "pandas",
    "scipy",
    "sklearn",  # scikit-learn
    "joblib",
    "statsmodels",
    "pydantic",
    "fastapi",
    "doubleml",
    "econml",
    "streamlit",
    "plotly",
]


def set_seed(seed: int = 42) -> None:
    """Fix random seeds across standard Python libraries and numerical engines.

    Parameters
    ----------
    seed : int, default=42
        Integer random seed for deterministic execution.
    """
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)

    # Optional PyTorch determinism if installed
    try:
        import torch  # type: ignore

        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    except ImportError:
        pass


def get_dependency_versions(
    packages: Optional[List[str]] = None,
) -> Dict[str, str]:
    """Record installed versions of key project dependencies.

    Parameters
    ----------
    packages : Optional[List[str]], default=None
        Custom list of package names to inspect. If None, uses CRITICAL_DEPENDENCIES.

    Returns
    -------
    Dict[str, str]
        Mapping of package name to its detected version string, or 'not_installed'.
    """
    target_pkgs = packages if packages is not None else CRITICAL_DEPENDENCIES
    versions: Dict[str, str] = {
        "python": sys.version.split()[0],
    }

    for pkg_name in target_pkgs:
        display_name = "scikit-learn" if pkg_name == "sklearn" else pkg_name
        try:
            mod = importlib.import_module(pkg_name)
            ver = getattr(mod, "__version__", "unknown")
            versions[display_name] = str(ver)
        except ImportError:
            versions[display_name] = "not_installed"
        except Exception as e:
            versions[display_name] = f"error: {e}"

    return versions


def get_system_info() -> Dict[str, Any]:
    """Capture runtime operating system, hardware, and architecture parameters.

    Returns
    -------
    Dict[str, Any]
        Structured system profile dictionary.
    """
    return {
        "os_name": os.name,
        "platform": platform.platform(),
        "system": platform.system(),
        "release": platform.release(),
        "machine": platform.machine(),
        "processor": platform.processor(),
        "cpu_count": os.cpu_count() or 1,
        "python_implementation": platform.python_implementation(),
        "python_version": platform.python_version(),
        "python_compiler": platform.python_compiler(),
    }


def get_git_info() -> Dict[str, Any]:
    """Extract current git repository commit SHA, branch, and working tree cleanliness.

    Returns
    -------
    Dict[str, Any]
        Git metadata dictionary. If not in a git repo or git is unavailable,
        returns graceful fallback values.
    """
    info: Dict[str, Any] = {
        "commit_sha": "unknown",
        "branch": "unknown",
        "is_dirty": False,
        "available": False,
    }

    try:
        # Check git commit
        res_sha = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
            timeout=5,
        )
        if res_sha.returncode == 0:
            info["commit_sha"] = res_sha.stdout.strip()
            info["available"] = True

        # Check git branch
        res_branch = subprocess.run(
            ["git", "rev-parse", "--abbrev-ref", "HEAD"],
            capture_output=True,
            text=True,
            timeout=5,
        )
        if res_branch.returncode == 0:
            info["branch"] = res_branch.stdout.strip()

        # Check dirty status
        res_status = subprocess.run(
            ["git", "status", "--porcelain"],
            capture_output=True,
            text=True,
            timeout=5,
        )
        if res_status.returncode == 0:
            info["is_dirty"] = len(res_status.stdout.strip()) > 0
    except Exception:
        pass

    return info


def compute_data_hash(
    data: Union[str, pd.DataFrame, np.ndarray, bytes],
) -> str:
    """Calculate a deterministic cryptographic SHA-256 hash for datasets.

    Parameters
    ----------
    data : Union[str, pd.DataFrame, np.ndarray, bytes]
        File path string, DataFrame, NumPy array, or raw bytes.

    Returns
    -------
    str
        64-character hexadecimal SHA-256 checksum string.
    """
    hasher = hashlib.sha256()

    if isinstance(data, str):
        if os.path.isfile(data):
            # Stream read file in 64KB chunks
            with open(data, "rb") as f:
                while True:
                    chunk = f.read(65536)
                    if not chunk:
                        break
                    hasher.update(chunk)
            return hasher.hexdigest()
        else:
            hasher.update(data.encode("utf-8"))
            return hasher.hexdigest()

    elif isinstance(data, pd.DataFrame):
        # Deterministically hash DataFrame rows and column metadata
        try:
            series_hashes = pd.util.hash_pandas_object(data, index=True)
            hasher.update(series_hashes.values.tobytes())
            hasher.update(str(list(data.columns)).encode("utf-8"))
        except Exception:
            # Fallback to deterministic CSV representation
            csv_bytes = data.to_csv(index=False).encode("utf-8")
            hasher.update(csv_bytes)
        return hasher.hexdigest()

    elif isinstance(data, np.ndarray):
        hasher.update(data.tobytes())
        hasher.update(str(data.shape).encode("utf-8"))
        hasher.update(str(data.dtype).encode("utf-8"))
        return hasher.hexdigest()

    elif isinstance(data, (bytes, bytearray)):
        hasher.update(data)
        return hasher.hexdigest()

    else:
        hasher.update(str(data).encode("utf-8"))
        return hasher.hexdigest()


def compute_array_hash(arr: Union[np.ndarray, List[Any]]) -> str:
    """Calculate deterministic SHA-256 hash for split indices or numeric arrays."""
    np_arr = np.asarray(arr)
    return compute_data_hash(np_arr)
