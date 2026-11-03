"""Data loading, schema validation, and causal feature extraction module.

Configured for the official Kaggle dataset:
'Retail Price Optimization' (retail_price.csv)
URL: https://www.kaggle.com/datasets/suddharshan/retail-price-optimization
"""

import os
from typing import Any, Dict, List, Optional, Tuple
import numpy as np
import pandas as pd


# Canonical columns and causal roles defined in docs/data_dictionary.md
DEFAULT_TREATMENT_COL = "unit_price"
DEFAULT_OUTCOME_COL = "qty"
DEFAULT_CONFOUNDER_COLS = [
    "comp_1",
    "comp_2",
    "comp_3",
    "fp1",
    "fp2",
    "fp3",
    "product_score",
    "customers",
    "s",
    "weekday",
    "holiday",
    "lag_price",
    "freight_price",
]

# Post-treatment or outcome variables that must never leak into confounder matrix X
EXCLUDED_FROM_CONFOUNDERS = {
    "qty",
    "units_sold",
    "total_price",
    "revenue",
    "profit",
}


def load_dataset(filepath: str) -> pd.DataFrame:
    """Load retail transaction dataset from CSV or Parquet file.

    Parameters
    ----------
    filepath : str
        Path to dataset file (.csv or .parquet).

    Returns
    -------
    pd.DataFrame
        Loaded raw DataFrame.

    Raises
    ------
    FileNotFoundError
        If file does not exist.
    ValueError
        If unsupported file format.
    """
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"File not found: {filepath}")

    if filepath.endswith(".csv"):
        return pd.read_csv(filepath)
    elif filepath.endswith(".parquet") or filepath.endswith(".pq"):
        return pd.read_parquet(filepath)
    else:
        raise ValueError(f"Unsupported file format for {filepath}. Expected .csv or .parquet")


def load_retail_price_dataset(filepath: str = "data/raw/retail_price.csv") -> pd.DataFrame:
    """Load and validate the official Kaggle Retail Price Optimization dataset.

    Parameters
    ----------
    filepath : str
        Path to retail_price.csv.

    Returns
    -------
    pd.DataFrame
        Cleaned, validated retail price DataFrame with standardized aliases.
    """
    df = load_dataset(filepath)

    # Standardize aliases for backward compatibility with backend
    df["price"] = df["unit_price"]
    df["units_sold"] = df["qty"]
    df["revenue"] = df["total_price"]

    validation = validate_schema(
        df,
        treatment_col="unit_price",
        outcome_col="qty",
        confounder_cols=DEFAULT_CONFOUNDER_COLS,
    )
    if not validation["is_valid"]:
        raise ValueError(f"Retail price dataset validation failed: {validation['errors']}")

    return df


def validate_schema(
    df: pd.DataFrame,
    treatment_col: str = DEFAULT_TREATMENT_COL,
    outcome_col: str = DEFAULT_OUTCOME_COL,
    confounder_cols: Optional[List[str]] = None,
) -> Dict[str, Any]:
    """Validate DataFrame schema, types, and causal constraints.

    Parameters
    ----------
    df : pd.DataFrame
        Dataset to validate.
    treatment_col : str
        Name of treatment column (e.g., 'unit_price').
    outcome_col : str
        Name of outcome column (e.g., 'qty').
    confounder_cols : List[str], optional
        List of pre-treatment confounder columns.

    Returns
    -------
    Dict[str, Any]
        Dictionary with validation results:
        - 'is_valid': bool
        - 'errors': list of fatal errors
        - 'warnings': list of non-fatal warnings
        - 'duplicate_count': int
        - 'row_count': int
    """
    if confounder_cols is None:
        confounder_cols = DEFAULT_CONFOUNDER_COLS

    errors: List[str] = []
    warnings: List[str] = []

    if df.empty:
        return {
            "is_valid": False,
            "errors": ["DataFrame is empty."],
            "warnings": [],
            "duplicate_count": 0,
            "row_count": 0,
        }

    # 1. Required column presence
    required_cols = [treatment_col, outcome_col] + confounder_cols
    missing_cols = [col for col in required_cols if col not in df.columns]
    if missing_cols:
        errors.append(f"Missing required columns: {missing_cols}")

    # If critical columns missing, return early
    if errors:
        return {
            "is_valid": False,
            "errors": errors,
            "warnings": warnings,
            "duplicate_count": int(df.duplicated().sum()),
            "row_count": len(df),
        }

    # 2. Check for post-treatment leakage in confounders
    leakage = [col for col in confounder_cols if col in EXCLUDED_FROM_CONFOUNDERS]
    if leakage:
        errors.append(f"Post-treatment leakage detected in confounders: {leakage}")

    # 3. Treatment constraints: price must be strictly positive
    if treatment_col in ["unit_price", "price"]:
        invalid_treatment = (df[treatment_col] <= 0).sum()
        if invalid_treatment > 0:
            errors.append(
                f"Found {invalid_treatment} rows with {treatment_col} <= 0. Price must be strictly positive."
            )

    # 4. Outcome constraints: qty must be non-negative
    if outcome_col in ["qty", "units_sold"]:
        invalid_outcome = (df[outcome_col] < 0).sum()
        if invalid_outcome > 0:
            errors.append(f"Found {invalid_outcome} rows with {outcome_col} < 0.")

    # 5. Check competitor price and freight non-negativity
    for comp in ["comp_1", "comp_2", "comp_3"]:
        if comp in df.columns and (df[comp] <= 0).sum() > 0:
            warnings.append(f"Found rows with {comp} <= 0.")

    for fp in ["freight_price", "fp1", "fp2", "fp3"]:
        if fp in df.columns and (df[fp] < 0).sum() > 0:
            errors.append(f"Found rows with negative {fp}.")

    # 6. Missing value checks
    missing_summary = df[required_cols].isnull().sum()
    cols_with_nulls = missing_summary[missing_summary > 0]
    if not cols_with_nulls.empty:
        warnings.append(f"Columns containing null values: {cols_with_nulls.to_dict()}")

    # 7. Duplicate check
    duplicate_count = int(df.duplicated().sum())
    if duplicate_count > 0:
        warnings.append(f"Found {duplicate_count} duplicate rows in dataset.")

    is_valid = len(errors) == 0

    return {
        "is_valid": is_valid,
        "errors": errors,
        "warnings": warnings,
        "duplicate_count": duplicate_count,
        "row_count": len(df),
    }


def extract_causal_matrices(
    df: pd.DataFrame,
    treatment_col: str = DEFAULT_TREATMENT_COL,
    outcome_col: str = DEFAULT_OUTCOME_COL,
    confounder_cols: Optional[List[str]] = None,
    drop_na: bool = True,
) -> Tuple[pd.DataFrame, pd.Series, pd.Series]:
    """Extract and validate (X, T, Y) matrices for causal DML modeling.

    Parameters
    ----------
    df : pd.DataFrame
        Validated source DataFrame.
    treatment_col : str
        Treatment variable name.
    outcome_col : str
        Outcome variable name.
    confounder_cols : List[str], optional
        List of confounder names.
    drop_na : bool
        Whether to drop rows with NaN values in (X, T, Y).

    Returns
    -------
    Tuple[pd.DataFrame, pd.Series, pd.Series]
        (X, T, Y) where X is confounder DataFrame, T is treatment Series, Y is outcome Series.

    Raises
    ------
    ValueError
        If schema validation fails or post-treatment leakage is detected.
    """
    if confounder_cols is None:
        confounder_cols = DEFAULT_CONFOUNDER_COLS

    validation = validate_schema(
        df,
        treatment_col=treatment_col,
        outcome_col=outcome_col,
        confounder_cols=confounder_cols,
    )
    if not validation["is_valid"]:
        raise ValueError(f"Schema validation failed: {validation['errors']}")

    cols_to_extract = confounder_cols + [treatment_col, outcome_col]
    subset = df[cols_to_extract].copy()

    if drop_na:
        subset = subset.dropna()

    X = subset[confounder_cols]
    T = subset[treatment_col]
    Y = subset[outcome_col]

    return X, T, Y


def generate_synthetic_retail_data(
    n_samples: int = 500,
    base_price: float = 100.0,
    true_elasticity: float = -1.75,
    random_seed: int = 42,
) -> pd.DataFrame:
    """Generate reproducible synthetic retail transactions matching schema for unit tests."""
    rng = np.random.default_rng(random_seed)

    comp_1 = np.round(base_price * (0.90 + 0.20 * rng.uniform(size=n_samples)), 2)
    comp_2 = np.round(base_price * (0.85 + 0.30 * rng.uniform(size=n_samples)), 2)
    comp_3 = np.round(base_price * (0.80 + 0.35 * rng.uniform(size=n_samples)), 2)

    fp1 = np.round(10.0 + 5.0 * rng.uniform(size=n_samples), 2)
    fp2 = np.round(10.0 + 5.0 * rng.uniform(size=n_samples), 2)
    fp3 = np.round(10.0 + 5.0 * rng.uniform(size=n_samples), 2)
    freight_price = np.round(12.0 + 4.0 * rng.uniform(size=n_samples), 2)

    product_score = np.round(3.5 + 1.5 * rng.uniform(size=n_samples), 1)
    customers = rng.integers(10, 200, size=n_samples)
    s = np.round(1.0 + 0.3 * np.sin(np.linspace(0, 4 * np.pi, n_samples)), 2)
    weekday = rng.integers(15, 23, size=n_samples)
    holiday = rng.integers(0, 5, size=n_samples)
    lag_price = np.round(base_price * (0.95 + 0.1 * rng.uniform(size=n_samples)), 2)

    price_jitter = rng.normal(0, 0.05, size=n_samples)
    unit_price = np.round(base_price * (1.0 + price_jitter), 2)
    unit_price = np.clip(unit_price, base_price * 0.5, base_price * 1.8)

    relative_price = unit_price / base_price
    demand = 80.0 * np.power(relative_price, true_elasticity) * s * rng.lognormal(0, 0.1, n_samples)
    qty = np.maximum(1, np.round(demand)).astype(int)
    total_price = np.round(unit_price * qty, 2)

    df = pd.DataFrame(
        {
            "product_id": ["bed1"] * n_samples,
            "product_category_name": ["bed_bath_table"] * n_samples,
            "month_year": ["01-05-2017"] * n_samples,
            "unit_price": unit_price,
            "qty": qty,
            "total_price": total_price,
            "comp_1": comp_1,
            "comp_2": comp_2,
            "comp_3": comp_3,
            "fp1": fp1,
            "fp2": fp2,
            "fp3": fp3,
            "product_score": product_score,
            "customers": customers,
            "s": s,
            "weekday": weekday,
            "holiday": holiday,
            "lag_price": lag_price,
            "freight_price": freight_price,
        }
    )

    # Aliases
    df["price"] = df["unit_price"]
    df["units_sold"] = df["qty"]
    df["revenue"] = df["total_price"]

    return df


def summarize_data_quality(df: pd.DataFrame) -> Dict[str, Any]:
    """Produce summary statistics and data quality diagnostics for dataset."""
    summary = {
        "total_records": len(df),
        "total_columns": len(df.columns),
        "null_counts": df.isnull().sum().to_dict(),
        "duplicates": int(df.duplicated().sum()),
    }

    price_col = "unit_price" if "unit_price" in df.columns else ("price" if "price" in df.columns else None)
    qty_col = "qty" if "qty" in df.columns else ("units_sold" if "units_sold" in df.columns else None)

    if price_col:
        summary["price_summary"] = {
            "min": float(df[price_col].min()),
            "median": float(df[price_col].median()),
            "mean": float(df[price_col].mean()),
            "max": float(df[price_col].max()),
            "std": float(df[price_col].std()),
        }

    if qty_col:
        summary["qty_summary"] = {
            "min": int(df[qty_col].min()),
            "median": float(df[qty_col].median()),
            "mean": float(df[qty_col].mean()),
            "max": int(df[qty_col].max()),
        }

    return summary
