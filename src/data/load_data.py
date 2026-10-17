"""Data loading, schema validation, and causal feature extraction module.

Adheres to the causal variable contract specified in docs/data_dictionary.md
and the backend schema (SalesRecord).
"""

from typing import Any, Dict, List, Optional, Tuple
import os
import numpy as np
import pandas as pd



# Canonical columns and causal roles defined in docs/data_dictionary.md
DEFAULT_TREATMENT_COL = "price"
DEFAULT_OUTCOME_COL = "units_sold"
DEFAULT_CONFOUNDER_COLS = [
    "competitor_price",
    "is_promotion",
    "seasonality_index",
    "inventory_level",
    "day_of_week",
]

# Post-treatment or outcome variables that must never leak into confounder matrix X
EXCLUDED_FROM_CONFOUNDERS = {
    "units_sold",
    "revenue",
    "profit",
    "return_rate",
    "customer_rating",
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
    if filepath.endswith(".csv"):
        return pd.read_csv(filepath)
    elif filepath.endswith(".parquet") or filepath.endswith(".pq"):
        return pd.read_parquet(filepath)
    else:
        raise ValueError(f"Unsupported file format for {filepath}. Expected .csv or .parquet")


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
        Name of treatment column (e.g., 'price' or 'discount_pct').
    outcome_col : str
        Name of outcome column (e.g., 'units_sold').
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

    # If critical columns missing, cannot perform value checks safely
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
    if treatment_col == "price":
        invalid_treatment = (df[treatment_col] <= 0).sum()
        if invalid_treatment > 0:
            errors.append(f"Found {invalid_treatment} rows with price <= 0. Price must be strictly positive.")
    elif treatment_col == "discount_pct":
        invalid_discount = ((df[treatment_col] < 0) | (df[treatment_col] > 100)).sum()
        if invalid_discount > 0:
            errors.append(f"Found {invalid_discount} rows with discount_pct outside [0, 100].")

    # 4. Outcome constraints: units_sold must be non-negative
    if outcome_col == "units_sold":
        invalid_outcome = (df[outcome_col] < 0).sum()
        if invalid_outcome > 0:
            errors.append(f"Found {invalid_outcome} rows with units_sold < 0.")

    # 5. Check competitor price and inventory non-negativity
    if "competitor_price" in df.columns and (df["competitor_price"] < 0).sum() > 0:
        errors.append("Found rows with negative competitor_price.")
    if "inventory_level" in df.columns and (df["inventory_level"] < 0).sum() > 0:
        errors.append("Found rows with negative inventory_level.")

    # 6. Seasonality index positivity
    if "seasonality_index" in df.columns and (df["seasonality_index"] <= 0).sum() > 0:
        warnings.append("Found rows with seasonality_index <= 0. Seasonality index should be positive.")

    # 7. Missing value checks
    missing_summary = df[required_cols].isnull().sum()
    cols_with_nulls = missing_summary[missing_summary > 0]
    if not cols_with_nulls.empty:
        warnings.append(f"Columns containing null values: {cols_with_nulls.to_dict()}")

    # 8. Duplicate check
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
    """Generate reproducible synthetic retail transactions matching backend schema.

    Follows econometric demand-price relationships:
    Q = Q0 * (P / P0)^true_elasticity * Confounder_effects * Noise

    Used for testing, baseline pipeline verification, and benchmark validation
    until real transaction datasets are provided.

    Parameters
    ----------
    n_samples : int
        Number of transaction records to generate.
    base_price : float
        Reference price for normalization.
    true_elasticity : float
        Target ground-truth price elasticity for causal verification.
    random_seed : int
        Seed for reproducibility.

    Returns
    -------
    pd.DataFrame
        Simulated retail transaction DataFrame adhering to docs/data_dictionary.md.
    """
    rng = np.random.default_rng(random_seed)

    # 1. Confounders (X)
    is_promotion = rng.binomial(1, 0.2, size=n_samples)
    seasonality_index = np.round(1.0 + 0.25 * np.sin(np.linspace(0, 4 * np.pi, n_samples)), 2)
    competitor_price = np.round(base_price * (0.90 + 0.20 * rng.uniform(size=n_samples)), 2)
    inventory_level = rng.integers(50, 400, size=n_samples)
    day_of_week = rng.integers(0, 7, size=n_samples)

    # 2. Treatment (T = price) influenced by confounders (endogeneity/confounding)
    # Retailers typically discount when competitor price is lower, or during promotions
    price_factor = 1.0 - 0.10 * is_promotion + 0.05 * (competitor_price / base_price - 1.0)
    jitter = rng.normal(0, 0.05, size=n_samples)
    price = np.round(base_price * (price_factor + jitter), 2)
    # Ensure price strictly positive
    price = np.clip(price, base_price * 0.5, base_price * 1.8)

    discount_pct = np.where(price < base_price, np.round((1 - price / base_price) * 100, 1), 0.0)

    # 3. Outcome (Y = units_sold)
    # True structural demand function: log(Q) = log(Q0) + elasticity * log(P/P0) + gamma*X + noise
    base_demand = 80.0
    promo_effect = 1.35 * is_promotion + 1.0 * (1 - is_promotion)
    competitor_effect = np.where(price < competitor_price, 1.15, 0.90)
    noise = rng.lognormal(mean=0.0, sigma=0.1, size=n_samples)

    relative_price = price / base_price
    demand = (
        base_demand
        * np.power(relative_price, true_elasticity)
        * promo_effect
        * competitor_effect
        * seasonality_index
        * noise
    )
    units_sold = np.maximum(1, np.round(demand)).astype(int)
    revenue = np.round(price * units_sold, 2)

    df = pd.DataFrame(
        {
            "product_id": ["PROD-TECH-001"] * n_samples,
            "date": pd.date_range("2026-01-01", periods=n_samples, freq="D").strftime("%Y-%m-%d"),
            "price": price,
            "discount_pct": discount_pct,
            "units_sold": units_sold,
            "revenue": revenue,
            "competitor_price": competitor_price,
            "is_promotion": is_promotion,
            "seasonality_index": seasonality_index,
            "inventory_level": inventory_level,
            "day_of_week": day_of_week,
        }
    )

    return df


def summarize_data_quality(df: pd.DataFrame) -> Dict[str, Any]:
    """Produce summary statistics and data quality diagnostics for dataset.

    Parameters
    ----------
    df : pd.DataFrame
        Dataset to analyze.

    Returns
    -------
    Dict[str, Any]
        Diagnostic summary including counts, nulls, price distribution, and correlation.
    """
    summary = {
        "total_records": len(df),
        "total_columns": len(df.columns),
        "null_counts": df.isnull().sum().to_dict(),
        "duplicates": int(df.duplicated().sum()),
    }

    if "price" in df.columns:
        summary["price_summary"] = {
            "min": float(df["price"].min()),
            "median": float(df["price"].median()),
            "mean": float(df["price"].mean()),
            "max": float(df["price"].max()),
            "std": float(df["price"].std()),
        }

    if "units_sold" in df.columns:
        summary["units_sold_summary"] = {
            "min": int(df["units_sold"].min()),
            "median": float(df["units_sold"].median()),
            "mean": float(df["units_sold"].mean()),
            "max": int(df["units_sold"].max()),
        }

    return summary


def load_and_merge_ecommerce_dataset(
    data_dir: str = "data/raw",
    sample_size: Optional[int] = None,
    random_state: int = 42,
) -> pd.DataFrame:
    """Load and merge relational tables from Indian E-Commerce dataset.

    Performs left-joins:
    - order_items (primary transaction grain)
    - orders (order_date, coupon_code, marketing_channel)
    - products (category, brand, cost_price, stock_quantity, rating_average)
    - customers (customer_segment, loyalty_tier, age_group, state)

    Features engineered:
    - day_of_week, month, is_weekend
    - is_coupon_applied, is_promotion
    - standardized aliases: discount_pct, price, units_sold, inventory_level

    Parameters
    ----------
    data_dir : str
        Path to folder with raw CSVs.
    sample_size : Optional[int]
        Number of order_items to sample (useful for quick experiments/testing).
    random_state : int
        Seed for reproducible sampling.

    Returns
    -------
    pd.DataFrame
        Enriched DataFrame ready for preprocessing and causal DML modeling.
    """
    order_items_path = os.path.join(data_dir, "order_items.csv")
    orders_path = os.path.join(data_dir, "orders.csv")
    products_path = os.path.join(data_dir, "products.csv")
    customers_path = os.path.join(data_dir, "customers.csv")

    for path in [order_items_path, orders_path, products_path, customers_path]:
        if not os.path.exists(path):
            raise FileNotFoundError(f"Required table not found at: {path}")

    # 1. Load primary transaction line items
    order_items = pd.read_csv(order_items_path)
    if sample_size is not None and sample_size < len(order_items):
        order_items = order_items.sample(n=sample_size, random_state=random_state).reset_index(drop=True)

    # 2. Load dimension tables
    orders = pd.read_csv(orders_path)
    products = pd.read_csv(products_path)
    customers = pd.read_csv(customers_path)

    # 3. Perform relational merges
    orders_cols = ["order_id", "customer_id", "order_date", "coupon_code", "marketing_channel"]
    products_cols = [
        "product_id",
        "category",
        "subcategory",
        "brand",
        "cost_price",
        "stock_quantity",
        "rating_average",
    ]
    customers_cols = ["customer_id", "customer_segment", "loyalty_tier", "age_group", "state"]

    merged = order_items.merge(orders[orders_cols], on="order_id", how="left")
    merged = merged.merge(products[products_cols], on="product_id", how="left")
    merged = merged.merge(customers[customers_cols], on="customer_id", how="left")

    # 4. Feature engineering for temporal & promotional confounders
    order_dates = pd.to_datetime(merged["order_date"], errors="coerce")
    merged["day_of_week"] = order_dates.dt.dayofweek
    merged["month"] = order_dates.dt.month
    merged["is_weekend"] = merged["day_of_week"].isin([5, 6]).astype(int)

    merged["is_coupon_applied"] = merged["coupon_code"].notna().astype(int)
    merged["is_promotion"] = (
        (merged["discount_percentage"] > 0) | (merged["is_coupon_applied"] == 1)
    ).astype(int)

    # 5. Standardize column names matching data dictionary
    merged["discount_pct"] = merged["discount_percentage"]
    merged["price"] = merged["unit_price"]
    merged["units_sold"] = merged["quantity"]
    merged["inventory_level"] = merged["stock_quantity"]

    return merged

