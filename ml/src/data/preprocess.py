"""Data preprocessing, causal feature engineering, and cross-fitting fold partitioning.

Specifically built for causal machine learning and Double Machine Learning (DML)
estimation on retail pricing and demand data.

Implements:
- Comprehensive confounder audit and pre-treatment vs. post-treatment separation
- Domain-justified causal feature engineering (competitor-price gap, discount history,
  cyclical seasonality, customer traffic density, listing characteristics)
- Strict anti-leakage guardrails (treatment leakage, target leakage, collider bias)
- Missing-data imputation fitted strictly on training folds
- Robust categorical encoding handling unseen categories
- Schema consistency guarantees (column ordering, missing/extra column resilience)
"""

from typing import Any, Dict, List, Optional, Set, Tuple
import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.model_selection import KFold, train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from src.data.load_data import (
    DEFAULT_CONFOUNDER_COLS,
    DEFAULT_OUTCOME_COL,
    DEFAULT_TREATMENT_COL,
    EXCLUDED_FROM_CONFOUNDERS,
    extract_causal_matrices,
)


class CausalLeakageError(ValueError):
    """Raised when target, treatment, collider, or post-treatment variables leak into confounders."""

    pass


# Canonical sets for causal variable separation
POST_TREATMENT_VARIABLES: Set[str] = {
    "qty",
    "units_sold",
    "total_price",
    "revenue",
    "profit",
    "gross_profit",
    "sales",
    "conversion_rate",
    "cart_adds",
    "post_treatment",
    "purchased",
}

TREATMENT_VARIABLES: Set[str] = {
    "unit_price",
    "price",
    "log_unit_price",
    "log_price",
}

# Systematic audit of the 13 canonical baseline confounders
CONFOUNDER_AUDIT_CATALOG: Dict[str, Dict[str, str]] = {
    "comp_1": {
        "role": "Market Competition (Direct)",
        "status": "Pre-treatment",
        "category": "Competitor Price",
        "description": "Primary competitor selling price. Influences pricing strategy and consumer substitution.",
    },
    "comp_2": {
        "role": "Market Competition (Secondary)",
        "status": "Pre-treatment",
        "category": "Competitor Price",
        "description": "Secondary competitor selling price. Controls for wider market price distribution.",
    },
    "comp_3": {
        "role": "Market Competition (Tertiary)",
        "status": "Pre-treatment",
        "category": "Competitor Price",
        "description": "Tertiary competitor selling price. Expands competitive horizon.",
    },
    "fp1": {
        "role": "Competitor Shipping Cost",
        "status": "Pre-treatment",
        "category": "Competitor Freight",
        "description": "Competitor 1 freight price. Affects competitor total landed price.",
    },
    "fp2": {
        "role": "Competitor Shipping Cost",
        "status": "Pre-treatment",
        "category": "Competitor Freight",
        "description": "Competitor 2 freight price. Shipping friction benchmark.",
    },
    "fp3": {
        "role": "Competitor Shipping Cost",
        "status": "Pre-treatment",
        "category": "Competitor Freight",
        "description": "Competitor 3 freight price. Shipping friction benchmark.",
    },
    "product_score": {
        "role": "Product Reputation",
        "status": "Pre-treatment",
        "category": "Product Quality",
        "description": "Customer review rating score (1.0 - 5.0). Reflects product quality and brand equity.",
    },
    "customers": {
        "role": "Market Potential",
        "status": "Pre-treatment",
        "category": "Customer Footprint",
        "description": "Store traffic / unique customer interactions during period. Controls for market size.",
    },
    "s": {
        "role": "Seasonality Index",
        "status": "Pre-treatment",
        "category": "Seasonality",
        "description": "Macro sales demand trend index. Controls for cyclic macro demand fluctuations.",
    },
    "weekday": {
        "role": "Calendar Composition",
        "status": "Pre-treatment",
        "category": "Calendar",
        "description": "Count of business days in observation period. Drives weekday purchasing patterns.",
    },
    "holiday": {
        "role": "Calendar Holiday Spike",
        "status": "Pre-treatment",
        "category": "Calendar",
        "description": "Count of public holidays in observation period. Controls for promotional calendar shifts.",
    },
    "lag_price": {
        "role": "Historical Price Anchor",
        "status": "Pre-treatment",
        "category": "Price History",
        "description": "Prior period unit price. Controls for price stickiness and consumer reference anchoring.",
    },
    "freight_price": {
        "role": "Direct Shipping Cost",
        "status": "Pre-treatment",
        "category": "Store Logistics",
        "description": "Merchant freight price charged to buyer. Drives cost floor and landed price perception.",
    },
}


def audit_confounders(
    df: Optional[pd.DataFrame] = None,
    confounder_cols: Optional[List[str]] = None,
    treatment_col: str = DEFAULT_TREATMENT_COL,
    outcome_col: str = DEFAULT_OUTCOME_COL,
    raise_on_leakage: bool = True,
) -> Dict[str, Any]:
    """Audit confounder columns for causal validity and anti-leakage compliance.

    Separates strictly pre-treatment confounders from variables that may be
    consequences of the price intervention (mediators, colliders, outcome targets,
    or treatment variables).

    Parameters
    ----------
    df : Optional[pd.DataFrame]
        DataFrame to inspect (optional, if column correlation/data checks desired).
    confounder_cols : Optional[List[str]]
        List of confounder columns to evaluate. If None, uses DEFAULT_CONFOUNDER_COLS.
    treatment_col : str
        Treatment variable name (e.g. 'unit_price').
    outcome_col : str
        Outcome variable name (e.g. 'qty').
    raise_on_leakage : bool
        If True, raises CausalLeakageError when leakage is detected.

    Returns
    -------
    Dict[str, Any]
        Audit report containing:
        - 'is_clean': bool
        - 'confounders': list of valid pre-treatment confounders
        - 'treatment_leakage': list of detected treatment leaks
        - 'outcome_leakage': list of detected target/outcome leaks
        - 'post_treatment_leakage': list of detected post-treatment / collider leaks
        - 'audit_details': metadata and causal justification for each variable

    Raises
    ------
    CausalLeakageError
        If raise_on_leakage is True and any form of leakage is detected.
    """
    if confounder_cols is None:
        confounder_cols = list(DEFAULT_CONFOUNDER_COLS)

    treatment_leaks: List[str] = []
    outcome_leaks: List[str] = []
    post_treatment_leaks: List[str] = []
    valid_confounders: List[str] = []
    audit_details: Dict[str, Dict[str, str]] = {}

    for col in confounder_cols:
        col_lower = col.lower()

        # 1. Direct Treatment Leakage Check
        if col == treatment_col or col_lower in TREATMENT_VARIABLES or "unit_price" in col_lower:
            treatment_leaks.append(col)
            audit_details[col] = {
                "role": "Treatment Variable (Violation)",
                "status": "FATAL LEAKAGE: Treatment",
                "justification": f"'{col}' is or directly contains the treatment. Causes circularity and collinearity.",
            }
            continue

        # 2. Direct Outcome / Target Leakage Check
        if col == outcome_col or col_lower in {"qty", "units_sold"}:
            outcome_leaks.append(col)
            audit_details[col] = {
                "role": "Outcome Variable (Violation)",
                "status": "FATAL LEAKAGE: Outcome",
                "justification": f"'{col}' is the target demand variable. Induces perfect target leakage.",
            }
            continue

        # 3. Post-treatment Mediators / Colliders Check
        if col_lower in POST_TREATMENT_VARIABLES or col in EXCLUDED_FROM_CONFOUNDERS:
            post_treatment_leaks.append(col)
            audit_details[col] = {
                "role": "Post-Treatment Collider / Mediator (Violation)",
                "status": "FATAL LEAKAGE: Post-Treatment",
                "justification": (
                    f"'{col}' is a post-treatment outcome or collider (e.g. revenue = price * qty). "
                    "Conditioning induces Berkson's bias or mediator blockage."
                ),
            }
            continue

        # 4. Check for suspicious derived post-treatment columns (e.g. 'discount_current', 'price_diff_unit')
        if "current" in col_lower and "discount" in col_lower:
            treatment_leaks.append(col)
            audit_details[col] = {
                "role": "Post-Treatment Discount Leakage (Violation)",
                "status": "FATAL LEAKAGE: Derived Treatment",
                "justification": "Current period discount is derived directly from unit_price intervention.",
            }
            continue

        # Valid pre-treatment confounder
        valid_confounders.append(col)
        if col in CONFOUNDER_AUDIT_CATALOG:
            audit_details[col] = CONFOUNDER_AUDIT_CATALOG[col]
        else:
            audit_details[col] = {
                "role": "Pre-treatment Covariate",
                "status": "Pre-treatment",
                "justification": "Verified pre-treatment variable determined prior to pricing intervention.",
            }

    # Optional empirical statistical check if DataFrame is provided
    if df is not None:
        for col in valid_confounders:
            if col in df.columns:
                # Check for numerical identity or perfect correlation with treatment
                if treatment_col in df.columns and pd.api.types.is_numeric_dtype(df[col]):
                    if df[col].equals(df[treatment_col]):
                        treatment_leaks.append(col)
                        valid_confounders.remove(col)
                        audit_details[col] = {
                            "role": "Empirical Treatment Identity (Violation)",
                            "status": "FATAL LEAKAGE: Treatment",
                            "justification": f"'{col}' is empirically identical to treatment '{treatment_col}'.",
                        }
                    elif df[col].std() > 1e-9 and df[treatment_col].std() > 1e-9:
                        corr = abs(df[col].corr(df[treatment_col]))
                        if corr > 0.99999:
                            treatment_leaks.append(col)
                            valid_confounders.remove(col)
                            audit_details[col] = {
                                "role": "Empirical Collinearity Leakage (Violation)",
                                "status": "FATAL LEAKAGE: Near-Perfect Collinearity",
                                "justification": f"'{col}' has Pearson |r| = {corr:.6f} with treatment. Deterministic leak.",
                            }

    is_clean = len(treatment_leaks) == 0 and len(outcome_leaks) == 0 and len(post_treatment_leaks) == 0

    if raise_on_leakage and not is_clean:
        error_msgs = []
        if treatment_leaks:
            error_msgs.append(f"Treatment leakage detected in confounders: {treatment_leaks}")
        if outcome_leaks:
            error_msgs.append(f"Outcome target leakage detected in confounders: {outcome_leaks}")
        if post_treatment_leaks:
            error_msgs.append(f"Post-treatment collider/mediator leakage detected: {post_treatment_leaks}")
        raise CausalLeakageError(" | ".join(error_msgs))

    return {
        "is_clean": is_clean,
        "confounders": valid_confounders,
        "treatment_leakage": treatment_leaks,
        "outcome_leakage": outcome_leaks,
        "post_treatment_leakage": post_treatment_leaks,
        "audit_details": audit_details,
    }


def engineer_causal_features(
    df: pd.DataFrame,
    reference_prices: Optional[Dict[str, float]] = None,
    include_base_confounders: bool = True,
    drop_leakage: bool = True,
) -> pd.DataFrame:
    """Engineer justified pre-treatment causal features for DML estimation.

    Produces:
    1. Pre-treatment Competitor-Price Gaps (lag_price - comp_1, comp_price_ratio_1,
       comp_price_min, comp_price_max, comp_price_mean, comp_price_spread, lag_price_diff_comp_min)
    2. Seasonality & Cyclical Dynamics (sin_month, cos_month, is_q4, weekend_ratio, holiday_ratio)
    3. Pre-treatment Discount History (historical_discount_rate, is_lag_discounted, lag_price_ratio_comp_mean)
    4. Product & Customer Characteristics (customers_per_day, product_photos_qty,
       product_weight_g, volume, product_description_lenght)
    5. Product Category (product_category_name)

    GUARANTEE: Uses only pre-treatment information (lag_price, competitor benchmarks,
    calendar, product metadata). Current unit_price is NEVER accessed during
    confounder engineering to prevent treatment leakage.

    Parameters
    ----------
    df : pd.DataFrame
        Raw or preprocessed dataset containing transaction/item features.
    reference_prices : Optional[Dict[str, float]]
        Optional mapping of product_id to baseline reference / MSRP price.
    include_base_confounders : bool
        Whether to retain the original baseline confounders alongside engineered features.
    drop_leakage : bool
        Whether to automatically exclude known post-treatment and treatment variables.

    Returns
    -------
    pd.DataFrame
        Engineered causal features DataFrame.
    """
    feat = pd.DataFrame(index=df.index)

    # 1. Base Confounder Retention (if requested and present)
    if include_base_confounders:
        for col in DEFAULT_CONFOUNDER_COLS:
            if col in df.columns:
                feat[col] = df[col]

    # Retain product category if present
    if "product_category_name" in df.columns:
        feat["product_category_name"] = df["product_category_name"].astype(str)

    # -------------------------------------------------------------------------
    # 2. Competitor-Price Gap & Dispersion (Strictly Pre-Treatment via lag_price)
    # -------------------------------------------------------------------------
    comp_cols = [c for c in ["comp_1", "comp_2", "comp_3"] if c in df.columns]
    has_lag = "lag_price" in df.columns

    if comp_cols:
        comp_min = df[comp_cols].min(axis=1)
        comp_max = df[comp_cols].max(axis=1)
        comp_mean = df[comp_cols].mean(axis=1)

        feat["comp_price_min"] = comp_min
        feat["comp_price_max"] = comp_max
        feat["comp_price_mean"] = comp_mean
        feat["comp_price_spread"] = comp_max - comp_min

        if has_lag:
            # Competitor price gap relative to prior-period price anchor
            if "comp_1" in df.columns:
                feat["comp_price_diff_1"] = df["lag_price"] - df["comp_1"]
                feat["comp_price_ratio_1"] = df["comp_1"] / np.clip(df["lag_price"], 1e-6, None)

            feat["lag_price_diff_comp_min"] = df["lag_price"] - comp_min
            feat["lag_price_diff_comp_mean"] = df["lag_price"] - comp_mean
            feat["lag_price_ratio_comp_min"] = comp_min / np.clip(df["lag_price"], 1e-6, None)
            feat["lag_price_ratio_comp_mean"] = df["lag_price"] / np.clip(comp_mean, 1e-6, None)

    # Freight cost differentials (Pre-treatment)
    if "freight_price" in df.columns and "fp1" in df.columns:
        feat["comp_freight_diff_1"] = df["freight_price"] - df["fp1"]

    # Product review rating differential vs key competitor
    if "product_score" in df.columns and "ps1" in df.columns:
        feat["comp_score_diff_1"] = df["product_score"] - df["ps1"]

    # -------------------------------------------------------------------------
    # 3. Seasonality & Cyclical Calendar Dynamics
    # -------------------------------------------------------------------------
    if "month" in df.columns:
        # Cyclical continuous encoding (Month 12 and Month 1 are adjacent)
        feat["sin_month"] = np.sin(2 * np.pi * df["month"] / 12.0)
        feat["cos_month"] = np.cos(2 * np.pi * df["month"] / 12.0)
        feat["is_q4"] = df["month"].isin([10, 11, 12]).astype(float)

    has_weekday = "weekday" in df.columns
    has_weekend = "weekend" in df.columns

    if has_weekday and has_weekend:
        total_days = np.clip(df["weekday"] + df["weekend"], 1.0, None)
        feat["weekend_ratio"] = df["weekend"] / total_days

        if "holiday" in df.columns:
            feat["holiday_ratio"] = df["holiday"] / total_days

    # -------------------------------------------------------------------------
    # 4. Pre-treatment Discount History & Reference Pricing
    # -------------------------------------------------------------------------
    if has_lag:
        # Determine pre-treatment reference anchor
        if reference_prices and "product_id" in df.columns:
            ref_anchor = df["product_id"].map(reference_prices).fillna(df["lag_price"])
        elif "product_id" in df.columns and len(df) > 1 and df["product_id"].nunique() < len(df):
            # Compute historical max lag price within SKU series
            ref_anchor = df.groupby("product_id")["lag_price"].transform("max")
        elif "comp_price_mean" in feat.columns:
            # Fallback benchmark: max of lag_price and market competitor mean
            ref_anchor = np.maximum(df["lag_price"], feat["comp_price_mean"])
        else:
            ref_anchor = df["lag_price"]

        ref_anchor = np.clip(ref_anchor, 1e-6, None)
        historical_discount = np.clip((ref_anchor - df["lag_price"]) / ref_anchor, 0.0, 1.0)
        feat["historical_discount_rate"] = historical_discount
        feat["is_lag_discounted"] = (historical_discount > 0.05).astype(float)

    # -------------------------------------------------------------------------
    # 5. Product & Customer Characteristics
    # -------------------------------------------------------------------------
    if "customers" in df.columns:
        if has_weekday and has_weekend:
            total_days = np.clip(df["weekday"] + df["weekend"], 1.0, None)
            feat["customers_per_day"] = df["customers"] / total_days
        else:
            feat["customers"] = df["customers"]

    for col in [
        "product_weight_g",
        "volume",
        "product_photos_qty",
        "product_description_lenght",
        "product_name_lenght",
    ]:
        if col in df.columns and col not in feat.columns:
            feat[col] = df[col]

    # -------------------------------------------------------------------------
    # 6. Strict Anti-Leakage Filter
    # -------------------------------------------------------------------------
    if drop_leakage:
        leakage_cols = [
            c for c in feat.columns if c.lower() in POST_TREATMENT_VARIABLES or c.lower() in TREATMENT_VARIABLES
        ]
        if leakage_cols:
            feat = feat.drop(columns=leakage_cols)

    return feat


class CausalPreprocessor:
    """Preprocessor for causal confounder matrix X.

    Applies:
    - Missing-data imputation strictly learned on training data (SimpleImputer)
    - StandardScaler to numerical confounders
    - OneHotEncoder to categorical confounders (handling unseen categories)
    - Strict anti-leakage audit preventing treatment, target, or collider leakage
    - Schema consistency guarantees (column ordering, extra/missing column resilience)
    - Retains semantic feature names for model explainability
    """

    def __init__(
        self,
        numerical_cols: Optional[List[str]] = None,
        categorical_cols: Optional[List[str]] = None,
        impute_missing: bool = True,
        numerical_impute_strategy: str = "median",
        categorical_impute_strategy: str = "most_frequent",
        check_leakage: bool = True,
        allow_missing_cols: bool = True,
    ):
        self.numerical_cols = list(numerical_cols) if numerical_cols is not None else None
        self.categorical_cols = list(categorical_cols) if categorical_cols is not None else None
        self.impute_missing = impute_missing
        self.numerical_impute_strategy = numerical_impute_strategy
        self.categorical_impute_strategy = categorical_impute_strategy
        self.check_leakage = check_leakage
        self.allow_missing_cols = allow_missing_cols

        # Fitted estimators
        self.scaler: Optional[StandardScaler] = None
        self.encoder: Optional[OneHotEncoder] = None
        self.numerical_imputer: Optional[SimpleImputer] = None
        self.categorical_imputer: Optional[SimpleImputer] = None

        # Schema metadata
        self.feature_names_out_: List[str] = []
        self.expected_numerical_cols_: List[str] = []
        self.expected_categorical_cols_: List[str] = []
        self.leakage_audit_: Optional[Dict[str, Any]] = None
        self._is_fitted: bool = False

    def fit(self, X: pd.DataFrame) -> "CausalPreprocessor":
        """Fit imputers, scaler on numerical columns, and encoder on categorical columns.

        Parameters
        ----------
        X : pd.DataFrame
            Confounder DataFrame.

        Returns
        -------
        CausalPreprocessor
            Fitted preprocessor instance.
        """
        # 1. Anti-leakage audit
        if self.check_leakage:
            self.leakage_audit_ = audit_confounders(
                df=X,
                confounder_cols=list(X.columns),
                raise_on_leakage=True,
            )

        # 2. Automatically detect column types if not explicitly specified
        if self.categorical_cols is None:
            self.categorical_cols = list(X.select_dtypes(include=["object", "category"]).columns)

        if self.numerical_cols is None:
            self.numerical_cols = [col for col in X.columns if col not in self.categorical_cols]

        self.expected_numerical_cols_ = list(self.numerical_cols)
        self.expected_categorical_cols_ = list(self.categorical_cols)
        self.feature_names_out_ = []

        # 3. Fit Numerical Components (Imputer + StandardScaler)
        if self.numerical_cols:
            num_data = X[self.numerical_cols].copy()
            if self.impute_missing:
                self.numerical_imputer = SimpleImputer(strategy=self.numerical_impute_strategy)
                num_imputed = self.numerical_imputer.fit_transform(num_data)
            else:
                num_imputed = num_data.values

            self.scaler = StandardScaler()
            self.scaler.fit(num_imputed)
            self.feature_names_out_.extend(self.numerical_cols)

        # 4. Fit Categorical Components (Imputer + OneHotEncoder)
        if self.categorical_cols:
            cat_data = X[self.categorical_cols].copy()
            if self.impute_missing:
                self.categorical_imputer = SimpleImputer(
                    strategy=self.categorical_impute_strategy,
                    fill_value="missing",
                )
                cat_imputed = self.categorical_imputer.fit_transform(cat_data)
            else:
                cat_imputed = cat_data.values

            self.encoder = OneHotEncoder(sparse_output=False, handle_unknown="ignore")
            self.encoder.fit(cat_imputed)
            cat_features = list(self.encoder.get_feature_names_out(self.categorical_cols))
            self.feature_names_out_.extend(cat_features)

        self._is_fitted = True
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """Transform confounder matrix X using fitted imputers, scaler, and encoder.

        Maintains strict schema consistency: deterministic column order,
        preserves original DataFrame index, handles unseen categories gracefully,
        and imputes missing columns or values with zero leakage from test set.

        Parameters
        ----------
        X : pd.DataFrame
            Confounder DataFrame to transform.

        Returns
        -------
        pd.DataFrame
            Scaled and encoded confounder DataFrame with readable column names.
        """
        if not self._is_fitted:
            raise ValueError("CausalPreprocessor is not fitted yet. Call 'fit' before 'transform'.")

        if self.check_leakage:
            # Audit columns passed to transform to prevent post-treatment leakage during inference
            audit_confounders(
                df=X,
                confounder_cols=[c for c in X.columns if c in self.expected_numerical_cols_ + self.expected_categorical_cols_],
                raise_on_leakage=True,
            )

        parts = []

        # 1. Transform Numerical Confounders
        if self.expected_numerical_cols_ and self.scaler:
            num_df = pd.DataFrame(index=X.index)
            for col in self.expected_numerical_cols_:
                if col in X.columns:
                    num_df[col] = X[col]
                elif self.allow_missing_cols:
                    # Impute missing column with training baseline median
                    if self.numerical_imputer is not None:
                        col_idx = self.expected_numerical_cols_.index(col)
                        num_df[col] = float(self.numerical_imputer.statistics_[col_idx])
                    else:
                        num_df[col] = 0.0
                else:
                    raise KeyError(f"Expected numerical column '{col}' is missing in transform input.")

            # Impute missing values using train-fitted imputer (anti-leakage!)
            if self.impute_missing and self.numerical_imputer is not None:
                num_array = self.numerical_imputer.transform(num_df[self.expected_numerical_cols_])
            else:
                num_array = num_df[self.expected_numerical_cols_].values

            num_scaled = self.scaler.transform(num_array)
            parts.append(pd.DataFrame(num_scaled, columns=self.expected_numerical_cols_, index=X.index))

        # 2. Transform Categorical Confounders
        if self.expected_categorical_cols_ and self.encoder:
            cat_df = pd.DataFrame(index=X.index)
            for col in self.expected_categorical_cols_:
                if col in X.columns:
                    cat_df[col] = X[col]
                elif self.allow_missing_cols:
                    if self.categorical_imputer is not None:
                        col_idx = self.expected_categorical_cols_.index(col)
                        cat_df[col] = str(self.categorical_imputer.statistics_[col_idx])
                    else:
                        cat_df[col] = "missing"
                else:
                    raise KeyError(f"Expected categorical column '{col}' is missing in transform input.")

            if self.impute_missing and self.categorical_imputer is not None:
                cat_array = self.categorical_imputer.transform(cat_df[self.expected_categorical_cols_])
            else:
                cat_array = cat_df[self.expected_categorical_cols_].values

            cat_encoded = self.encoder.transform(cat_array)
            cat_cols = list(self.encoder.get_feature_names_out(self.expected_categorical_cols_))
            parts.append(pd.DataFrame(cat_encoded, columns=cat_cols, index=X.index))

        if not parts:
            return pd.DataFrame(index=X.index)

        res = pd.concat(parts, axis=1)
        # Ensure column ordering strictly matches feature_names_out_
        return res[self.feature_names_out_]

    def fit_transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """Fit preprocessor and transform confounders in a single step."""
        return self.fit(X).transform(X)


def transform_treatment_and_outcome(
    T: pd.Series,
    Y: pd.Series,
    log_transform_treatment: bool = False,
    log_transform_outcome: bool = False,
) -> Tuple[pd.Series, pd.Series]:
    """Optionally apply log transformations to Treatment and Outcome.

    In economic demand theory:
    - log(Q) ~ beta * log(P) directly yields constant price elasticity beta.
    - Linear Q ~ theta * P yields marginal demand change theta = dQ/dP.

    Parameters
    ----------
    T : pd.Series
        Treatment series (e.g., unit_price).
    Y : pd.Series
        Outcome series (e.g., qty).
    log_transform_treatment : bool
        If True, returns ln(T).
    log_transform_outcome : bool
        If True, returns ln(Y) (using ln(Y) or ln(Y + 1) for non-zero safety).

    Returns
    -------
    Tuple[pd.Series, pd.Series]
        (T_transformed, Y_transformed).
    """
    T_transformed = T.copy()
    Y_transformed = Y.copy()

    if log_transform_treatment:
        if (T_transformed <= 0).any():
            raise ValueError("Cannot log-transform non-positive treatment values.")
        T_transformed = np.log(T_transformed)
        T_transformed.name = f"log_{T.name}"

    if log_transform_outcome:
        if (Y_transformed < 0).any():
            raise ValueError("Cannot log-transform negative outcome values.")
        # If zeros exist, use log1p (ln(1 + Y)) to avoid -inf
        if (Y_transformed == 0).any():
            Y_transformed = np.log1p(Y_transformed)
            Y_transformed.name = f"log1p_{Y.name}"
        else:
            Y_transformed = np.log(Y_transformed)
            Y_transformed.name = f"log_{Y.name}"

    return T_transformed, Y_transformed


def split_causal_data(
    df: pd.DataFrame,
    test_size: float = 0.2,
    random_state: int = 42,
    treatment_col: str = DEFAULT_TREATMENT_COL,
    outcome_col: str = DEFAULT_OUTCOME_COL,
    confounder_cols: Optional[List[str]] = None,
    categorical_confounders: Optional[List[str]] = None,
    log_transform_treatment: bool = False,
    log_transform_outcome: bool = False,
    engineer_features: bool = False,
    impute_missing: bool = True,
) -> Dict[str, Any]:
    """Extract, split, and preprocess causal datasets without data leakage.

    Parameters
    ----------
    df : pd.DataFrame
        Raw or loaded source DataFrame.
    test_size : float
        Fraction for holdout evaluation test set.
    random_state : int
        Seed for reproducible splitting.
    treatment_col : str
        Treatment variable name.
    outcome_col : str
        Outcome variable name.
    confounder_cols : List[str], optional
        List of confounders (numerical + categorical).
    categorical_confounders : List[str], optional
        Explicit categorical confounders to encode.
    log_transform_treatment : bool
        Whether to log-transform treatment for elasticity modeling.
    log_transform_outcome : bool
        Whether to log-transform outcome for elasticity modeling.
    engineer_features : bool
        Whether to apply domain causal feature engineering (competitor gaps,
        discount history, seasonality cyclical dynamics).
    impute_missing : bool
        Whether to apply training-fold median/mode imputation for missing data.

    Returns
    -------
    Dict[str, Any]
        Dictionary with:
        - 'X_train': pd.DataFrame (scaled/encoded)
        - 'X_test': pd.DataFrame (scaled/encoded)
        - 'T_train': pd.Series
        - 'T_test': pd.Series
        - 'Y_train': pd.Series
        - 'Y_test': pd.Series
        - 'preprocessor': CausalPreprocessor fitted on training set only
        - 'feature_names': list of preprocessed feature names
        - 'leakage_audit': dict with audit validation results
    """
    working_df = df.copy()

    # Optional Causal Feature Engineering
    if engineer_features:
        eng_df = engineer_causal_features(
            working_df,
            include_base_confounders=True,
            drop_leakage=True,
        )
        # Combine engineered confounders with treatment and outcome
        for col in eng_df.columns:
            working_df[col] = eng_df[col]

        if confounder_cols is None:
            confounder_cols = list(eng_df.columns)
            # Ensure treatment and outcome are excluded
            confounder_cols = [c for c in confounder_cols if c not in [treatment_col, outcome_col]]

    if confounder_cols is None:
        confounder_cols = list(DEFAULT_CONFOUNDER_COLS)

    # If categorical confounders exist (e.g. product_category_name), include them
    if categorical_confounders:
        for cat in categorical_confounders:
            if cat in working_df.columns and cat not in confounder_cols:
                confounder_cols.append(cat)

    # Strict anti-leakage audit prior to matrix extraction
    audit_results = audit_confounders(
        df=working_df,
        confounder_cols=confounder_cols,
        treatment_col=treatment_col,
        outcome_col=outcome_col,
        raise_on_leakage=True,
    )

    X, T, Y = extract_causal_matrices(
        working_df,
        treatment_col=treatment_col,
        outcome_col=outcome_col,
        confounder_cols=confounder_cols,
    )

    T, Y = transform_treatment_and_outcome(
        T,
        Y,
        log_transform_treatment=log_transform_treatment,
        log_transform_outcome=log_transform_outcome,
    )

    X_train_raw, X_test_raw, T_train, T_test, Y_train, Y_test = train_test_split(
        X, T, Y, test_size=test_size, random_state=random_state
    )

    # Strictly fit preprocessor ONLY on training set to avoid data leakage
    preprocessor = CausalPreprocessor(
        categorical_cols=categorical_confounders,
        impute_missing=impute_missing,
        check_leakage=True,
    )
    preprocessor.fit(X_train_raw)

    X_train = preprocessor.transform(X_train_raw)
    X_test = preprocessor.transform(X_test_raw)

    return {
        "X_train": X_train,
        "X_test": X_test,
        "T_train": T_train,
        "T_test": T_test,
        "Y_train": Y_train,
        "Y_test": Y_test,
        "preprocessor": preprocessor,
        "feature_names": preprocessor.feature_names_out_,
        "leakage_audit": audit_results,
    }


def get_cross_fitting_folds(
    n_samples: int,
    n_folds: int = 5,
    random_state: int = 42,
    shuffle: bool = True,
) -> List[Tuple[np.ndarray, np.ndarray]]:
    """Generate K-fold train/test index splits for DML Neyman-orthogonal cross-fitting.

    Parameters
    ----------
    n_samples : int
        Total number of sample observations.
    n_folds : int
        Number of cross-fitting folds (default: 5).
    random_state : int
        Seed for reproducible folds.
    shuffle : bool
        Whether to shuffle indices before partitioning.

    Returns
    -------
    List[Tuple[np.ndarray, np.ndarray]]
        List of (train_indices, test_indices) tuples.
    """
    kf = KFold(n_splits=n_folds, shuffle=shuffle, random_state=random_state)
    dummy_x = np.zeros(n_samples)
    return list(kf.split(dummy_x))
