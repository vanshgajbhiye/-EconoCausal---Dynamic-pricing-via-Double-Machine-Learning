"""Data preprocessing, feature transformation, and cross-fitting fold partitioning.

Specifically built for causal machine learning and Double Machine Learning (DML)
estimation on retail pricing and demand data.
"""

from typing import Any, Dict, List, Optional, Tuple
import numpy as np
import pandas as pd
from sklearn.model_selection import KFold, train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from src.data.load_data import (
    DEFAULT_CONFOUNDER_COLS,
    DEFAULT_OUTCOME_COL,
    DEFAULT_TREATMENT_COL,
    extract_causal_matrices,
)


class CausalPreprocessor:
    """Preprocessor for causal confounder matrix X.

    Applies:
    - StandardScaler to numerical confounders
    - OneHotEncoder to categorical confounders (e.g. product_category_name)
    - Anti-leakage preservation: fits only on training folds/samples
    - Retains semantic feature names for model explainability
    """

    def __init__(
        self,
        numerical_cols: Optional[List[str]] = None,
        categorical_cols: Optional[List[str]] = None,
    ):
        self.numerical_cols = numerical_cols
        self.categorical_cols = categorical_cols
        self.scaler: Optional[StandardScaler] = None
        self.encoder: Optional[OneHotEncoder] = None
        self.feature_names_out_: List[str] = []
        self._is_fitted: bool = False

    def fit(self, X: pd.DataFrame) -> "CausalPreprocessor":
        """Fit scaler on numerical columns and encoder on categorical columns.

        Parameters
        ----------
        X : pd.DataFrame
            Confounder DataFrame.

        Returns
        -------
        CausalPreprocessor
            Fitted preprocessor instance.
        """
        # Automatically detect column types if not explicitly specified
        if self.categorical_cols is None:
            self.categorical_cols = list(X.select_dtypes(include=["object", "category"]).columns)

        if self.numerical_cols is None:
            self.numerical_cols = [col for col in X.columns if col not in self.categorical_cols]

        self.feature_names_out_ = []


        if self.numerical_cols:
            self.scaler = StandardScaler()
            self.scaler.fit(X[self.numerical_cols])
            self.feature_names_out_.extend(self.numerical_cols)

        if self.categorical_cols:
            self.encoder = OneHotEncoder(sparse_output=False, handle_unknown="ignore")
            self.encoder.fit(X[self.categorical_cols])
            cat_features = list(self.encoder.get_feature_names_out(self.categorical_cols))
            self.feature_names_out_.extend(cat_features)

        self._is_fitted = True
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """Transform confounder matrix X using fitted scaler and encoder.

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

        parts = []

        if self.numerical_cols and self.scaler:
            num_scaled = self.scaler.transform(X[self.numerical_cols])
            num_df = pd.DataFrame(num_scaled, columns=self.numerical_cols, index=X.index)
            parts.append(num_df)

        if self.categorical_cols and self.encoder:
            cat_encoded = self.encoder.transform(X[self.categorical_cols])
            cat_cols = list(self.encoder.get_feature_names_out(self.categorical_cols))
            cat_df = pd.DataFrame(cat_encoded, columns=cat_cols, index=X.index)
            parts.append(cat_df)

        if not parts:
            return pd.DataFrame(index=X.index)

        return pd.concat(parts, axis=1)

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
    """
    if confounder_cols is None:
        confounder_cols = list(DEFAULT_CONFOUNDER_COLS)

    # If categorical confounders exist (e.g. product_category_name), include them
    if categorical_confounders:
        for cat in categorical_confounders:
            if cat in df.columns and cat not in confounder_cols:
                confounder_cols.append(cat)

    X, T, Y = extract_causal_matrices(
        df,
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
    preprocessor = CausalPreprocessor(categorical_cols=categorical_confounders)
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
