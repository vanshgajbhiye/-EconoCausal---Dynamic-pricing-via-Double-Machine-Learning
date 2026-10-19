"""Double Machine Learning (DML) / Robinson's Residualization Engine.

Implements Partially Linear DML for causal effect estimation and price elasticity
under Neyman orthogonality and K-fold cross-fitting (Chernozhukov et al., 2018; Robinson, 1988).

Model Formulation:
    Y = theta * T + g(X) + U,   E[U | X, T] = 0
    T = m(X) + V,               E[V | X] = 0

Residualization:
    Y_tilde = Y - E[Y | X]
    T_tilde = T - E[T | X]
    Y_tilde = theta * T_tilde + epsilon,   E[T_tilde * epsilon] = 0
"""

from typing import Any, Dict, List, Optional, Tuple, Union
import numpy as np
import pandas as pd
from scipy import stats
from sklearn.base import BaseEstimator, RegressorMixin, clone
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from src.data.preprocess import (
    get_cross_fitting_folds,
    split_causal_data,
)


class DoubleMachineLearning:
    """Partially Linear Double Machine Learning (DML) estimator with cross-fitting.

    Estimates the causal treatment effect theta (e.g. price elasticity of demand)
    while controlling for high-dimensional confounders X using arbitrary machine learning
    estimators for the nuisance functions E[Y|X] and E[T|X].

    Parameters
    ----------
    model_y : RegressorMixin, optional
        Estimator for outcome nuisance function E[Y|X].
        Defaults to RandomForestRegressor(n_estimators=100, max_depth=6, random_state=random_state).
    model_t : RegressorMixin, optional
        Estimator for treatment nuisance function E[T|X].
        Defaults to RandomForestRegressor(n_estimators=100, max_depth=6, random_state=random_state).
    n_folds : int, default=5
        Number of cross-fitting folds (K >= 2).
    random_state : int, default=42
        Random seed for cross-fitting splits and reproducible nuisance models.
    shuffle : bool, default=True
        Whether to shuffle data before cross-fitting partitioning.
    """

    def __init__(
        self,
        model_y: Optional[Any] = None,
        model_t: Optional[Any] = None,
        n_folds: int = 5,
        random_state: int = 42,
        shuffle: bool = True,
    ):
        if n_folds < 2:
            raise ValueError(f"n_folds must be at least 2, got {n_folds}")

        self.model_y = model_y
        self.model_t = model_t
        self.n_folds = n_folds
        self.random_state = random_state
        self.shuffle = shuffle

        # Learned attributes
        self.theta_: Optional[float] = None
        self.std_error_: Optional[float] = None
        self.t_stat_: Optional[float] = None
        self.p_value_: Optional[float] = None
        self.ci_: Optional[Tuple[float, float]] = None

        # Residuals and predictions
        self.residuals_y_: Optional[np.ndarray] = None
        self.residuals_t_: Optional[np.ndarray] = None
        self.pred_y_: Optional[np.ndarray] = None
        self.pred_t_: Optional[np.ndarray] = None

        # Nuisance models and evaluation metrics
        self.models_y_: List[Any] = []
        self.models_t_: List[Any] = []
        self.nuisance_scores_: Dict[str, float] = {}
        self.residual_diagnostics_: Dict[str, float] = {}
        self._is_fitted: bool = False

    def _get_default_estimators(self) -> Tuple[Any, Any]:
        """Provide default nuisance estimators if none were supplied."""
        base_y = (
            self.model_y
            if self.model_y is not None
            else RandomForestRegressor(
                n_estimators=100, max_depth=6, random_state=self.random_state
            )
        )
        base_t = (
            self.model_t
            if self.model_t is not None
            else RandomForestRegressor(
                n_estimators=100, max_depth=6, random_state=self.random_state
            )
        )
        return clone(base_y), clone(base_t)

    def fit(
        self,
        X: Union[pd.DataFrame, np.ndarray],
        T: Union[pd.Series, np.ndarray],
        Y: Union[pd.Series, np.ndarray],
    ) -> "DoubleMachineLearning":
        """Fit the DML model using K-fold Neyman-orthogonal cross-fitting.

        Parameters
        ----------
        X : Union[pd.DataFrame, np.ndarray]
            Covariate / confounder matrix of shape (n_samples, n_features).
        T : Union[pd.Series, np.ndarray]
            Treatment vector (e.g. unit_price or log(unit_price)) of shape (n_samples,).
        Y : Union[pd.Series, np.ndarray]
            Outcome vector (e.g. qty or log(qty)) of shape (n_samples,).

        Returns
        -------
        DoubleMachineLearning
            Fitted estimator with computed theta, standard error, and confidence intervals.
        """
        # Convert inputs to clean 2D/1D numpy arrays
        X_arr = np.asarray(X, dtype=np.float64)
        T_arr = np.asarray(T, dtype=np.float64).ravel()
        Y_arr = np.asarray(Y, dtype=np.float64).ravel()

        n_samples = len(X_arr)
        if len(T_arr) != n_samples or len(Y_arr) != n_samples:
            raise ValueError(
                f"Sample size mismatch: len(X)={n_samples}, len(T)={len(T_arr)}, len(Y)={len(Y_arr)}"
            )

        if n_samples < self.n_folds:
            raise ValueError(
                f"n_samples ({n_samples}) must be greater than or equal to n_folds ({self.n_folds})"
            )

        # Check for NaN / Inf
        if np.isnan(X_arr).any() or np.isnan(T_arr).any() or np.isnan(Y_arr).any():
            raise ValueError("Input matrices contain NaN values. Ensure preprocessing is complete.")
        if np.isinf(X_arr).any() or np.isinf(T_arr).any() or np.isinf(Y_arr).any():
            raise ValueError("Input matrices contain infinite values.")

        # Partition cross-fitting folds
        folds = get_cross_fitting_folds(
            n_samples=n_samples,
            n_folds=self.n_folds,
            random_state=self.random_state,
            shuffle=self.shuffle,
        )

        pred_y = np.zeros(n_samples, dtype=np.float64)
        pred_t = np.zeros(n_samples, dtype=np.float64)
        self.models_y_ = []
        self.models_t_ = []

        base_y, base_t = self._get_default_estimators()

        # Step 1: K-Fold Cross-Fitting for Nuisance Models E[Y|X] and E[T|X]
        for fold_idx, (train_idx, test_idx) in enumerate(folds):
            X_tr, X_te = X_arr[train_idx], X_arr[test_idx]
            T_tr, _ = T_arr[train_idx], T_arr[test_idx]
            Y_tr, _ = Y_arr[train_idx], Y_arr[test_idx]

            # Clone fresh estimators for this fold
            est_y = clone(base_y)
            est_t = clone(base_t)

            # Fit nuisance models exclusively on fold training partition
            est_y.fit(X_tr, Y_tr)
            est_t.fit(X_tr, T_tr)

            # Predict out-of-fold
            pred_y[test_idx] = est_y.predict(X_te)
            pred_t[test_idx] = est_t.predict(X_te)

            self.models_y_.append(est_y)
            self.models_t_.append(est_t)

        self.pred_y_ = pred_y
        self.pred_t_ = pred_t

        # Step 2: Robinson's Residualization (Neyman Orthogonal Score)
        res_y = Y_arr - pred_y
        res_t = T_arr - pred_t
        self.residuals_y_ = res_y
        self.residuals_t_ = res_t

        # Step 3: Second-Stage Orthogonal Causal Estimation
        denom = np.sum(res_t**2)
        if denom < 1e-12:
            raise ValueError(
                "Treatment residuals variance is zero or negligible. "
                "Confounders may have completely collinearly explained treatment assignment."
            )

        numer = np.sum(res_t * res_y)
        theta_hat = numer / denom
        self.theta_ = float(theta_hat)

        # Step 4: Asymptotic Standard Error via DML Sandwich Variance Estimator
        # epsilon_hat = Y_tilde - theta_hat * T_tilde
        epsilon_hat = res_y - theta_hat * res_t

        # J_hat = (1/N) * sum(res_t^2)
        # Omega_hat = (1/N) * sum((res_t * epsilon_hat)^2)
        # Var(theta) = (1/N) * J_hat^(-1) * Omega_hat * J_hat^(-1) = sum((res_t * epsilon_hat)^2) / (denom^2)
        var_theta = np.sum((res_t * epsilon_hat) ** 2) / (denom**2)
        std_error = np.sqrt(max(var_theta, 1e-14))
        self.std_error_ = float(std_error)

        # Step 5: Inference (t-statistic, p-value, 95% Confidence Interval)
        t_stat = self.theta_ / self.std_error_
        self.t_stat_ = float(t_stat)

        # Two-sided p-value from normal distribution (asymptotic normality of DML)
        p_val = 2.0 * (1.0 - stats.norm.cdf(abs(t_stat)))
        self.p_value_ = float(p_val)

        z_crit = 1.959963984540054  # 95% confidence critical value
        ci_lower = self.theta_ - z_crit * self.std_error_
        ci_upper = self.theta_ + z_crit * self.std_error_
        self.ci_ = (float(ci_lower), float(ci_upper))

        # Step 6: Nuisance Model Diagnostics & Residual Orthogonality
        r2_y = r2_score(Y_arr, pred_y)
        rmse_y = np.sqrt(mean_squared_error(Y_arr, pred_y))
        mae_y = mean_absolute_error(Y_arr, pred_y)

        r2_t = r2_score(T_arr, pred_t)
        rmse_t = np.sqrt(mean_squared_error(T_arr, pred_t))
        mae_t = mean_absolute_error(T_arr, pred_t)

        self.nuisance_scores_ = {
            "r2_y": float(r2_y),
            "rmse_y": float(rmse_y),
            "mae_y": float(mae_y),
            "r2_t": float(r2_t),
            "rmse_t": float(rmse_t),
            "mae_t": float(mae_t),
        }

        # Check orthogonality: inner product of treatment residual and structural noise epsilon
        inner_prod = float(np.sum(res_t * epsilon_hat))
        norm_t = float(np.linalg.norm(res_t))
        norm_eps = float(np.linalg.norm(epsilon_hat))
        ortho_score = abs(inner_prod) / (norm_t * norm_eps + 1e-12)

        # Centered Pearson correlation
        std_eps = np.std(epsilon_hat)
        std_rest = np.std(res_t)
        if std_eps > 1e-12 and std_rest > 1e-12:
            ortho_corr = float(np.corrcoef(res_t, epsilon_hat)[0, 1])
        else:
            ortho_corr = 0.0

        self.residual_diagnostics_ = {
            "orthogonality_score": float(ortho_score),
            "orthogonality_corr": float(ortho_corr),
            "mean_residual_y": float(np.mean(res_y)),
            "mean_residual_t": float(np.mean(res_t)),
            "var_residual_t": float(np.var(res_t)),
        }

        self._is_fitted = True
        return self

    def summary(self) -> Dict[str, Any]:
        """Return structured summary of the DML estimation results."""
        if not self._is_fitted:
            raise ValueError("Model is not fitted yet. Call 'fit' before 'summary'.")

        return {
            "theta": self.theta_,
            "std_error": self.std_error_,
            "t_stat": self.t_stat_,
            "p_value": self.p_value_,
            "ci_95": self.ci_,
            "ci_lower": self.ci_[0],
            "ci_upper": self.ci_[1],
            "n_folds": self.n_folds,
            "nuisance_scores": self.nuisance_scores_,
            "residual_diagnostics": self.residual_diagnostics_,
        }

    def format_summary_table(self) -> str:
        """Format estimation results as a readable text summary table."""
        if not self._is_fitted:
            raise ValueError("Model is not fitted yet. Call 'fit' before formatting.")

        ci_str = f"[{self.ci_[0]:.4f}, {self.ci_[1]:.4f}]"
        lines = [
            "=" * 70,
            "          DOUBLE MACHINE LEARNING (DML) ESTIMATION SUMMARY",
            "=" * 70,
            f"Causal Parameter (theta / Elasticity): {self.theta_:.4f}",
            f"Standard Error (SE):                   {self.std_error_:.4f}",
            f"t-statistic:                           {self.t_stat_:.4f}",
            f"p-value:                               {self.p_value_:.4e}",
            f"95% Confidence Interval:               {ci_str}",
            f"Number of Cross-Fitting Folds (K):     {self.n_folds}",
            "-" * 70,
            "NUISANCE ESTIMATOR PERFORMANCE (Out-of-Fold Cross-Fitting):",
            f"  - Outcome Model E[Y|X]:   R2 = {self.nuisance_scores_['r2_y']:.4f}, RMSE = {self.nuisance_scores_['rmse_y']:.4f}",
            f"  - Treatment Model E[T|X]: R2 = {self.nuisance_scores_['r2_t']:.4f}, RMSE = {self.nuisance_scores_['rmse_t']:.4f}",
            "-" * 70,
            "RESIDUAL ORTHOGONALITY DIAGNOSTICS:",
            f"  - Cov(T_tilde, epsilon) Corr:        {self.residual_diagnostics_['orthogonality_corr']:.4e} (ideal: ~0.0)",
            f"  - Mean Y Residual:                   {self.residual_diagnostics_['mean_residual_y']:.4e}",
            f"  - Mean T Residual:                   {self.residual_diagnostics_['mean_residual_t']:.4e}",
            "=" * 70,
        ]
        return "\n".join(lines)

    def predict_counterfactual(
        self,
        T_new: Union[float, np.ndarray],
        T_old: Union[float, np.ndarray],
    ) -> np.ndarray:
        """Predict the expected change in outcome Delta Y given a change in treatment Delta T.

        Under the partially linear model:
            Delta Y = theta * (T_new - T_old)

        For log-log models, Delta Y represents percentage change in demand for
        percentage change in price.

        Parameters
        ----------
        T_new : Union[float, np.ndarray]
            Counterfactual new treatment values.
        T_old : Union[float, np.ndarray]
            Baseline / original treatment values.

        Returns
        -------
        np.ndarray
            Predicted change in outcome Delta Y.
        """
        if not self._is_fitted:
            raise ValueError("Model is not fitted. Call 'fit' before predicting counterfactuals.")

        delta_t = np.asarray(T_new, dtype=np.float64) - np.asarray(T_old, dtype=np.float64)
        return self.theta_ * delta_t


def train_dml_pipeline(
    df: pd.DataFrame,
    treatment_col: str = "unit_price",
    outcome_col: str = "qty",
    confounder_cols: Optional[List[str]] = None,
    categorical_confounders: Optional[List[str]] = None,
    log_transform_treatment: bool = True,
    log_transform_outcome: bool = True,
    test_size: float = 0.2,
    model_y: Optional[Any] = None,
    model_t: Optional[Any] = None,
    n_folds: int = 5,
    random_state: int = 42,
) -> Dict[str, Any]:
    """End-to-end causal training pipeline from DataFrame to fitted DML model.

    Parameters
    ----------
    df : pd.DataFrame
        Raw dataset (e.g. retail_price.csv).
    treatment_col : str
        Treatment column name.
    outcome_col : str
        Outcome column name.
    confounder_cols : Optional[List[str]]
        List of continuous and categorical confounder column names.
    categorical_confounders : Optional[List[str]]
        List of categorical column names to one-hot encode.
    log_transform_treatment : bool
        If True, estimate log-log price elasticity.
    log_transform_outcome : bool
        If True, estimate log-log price elasticity.
    test_size : float
        Holdout test set fraction.
    model_y : Optional[Any]
        Regressor for E[Y|X].
    model_t : Optional[Any]
        Regressor for E[T|X].
    n_folds : int
        Cross-fitting folds.
    random_state : int
        Seed for reproducibility.

    Returns
    -------
    Dict[str, Any]
        Pipeline execution dictionary with:
        - 'model': fitted DoubleMachineLearning instance
        - 'summary': dictionary of results
        - 'summary_table': formatted string report
        - 'data_splits': preprocessed train and test data
    """
    # 1. Preprocess and split without data leakage
    splits = split_causal_data(
        df=df,
        test_size=test_size,
        random_state=random_state,
        treatment_col=treatment_col,
        outcome_col=outcome_col,
        confounder_cols=confounder_cols,
        categorical_confounders=categorical_confounders,
        log_transform_treatment=log_transform_treatment,
        log_transform_outcome=log_transform_outcome,
    )

    X_train = splits["X_train"]
    T_train = splits["T_train"]
    Y_train = splits["Y_train"]

    # 2. Instantiate and fit DML on training partition
    dml = DoubleMachineLearning(
        model_y=model_y,
        model_t=model_t,
        n_folds=n_folds,
        random_state=random_state,
    )
    dml.fit(X=X_train, T=T_train, Y=Y_train)

    summary = dml.summary()
    table = dml.format_summary_table()

    return {
        "model": dml,
        "summary": summary,
        "summary_table": table,
        "data_splits": splits,
    }
