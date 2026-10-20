"""Heterogeneous Treatment Effect / Conditional Average Treatment Effect (CATE) Engine.

Estimates heterogeneous price elasticity theta(X) across product categories,
customer footfall segments, and market conditions using the Double Machine Learning (DML)
R-learner / Robinson residual interaction formulation (Chernozhukov et al., 2018; Nie & Wager, 2021).

Formulation:
    Y_tilde = theta(X) * T_tilde + epsilon
    theta(X) = W(X)^T * beta
    Y_tilde = sum_j beta_j * (W_j(X) * T_tilde) + epsilon
"""

from typing import Any, Dict, List, Optional, Tuple, Union
import numpy as np
import pandas as pd
from scipy import stats
from sklearn.linear_model import Ridge

from src.models.train_dml import DoubleMachineLearning


class CATEEstimator:
    """Estimator for Conditional Average Treatment Effects (Heterogeneous Elasticity).

    Models the conditional price elasticity theta(X) as a linear combination of effect
    moderators W(X):
        theta(X) = beta_0 + sum_{j=1}^M beta_j * W_j(X)

    Parameters
    ----------
    dml_base : Optional[DoubleMachineLearning]
        Pre-configured or fitted DoubleMachineLearning instance for generating out-of-fold
        residuals Y_tilde and T_tilde. If None, instantiates a default DoubleMachineLearning.
    moderator_cols : Optional[List[str]]
        Specific column names in X to use as moderators W(X).
        If None, all columns in X are used.
    alpha : float, default=1.0
        Ridge regularization penalty for second-stage CATE parameter estimation.
    include_intercept : bool, default=True
        Whether to include a baseline intercept beta_0 (representing average treatment effect ATE).
    random_state : int, default=42
        Seed for reproducibility.
    """

    def __init__(
        self,
        dml_base: Optional[DoubleMachineLearning] = None,
        moderator_cols: Optional[List[str]] = None,
        alpha: float = 1.0,
        include_intercept: bool = True,
        random_state: int = 42,
    ):
        self.dml_base = dml_base if dml_base is not None else DoubleMachineLearning(random_state=random_state)
        self.moderator_cols = moderator_cols
        self.alpha = alpha
        self.include_intercept = include_intercept
        self.random_state = random_state

        # Fitted parameters
        self.beta_: Optional[np.ndarray] = None
        self.beta_names_: List[str] = []
        self.intercept_: float = 0.0
        self.cate_predictions_: Optional[np.ndarray] = None
        self.std_error_beta_: Optional[np.ndarray] = None
        self._is_fitted: bool = False

    def _extract_moderators(
        self,
        X: Union[pd.DataFrame, np.ndarray],
    ) -> Tuple[np.ndarray, List[str]]:
        """Extract moderator matrix W and column names."""
        if isinstance(X, pd.DataFrame):
            if self.moderator_cols is not None:
                avail = [c for c in self.moderator_cols if c in X.columns]
                if not avail:
                    raise ValueError(f"None of specified moderator_cols {self.moderator_cols} found in X.")
                W_df = X[avail]
                names = list(avail)
            else:
                W_df = X
                names = list(X.columns)
            W_arr = np.asarray(W_df, dtype=np.float64)
        else:
            W_arr = np.asarray(X, dtype=np.float64)
            names = [f"mod_{i}" for i in range(W_arr.shape[1])]

        return W_arr, names

    def fit(
        self,
        X: Union[pd.DataFrame, np.ndarray],
        T: Union[pd.Series, np.ndarray],
        Y: Union[pd.Series, np.ndarray],
    ) -> "CATEEstimator":
        """Fit CATE model using DML residualization and second-stage interaction regression.

        Parameters
        ----------
        X : Union[pd.DataFrame, np.ndarray]
            Confounder matrix of shape (n_samples, n_features).
        T : Union[pd.Series, np.ndarray]
            Treatment vector (e.g. log(unit_price)) of shape (n_samples,).
        Y : Union[pd.Series, np.ndarray]
            Outcome vector (e.g. log(qty)) of shape (n_samples,).

        Returns
        -------
        CATEEstimator
            Fitted estimator with learned moderation parameters beta.
        """
        # Step 1: Fit base DML model to obtain out-of-fold residuals
        if not self.dml_base._is_fitted:
            self.dml_base.fit(X, T, Y)

        res_y = self.dml_base.residuals_y_
        res_t = self.dml_base.residuals_t_
        n_samples = len(res_y)

        # Step 2: Extract moderator design matrix W
        W_arr, base_names = self._extract_moderators(X)
        n_mods = W_arr.shape[1]

        # Construct interaction matrix Z = W * T_tilde
        # If include_intercept is True, column 0 is T_tilde itself (representing baseline ATE)
        cols = []
        col_names = []

        if self.include_intercept:
            cols.append(res_t.reshape(-1, 1))
            col_names.append("baseline_elasticity_intercept")

        for j in range(n_mods):
            interaction = (W_arr[:, j] * res_t).reshape(-1, 1)
            cols.append(interaction)
            col_names.append(f"{base_names[j]} * T_tilde")

        Z = np.hstack(cols)
        self.beta_names_ = col_names

        # Step 3: Solve regularized second-stage regression: Y_tilde = Z * beta + epsilon
        # Using Ridge closed form: beta = (Z^T Z + alpha * I)^{-1} Z^T Y_tilde
        dim = Z.shape[1]
        eye = np.eye(dim)
        if self.include_intercept:
            # Do not penalize baseline intercept to preserve unbiased ATE anchoring
            eye[0, 0] = 0.0

        A = Z.T @ Z + self.alpha * eye
        b = Z.T @ res_y
        beta = np.linalg.solve(A, b)
        self.beta_ = beta

        # Step 4: Residuals and standard error computation
        epsilon_hat = res_y - Z @ beta
        # Sandwich variance: (Z^T Z + alpha I)^(-1) (Z^T diag(eps^2) Z) (Z^T Z + alpha I)^(-1)
        A_inv = np.linalg.pinv(A)
        meat = Z.T @ np.diag(epsilon_hat**2) @ Z
        var_beta = A_inv @ meat @ A_inv
        std_errors = np.sqrt(np.maximum(np.diag(var_beta), 1e-14))
        self.std_error_beta_ = std_errors

        # Step 5: Predict in-sample CATE values
        self.cate_predictions_ = self.predict_cate(X)
        self._is_fitted = True
        return self

    def predict_cate(
        self,
        X: Union[pd.DataFrame, np.ndarray],
    ) -> np.ndarray:
        """Predict conditional price elasticity theta(X) for new or existing observations.

        Parameters
        ----------
        X : Union[pd.DataFrame, np.ndarray]
            Confounder/covariate matrix.

        Returns
        -------
        np.ndarray
            Predicted individual price elasticities theta(x_i) for each sample.
        """
        if self.beta_ is None:
            raise ValueError("CATEEstimator is not fitted yet. Call 'fit' first.")

        W_arr, _ = self._extract_moderators(X)
        n_samples = W_arr.shape[0]

        if self.include_intercept:
            # theta(X) = beta_0 + sum_{j=1}^M beta_j * W_j(X)
            intercept = self.beta_[0]
            slopes = self.beta_[1:]
            cate = intercept + W_arr @ slopes
        else:
            cate = W_arr @ self.beta_

        return cate

    def cate_by_segment(
        self,
        segment_labels: Union[pd.Series, np.ndarray],
        X: Optional[Union[pd.DataFrame, np.ndarray]] = None,
    ) -> pd.DataFrame:
        """Aggregate predicted elasticities by segment or product category.

        Parameters
        ----------
        segment_labels : Union[pd.Series, np.ndarray]
            Categorical series (e.g. product_category_name or customer_segment).
        X : Optional[Union[pd.DataFrame, np.ndarray]]
            Covariate matrix. If None, uses in-sample predictions.

        Returns
        -------
        pd.DataFrame
            Summary table with mean, median, std, min, max elasticity and dynamic pricing recommendations.
        """
        if not self._is_fitted:
            raise ValueError("Model is not fitted. Call 'fit' before segment aggregation.")

        if X is not None:
            cates = self.predict_cate(X)
        else:
            cates = self.cate_predictions_

        labels = np.asarray(segment_labels)
        if len(labels) != len(cates):
            raise ValueError(f"Length mismatch: segment_labels ({len(labels)}) != cates ({len(cates)})")

        df = pd.DataFrame({"segment": labels, "elasticity": cates})
        grouped = df.groupby("segment")["elasticity"].agg(
            sample_count="count",
            mean_elasticity="mean",
            median_elasticity="median",
            std_elasticity="std",
            min_elasticity="min",
            max_elasticity="max",
        ).reset_index()

        # Dynamic pricing strategic recommendations based on microeconomic theory
        def _get_recommendation(row: pd.Series) -> str:
            mean_e = row["mean_elasticity"]
            if mean_e < -1.2:
                return "Highly Elastic: Aggressive discounting generates high unit surge"
            elif -1.2 <= mean_e < -0.8:
                return "Unit Elastic: Moderate promotions preserve revenue balance"
            elif -0.8 <= mean_e < 0.0:
                return "Inelastic: Avoid deep discounting; price increases expand margin"
            else:
                return "Anomalous/Prestige: Demand increases with price (Veblen/Giffen effect)"

        grouped["pricing_strategy"] = grouped.apply(_get_recommendation, axis=1)
        return grouped.sort_values(by="mean_elasticity", ascending=True)

    def get_moderation_effects(self) -> pd.DataFrame:
        """Return table of moderation coefficients beta, standard errors, and t-stats."""
        if not self._is_fitted:
            raise ValueError("Model is not fitted. Call 'fit' before retrieving moderation effects.")

        t_stats = self.beta_ / self.std_error_beta_
        p_vals = 2.0 * (1.0 - stats.norm.cdf(np.abs(t_stats)))

        df = pd.DataFrame({
            "moderator": self.beta_names_,
            "coefficient": self.beta_,
            "std_error": self.std_error_beta_,
            "t_stat": t_stats,
            "p_value": p_vals,
        })
        return df

    def summary(self) -> Dict[str, Any]:
        """Return dictionary summary of CATE estimation."""
        if not self._is_fitted:
            raise ValueError("Model is not fitted yet.")

        return {
            "mean_cate": float(np.mean(self.cate_predictions_)),
            "median_cate": float(np.median(self.cate_predictions_)),
            "std_cate": float(np.std(self.cate_predictions_)),
            "min_cate": float(np.min(self.cate_predictions_)),
            "max_cate": float(np.max(self.cate_predictions_)),
            "n_moderators": len(self.beta_names_),
            "moderation_table": self.get_moderation_effects(),
        }

    def format_summary_table(self) -> str:
        """Format CATE estimation results as a readable ASCII report."""
        if not self._is_fitted:
            raise ValueError("Model is not fitted yet.")

        lines = [
            "=" * 75,
            "     CONDITIONAL AVERAGE TREATMENT EFFECT (CATE / HETEROGENEITY) REPORT",
            "=" * 75,
            f"Mean Predicted Elasticity across samples:   {np.mean(self.cate_predictions_):.4f}",
            f"Median Elasticity:                          {np.median(self.cate_predictions_):.4f}",
            f"Elasticity Dispersion (Std Dev):            {np.std(self.cate_predictions_):.4f}",
            f"Min Elasticity (Most Price-Sensitive):      {np.min(self.cate_predictions_):.4f}",
            f"Max Elasticity (Least Price-Sensitive):     {np.max(self.cate_predictions_):.4f}",
            "-" * 75,
            "TOP MODERATION EFFECTS (Beta Coefficients on T_tilde interaction):",
        ]
        mod_df = self.get_moderation_effects()
        for _, row in mod_df.head(8).iterrows():
            lines.append(
                f"  - {row['moderator'][:38]:<38} | Coef: {row['coefficient']:+.4f} | SE: {row['std_error']:.4f} | p: {row['p_value']:.4e}"
            )
        lines.append("=" * 75)
        return "\n".join(lines)
