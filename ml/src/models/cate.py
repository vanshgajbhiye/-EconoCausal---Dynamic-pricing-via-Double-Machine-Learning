"""Heterogeneous Treatment Effect / Conditional Average Treatment Effect (CATE) Engine.

Estimates heterogeneous price elasticity theta(X) across product categories,
customer footfall segments, and market conditions using the Double Machine Learning (DML)
R-learner / Robinson residual interaction formulation (Chernozhukov et al., 2018; Nie & Wager, 2021).

Formulation:
    Y_tilde = theta(X) * T_tilde + epsilon
    theta(X) = W(X)^T * beta
    Y_tilde = sum_j beta_j * (W_j(X) * T_tilde) + epsilon

Unit-Elasticity Boundary:
    Under log-log demand (Y = log(Q), T = log(P)):
        theta = d log(Q) / d log(P) = (P / Q) * (dQ / dP)
        Revenue R(P) = P * Q(P)
        dR / dP = (1 + theta) * Q(P)
    When theta = -1 (Unit Elasticity), dR / dP = 0: marginal revenue is zero,
    and revenue is maximized with respect to price.
    When theta < -1 (Elastic), dR / dP < 0: price reductions increase revenue.
    When -1 < theta < 0 (Inelastic), dR / dP > 0: price increases expand revenue and margin.
"""

from typing import Any, Dict, List, Optional, Tuple, Union
import numpy as np
import pandas as pd
from scipy import stats

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
        self.var_beta_: Optional[np.ndarray] = None
        self.residual_eps_: Optional[np.ndarray] = None
        self.n_samples_train_: int = 0

        # Baseline ATE anchoring
        self.baseline_ate_: Optional[float] = None
        self.baseline_ate_se_: Optional[float] = None
        self.baseline_ate_ci_: Optional[Tuple[float, float]] = None

        # Empirical moderator support bounding box
        self.moderator_ranges_: Dict[str, Dict[str, float]] = {}
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
        self.n_samples_train_ = n_samples

        self.baseline_ate_ = float(self.dml_base.theta_)
        self.baseline_ate_se_ = float(self.dml_base.std_error_)
        self.baseline_ate_ci_ = self.dml_base.ci_

        # Step 2: Extract moderator design matrix W
        W_arr, base_names = self._extract_moderators(X)
        n_mods = W_arr.shape[1]

        # Record empirical moderator support bounds
        self.moderator_ranges_ = {}
        for j, col_name in enumerate(base_names):
            vals = W_arr[:, j]
            self.moderator_ranges_[col_name] = {
                "min": float(np.min(vals)),
                "max": float(np.max(vals)),
                "mean": float(np.mean(vals)),
                "std": float(np.std(vals)),
                "p01": float(np.percentile(vals, 1)),
                "p99": float(np.percentile(vals, 99)),
            }

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

        # Step 4: Residuals and sandwich standard error computation
        epsilon_hat = res_y - Z @ beta
        self.residual_eps_ = epsilon_hat

        # Sandwich variance: (Z^T Z + alpha I)^(-1) (Z^T diag(eps^2) Z) (Z^T Z + alpha I)^(-1)
        A_inv = np.linalg.pinv(A)
        meat = Z.T @ np.diag(epsilon_hat**2) @ Z
        var_beta = A_inv @ meat @ A_inv
        self.var_beta_ = var_beta
        std_errors = np.sqrt(np.maximum(np.diag(var_beta), 1e-14))
        self.std_error_beta_ = std_errors

        # Step 5: Predict in-sample CATE values
        self._is_fitted = True
        self.cate_predictions_ = self.predict_cate(X)
        return self

    def _get_design_matrix(self, X: Union[pd.DataFrame, np.ndarray]) -> np.ndarray:
        """Construct evaluation design matrix W* with optional leading intercept."""
        W_arr, _ = self._extract_moderators(X)
        n_samples = W_arr.shape[0]
        if self.include_intercept:
            return np.hstack([np.ones((n_samples, 1), dtype=np.float64), W_arr])
        return W_arr

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

        W_star = self._get_design_matrix(X)
        return W_star @ self.beta_

    def predict_cate_with_uncertainty(
        self,
        X: Union[pd.DataFrame, np.ndarray],
        alpha: float = 0.05,
    ) -> pd.DataFrame:
        """Predict individual price elasticity theta(X) with asymptotic standard errors and CIs.

        Computes point estimates and parameter-induced prediction variance:
            Var(theta(x_i)) = w*(x_i)^T Var(beta) w*(x_i)

        Parameters
        ----------
        X : Union[pd.DataFrame, np.ndarray]
            Confounder/covariate matrix.
        alpha : float, default=0.05
            Significance level for confidence intervals (0.05 yields 95% CI).

        Returns
        -------
        pd.DataFrame
            DataFrame with columns:
            - 'predicted_elasticity': point estimate
            - 'std_error': standard error of predicted elasticity
            - 'ci_lower': lower bound of confidence interval
            - 'ci_upper': upper bound of confidence interval
            - 't_stat_unit': t-statistic against unit elasticity theta = -1.0
            - 'p_value_unit': two-sided p-value against theta = -1.0
            - 'elasticity_regime': regime classification ("Elastic", "Inelastic", "Unit Elastic", "Anomalous")
            - 'is_statistically_significant': whether effect is significantly non-zero
        """
        if self.beta_ is None or self.var_beta_ is None or not self._is_fitted:
            raise ValueError("CATEEstimator is not fitted yet. Call 'fit' first.")

        W_star = self._get_design_matrix(X)
        cates = W_star @ self.beta_

        # Fast vectorised diagonal quadratic form: diag(W* Var(beta) W*^T)
        vars_ind = np.sum((W_star @ self.var_beta_) * W_star, axis=1)
        se_ind = np.sqrt(np.maximum(vars_ind, 1e-14))

        z_crit = float(stats.norm.ppf(1.0 - alpha / 2.0))
        ci_lower = cates - z_crit * se_ind
        ci_upper = cates + z_crit * se_ind

        # Test against unit elasticity H0: theta = -1.0
        t_unit = (cates - (-1.0)) / se_ind
        p_unit = 2.0 * (1.0 - stats.norm.cdf(np.abs(t_unit)))

        # Test against H0: theta = 0.0
        t_zero = cates / se_ind
        p_zero = 2.0 * (1.0 - stats.norm.cdf(np.abs(t_zero)))

        regimes = []
        for c, pu in zip(cates, p_unit):
            if c >= 0.0:
                regimes.append("Anomalous (Positive)")
            elif pu >= alpha:
                regimes.append("Unit Elastic")
            elif c < -1.0:
                regimes.append("Elastic")
            else:
                regimes.append("Inelastic")

        return pd.DataFrame({
            "predicted_elasticity": cates,
            "std_error": se_ind,
            "ci_lower": ci_lower,
            "ci_upper": ci_upper,
            "t_stat_unit": t_unit,
            "p_value_unit": p_unit,
            "elasticity_regime": regimes,
            "is_statistically_significant": p_zero < alpha,
        })

    def cate_by_segment(
        self,
        segment_labels: Union[pd.Series, np.ndarray],
        X: Optional[Union[pd.DataFrame, np.ndarray]] = None,
        alpha: float = 0.05,
        min_sample_size: int = 20,
        max_rel_se: float = 0.40,
    ) -> pd.DataFrame:
        """Aggregate predicted elasticities by segment or product category with uncertainty & tests.

        Parameters
        ----------
        segment_labels : Union[pd.Series, np.ndarray]
            Categorical series (e.g. product_category_name or customer footfall tier).
        X : Optional[Union[pd.DataFrame, np.ndarray]]
            Covariate matrix. If None, uses in-sample predictions.
        alpha : float, default=0.05
            Significance level for confidence intervals and hypothesis tests.
        min_sample_size : int, default=20
            Threshold below which segment estimates are flagged as unsupported extrapolation.
        max_rel_se : float, default=0.40
            Threshold on SE / |mean| above which segment estimates are flagged as high uncertainty.

        Returns
        -------
        pd.DataFrame
            Summary table with sample count, mean, median, std, standard error, 95% CI,
            unit-elasticity boundary hypothesis tests, comparison to catalog ATE,
            extrapolation risk flags, and dynamic pricing recommendations.
        """
        if not self._is_fitted:
            raise ValueError("Model is not fitted. Call 'fit' before segment aggregation.")

        labels = np.asarray(segment_labels)
        if X is not None:
            cates = self.predict_cate(X)
            W_star = self._get_design_matrix(X)
        else:
            cates = self.cate_predictions_
            # Reconstruct training W_star if X is None
            # We can use in-sample predictions and stored beta
            if hasattr(self, "_in_sample_w_star"):
                W_star = self._in_sample_w_star
            else:
                # We need W_star for model standard errors
                # If X is None, we need X from training, but since not passed,
                # we can approximate or compute from labels if X was stored.
                # To be completely safe and avoid storing large X, let's allow X or use sample SE fallback.
                W_star = None

        if len(labels) != len(cates):
            raise ValueError(f"Length mismatch: segment_labels ({len(labels)}) != cates ({len(cates)})")

        z_crit = float(stats.norm.ppf(1.0 - alpha / 2.0))
        unique_segments = pd.Series(labels).unique()
        rows = []

        overall_ate = self.baseline_ate_ if self.baseline_ate_ is not None else float(np.mean(cates))
        overall_se = self.baseline_ate_se_ if self.baseline_ate_se_ is not None else 0.5110

        for seg in unique_segments:
            mask = (labels == seg)
            seg_cates = cates[mask]
            n_samples = len(seg_cates)
            mean_e = float(np.mean(seg_cates))
            median_e = float(np.median(seg_cates))
            std_e = float(np.std(seg_cates))
            min_e = float(np.min(seg_cates))
            max_e = float(np.max(seg_cates))

            # Compute standard error:
            # If W_star and var_beta_ are available, use model-based asymptotic standard error
            if W_star is not None and self.var_beta_ is not None:
                seg_w_star = W_star[mask]
                mean_w = np.mean(seg_w_star, axis=0)
                model_var = float(mean_w @ self.var_beta_ @ mean_w)
                se_e = float(np.sqrt(max(model_var, 1e-14)))
            else:
                # Fallback to sample standard error of the mean
                se_e = std_e / np.sqrt(n_samples) if n_samples > 1 else 0.5

            ci_l = mean_e - z_crit * se_e
            ci_u = mean_e + z_crit * se_e

            # Unit elasticity test against H0: theta = -1.0
            t_unit = (mean_e - (-1.0)) / (se_e if se_e > 1e-6 else 1e-6)
            p_unit = 2.0 * (1.0 - stats.norm.cdf(abs(t_unit)))

            # Comparison to overall catalog ATE
            se_diff = np.sqrt(se_e**2 + overall_se**2)
            t_vs_overall = (mean_e - overall_ate) / se_diff
            p_vs_overall = 2.0 * (1.0 - stats.norm.cdf(abs(t_vs_overall)))
            is_heterogeneous = bool(p_vs_overall < alpha)

            # Economic regime classification
            if mean_e >= 0.0:
                regime = "Anomalous (Positive)"
            elif p_unit >= alpha:
                regime = "Unit Elastic"
            elif mean_e < -1.2:
                regime = "Highly Elastic"
            elif mean_e < -1.0:
                regime = "Elastic"
            else:
                regime = "Inelastic"

            # Extrapolation and risk checks
            reasons = []
            if n_samples < min_sample_size:
                reasons.append(f"Small sample size (N={n_samples} < {min_sample_size})")
            rel_se = se_e / (abs(mean_e) + 1e-8)
            if rel_se > max_rel_se:
                reasons.append(f"High relative SE ({rel_se:.2f} > {max_rel_se:.2f})")
            if ci_l <= 0.0 <= ci_u:
                reasons.append("CI spans zero (insignificant sign)")
            if mean_e >= 0.0:
                reasons.append("Anomalous upward-sloping demand")

            is_extrap = len(reasons) > 0
            extrap_str = "; ".join(reasons) if is_extrap else "Supported"

            # Strategic pricing recommendation
            if mean_e < -1.2:
                strat = "Highly Elastic: Aggressive discounting generates high unit surge"
            elif regime == "Unit Elastic":
                strat = "Unit Elastic: Moderate promotions preserve revenue balance"
            elif -1.0 <= mean_e < 0.0:
                strat = "Inelastic: Avoid deep discounting; price increases expand margin"
            else:
                strat = "Anomalous/Prestige: Demand increases with price (Veblen/Giffen effect)"

            rows.append({
                "segment": seg,
                "sample_count": n_samples,
                "mean_elasticity": mean_e,
                "median_elasticity": median_e,
                "std_elasticity": std_e,
                "se_elasticity": se_e,
                "ci_lower": ci_l,
                "ci_upper": ci_u,
                "min_elasticity": min_e,
                "max_elasticity": max_e,
                "t_stat_unit": t_unit,
                "p_value_unit": p_unit,
                "elasticity_regime": regime,
                "t_stat_vs_overall": t_vs_overall,
                "p_value_vs_overall": p_vs_overall,
                "is_heterogeneous_vs_overall": is_heterogeneous,
                "is_unsupported_extrapolation": is_extrap,
                "extrapolation_reasons": extrap_str,
                "pricing_strategy": strat,
            })

        df_summary = pd.DataFrame(rows)
        return df_summary.sort_values(by="mean_elasticity", ascending=True).reset_index(drop=True)

    def check_support_and_extrapolation(
        self,
        X: Union[pd.DataFrame, np.ndarray],
        segment_labels: Optional[Union[pd.Series, np.ndarray]] = None,
        min_sample_size: int = 20,
        max_rel_se: float = 0.40,
    ) -> pd.DataFrame:
        """Validate covariate overlap and flag unsupported extrapolations.

        Checks:
        1. Bounding box / empirical support violations: moderators falling outside training [min, max].
        2. Prediction uncertainty: relative standard error exceeding threshold.
        3. Ambiguous sign: confidence interval containing zero.
        4. Subgroup sparsity: fewer samples than min_sample_size.

        Parameters
        ----------
        X : Union[pd.DataFrame, np.ndarray]
            Covariate matrix to evaluate.
        segment_labels : Optional[Union[pd.Series, np.ndarray]]
            Optional segment labels for group-level sparsity checks.
        min_sample_size : int, default=20
            Minimum samples required per segment.
        max_rel_se : float, default=0.40
            Maximum allowable relative standard error.

        Returns
        -------
        pd.DataFrame
            Diagnostics table per sample with extrapolation flags, reasons, and bounds violations.
        """
        if not self._is_fitted:
            raise ValueError("Model is not fitted. Call 'fit' before checking support.")

        W_arr, names = self._extract_moderators(X)
        n_samples = W_arr.shape[0]

        pred_df = self.predict_cate_with_uncertainty(X)
        cates = pred_df["predicted_elasticity"].values
        ses = pred_df["std_error"].values
        ci_lowers = pred_df["ci_lower"].values
        ci_uppers = pred_df["ci_upper"].values

        # Group counts if labels provided
        seg_counts: Dict[Any, int] = {}
        if segment_labels is not None:
            lbl_arr = np.asarray(segment_labels)
            for lbl in lbl_arr:
                seg_counts[lbl] = seg_counts.get(lbl, 0) + 1

        is_extrap = []
        reasons_list = []
        bounds_violated_count = []

        for i in range(n_samples):
            flags = []
            violated = 0
            # 1. Moderator bounds check
            for j, col_name in enumerate(names):
                val = W_arr[i, j]
                ref = self.moderator_ranges_.get(col_name)
                if ref is not None:
                    if val < ref["min"] - 1e-8 or val > ref["max"] + 1e-8:
                        violated += 1
                        flags.append(f"{col_name}={val:.2f} out of [{ref['min']:.2f}, {ref['max']:.2f}]")

            # 2. Relative SE check
            rel_se = ses[i] / (abs(cates[i]) + 1e-8)
            if rel_se > max_rel_se:
                flags.append(f"High rel SE ({rel_se:.2f})")

            # 3. CI spans zero check
            if ci_lowers[i] <= 0.0 <= ci_uppers[i]:
                flags.append("CI spans zero")

            # 4. Anomalous positive elasticity
            if cates[i] >= 0.0:
                flags.append("Anomalous positive elasticity")

            # 5. Segment sparsity
            if segment_labels is not None:
                lbl = segment_labels.iloc[i] if hasattr(segment_labels, "iloc") else segment_labels[i]
                count = seg_counts.get(lbl, 0)
                if count < min_sample_size:
                    flags.append(f"Segment '{lbl}' has N={count} < {min_sample_size}")

            is_extrap.append(len(flags) > 0)
            reasons_list.append("; ".join(flags) if flags else "Supported")
            bounds_violated_count.append(violated)

        return pd.DataFrame({
            "predicted_elasticity": cates,
            "std_error": ses,
            "is_unsupported_extrapolation": is_extrap,
            "bounds_violated_features": bounds_violated_count,
            "extrapolation_reasons": reasons_list,
        })

    def evaluate_subgroup_stability(
        self,
        X: Union[pd.DataFrame, np.ndarray],
        T: Union[pd.Series, np.ndarray],
        Y: Union[pd.Series, np.ndarray],
        segment_labels: Union[pd.Series, np.ndarray],
        n_splits: int = 5,
        random_state: int = 42,
    ) -> pd.DataFrame:
        """Evaluate stability of subgroup elasticity estimates across cross-fitting folds.

        Parameters
        ----------
        X : Union[pd.DataFrame, np.ndarray]
            Confounder matrix.
        T : Union[pd.Series, np.ndarray]
            Treatment vector.
        Y : Union[pd.Series, np.ndarray]
            Outcome vector.
        segment_labels : Union[pd.Series, np.ndarray]
            Segment labels for each row.
        n_splits : int, default=5
            Number of stability validation folds.
        random_state : int, default=42
            Random seed for fold generation.

        Returns
        -------
        pd.DataFrame
            Stability metrics table containing fold mean, fold std, coefficient of variation (CV),
            sign consistency percentage, and stability classification.
        """
        from sklearn.model_selection import KFold

        labels = np.asarray(segment_labels)
        unique_segments = pd.Series(labels).unique()
        fold_estimates: Dict[Any, List[float]] = {seg: [] for seg in unique_segments}

        kf = KFold(n_splits=n_splits, shuffle=True, random_state=random_state)
        X_df = pd.DataFrame(X) if not isinstance(X, pd.DataFrame) else X
        T_ser = pd.Series(T) if not isinstance(T, pd.Series) else T
        Y_ser = pd.Series(Y) if not isinstance(Y, pd.Series) else Y

        for fold_idx, (tr_idx, _) in enumerate(kf.split(X_df)):
            X_tr = X_df.iloc[tr_idx]
            T_tr = T_ser.iloc[tr_idx]
            Y_tr = Y_ser.iloc[tr_idx]

            dml_f = DoubleMachineLearning(n_folds=3, random_state=random_state + fold_idx)
            cate_f = CATEEstimator(
                dml_base=dml_f,
                moderator_cols=self.moderator_cols,
                alpha=self.alpha,
                include_intercept=self.include_intercept,
                random_state=random_state,
            )
            cate_f.fit(X_tr, T_tr, Y_tr)

            preds_f = cate_f.predict_cate(X_df)
            for seg in unique_segments:
                seg_mask = (labels == seg)
                if np.sum(seg_mask) > 0:
                    fold_estimates[seg].append(float(np.mean(preds_f[seg_mask])))
                else:
                    fold_estimates[seg].append(np.nan)

        rows = []
        for seg in unique_segments:
            vals = [v for v in fold_estimates[seg] if not np.isnan(v)]
            if not vals:
                continue
            mean_val = float(np.mean(vals))
            std_val = float(np.std(vals))
            cv = float(std_val / (abs(mean_val) + 1e-8))
            min_v = float(np.min(vals))
            max_v = float(np.max(vals))
            neg_pct = float(np.mean([v < 0 for v in vals]) * 100.0)

            # Stability rating
            if cv < 0.25 and neg_pct == 100.0:
                rating = "HIGHLY_STABLE"
            elif cv < 0.40 and neg_pct >= 80.0:
                rating = "MODERATELY_STABLE"
            else:
                rating = "UNSTABLE"

            rows.append({
                "segment": seg,
                "mean_across_folds": mean_val,
                "std_across_folds": std_val,
                "cv_dispersion": cv,
                "min_fold": min_v,
                "max_fold": max_v,
                "negative_elasticity_pct": neg_pct,
                "stability_rating": rating,
            })

        df_stab = pd.DataFrame(rows)
        return df_stab.sort_values(by="mean_across_folds", ascending=True).reset_index(drop=True)

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
            "baseline_ate": self.baseline_ate_,
            "baseline_ate_se": self.baseline_ate_se_,
            "baseline_ate_ci": self.baseline_ate_ci_,
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
            f"Baseline Catalog ATE (Anchor):              {self.baseline_ate_:.4f} (SE: {self.baseline_ate_se_:.4f})",
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
