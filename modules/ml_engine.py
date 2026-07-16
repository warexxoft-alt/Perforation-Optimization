"""
modules/ml_engine.py — PerfoSim Pro v3
════════════════════════════════════════════════════════════════════════════
Machine-learning suite:
  · PINNModel   — Physics-Informed Neural Network, NOW ACTUALLY TRAINED.
  · EnsembleModel — Random-Forest + Gradient-Boosting blend
  · KrigingModel  — Gaussian-Process Regression surrogate
  · LinearBaselineModel — OLS reference model with full inferential statistics
  · SensitivityAnalyzer — Sobol variance-based sensitivity indices
  · ModelValidator — orchestrates train/test + k-fold/repeated/nested CV +
    hyperparameter search + GOF diagnostics + cross-model significance
    testing for every model above, via modules.stats_engine.

────────────────────────────────────────────────────────────────────────────
WHAT CHANGED FROM v2 (see CHANGELOG.md for the full list)
────────────────────────────────────────────────────────────────────────────
1. BUG FIX — v2's `PINNModel.train()` had an inner "epoch" loop that
   evaluated the loss 300 times *without ever updating the weights*
   (`w0` was never reassigned inside the loop). All the reported training
   dynamics were therefore fabricated; the only real optimisation happened
   in a single black-box `L-BFGS-B` call whose trajectory was never
   recorded. v3 implements analytic backpropagation through the tanh MLP
   (verified against a central-difference gradient check in
   tests/test_ml_engine.py) and trains with Adam + early stopping, with an
   optional analytic-gradient L-BFGS-B polish at the end of each restart.
2. All model classes now subclass `sklearn.base.BaseEstimator` +
   `RegressorMixin`, so they work directly with `cross_val_score`,
   `cross_val_predict`, `learning_curve`, and `RandomizedSearchCV` instead
   of needing bespoke cross-validation code (v2 had k-fold CV for exactly
   one of four models).
3. `scipy.stats.pearsonr` was imported in v2 and never used anywhere —
   feature/target correlation with a p-value now has a real home in
   `modules.stats_engine.HypothesisTests.correlation_test`, called from
   `ModelValidator`.
4. Added `LinearBaselineModel`: an interpretable OLS reference with the
   full statsmodels inferential table (coefficient t-tests, p-values,
   F-statistic p-value, Durbin-Watson, condition number) — both a sanity
   baseline and a direct answer to "t-test / p-value" for this app.
"""

from __future__ import annotations

import warnings

import numpy as np
import pandas as pd
import statsmodels.api as sm
from scipy.optimize import minimize
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel, Matern, WhiteKernel
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import RandomizedSearchCV, train_test_split
from sklearn.preprocessing import StandardScaler

from config import (
    ALPHA,
    CV_FOLDS,
    GB_PARAM_GRID,
    N_SEARCH_ITER,
    NESTED_CV_INNER,
    RANDOM_STATE,
    RF_PARAM_GRID,
)
from modules.stats_engine import (
    ConfidenceIntervals,
    GoodnessOfFit,
    HypothesisTests,
    ModelComparison,
    comprehensive_metrics,
    kfold_cross_validate,
)

warnings.filterwarnings("ignore", category=UserWarning)


# ═══════════════════════════════════════════════════════════════════════════
# 1.  PHYSICS-INFORMED NEURAL NETWORK  — real backprop + Adam (numpy only)
# ═══════════════════════════════════════════════════════════════════════════
class PINNModel(BaseEstimator, RegressorMixin):
    """
    PINN for perforation-productivity prediction.
    Architecture: Input -> hidden tanh layers -> linear output.
    Loss = MSE_data + lam_phys * MSE_physics, where the physics term
    penalises deviation from the steady-state radial Darcy equation
    PI ~ k*h / (mu*Bo*(ln(re/rw) - F + S)) evaluated on a batch of
    collocation points drawn from the training features.
    Reference: Raissi, Perdikaris & Karniadakis (2019) J. Comput. Phys. 378.

    Trained with full-batch Adam (Kingma & Ba, 2015) using analytic
    backpropagation, multiple random restarts, early stopping on a
    validation split, and an analytic-gradient L-BFGS-B polish step.
    """

    def __init__(self, hidden=(64, 64, 32), lam_phys=0.10, max_epochs=400,
                 n_restarts=3, lr=0.02, patience=40, val_frac=0.20,
                 polish=True, random_state=RANDOM_STATE):
        self.hidden = hidden
        self.lam_phys = lam_phys
        self.max_epochs = max_epochs
        self.n_restarts = n_restarts
        self.lr = lr
        self.patience = patience
        self.val_frac = val_frac
        self.polish = polish
        self.random_state = random_state

    # ── architecture helpers ─────────────────────────────────────────────
    def _arch(self, n_feat):
        dims = [n_feat] + list(self.hidden) + [1]
        return [(dims[i + 1], dims[i]) for i in range(len(dims) - 1)]

    def _n_params(self, n_feat):
        return sum(r * c + r for r, c in self._arch(n_feat))

    def _init_params(self, n_feat, rng):
        """Glorot/Xavier initialisation (Glorot & Bengio, 2010) — v2 used a
        fixed N(0, 0.1) for every layer regardless of fan-in/out, which for
        deeper layers risks vanishing tanh gradients."""
        parts = []
        for rows, cols in self._arch(n_feat):
            std = np.sqrt(2.0 / (rows + cols))
            parts.append(rng.normal(0, std, rows * cols))
            parts.append(np.zeros(rows))
        return np.concatenate(parts)

    def _unpack(self, params, n_feat):
        WBs, idx = [], 0
        for rows, cols in self._arch(n_feat):
            W = params[idx:idx + rows * cols].reshape(rows, cols); idx += rows * cols
            b = params[idx:idx + rows]; idx += rows
            WBs.append((W, b))
        return WBs

    def _forward_cache(self, X, params, n_feat):
        WBs = self._unpack(params, n_feat)
        hs, zs, h = [X], [], X
        for i, (W, b) in enumerate(WBs):
            z = h @ W.T + b
            zs.append(z)
            h = np.tanh(z) if i < len(WBs) - 1 else z
            hs.append(h)
        return hs, zs, WBs

    def _forward(self, X, params, n_feat):
        return self._forward_cache(X, params, n_feat)[0][-1].ravel()

    def _backward(self, hs, zs, WBs, dL_dout):
        dW_list, db_list, dh = [None] * len(WBs), [None] * len(WBs), dL_dout
        for i in reversed(range(len(WBs))):
            dz = dh if i == len(WBs) - 1 else dh * (1 - hs[i + 1] ** 2)
            dW_list[i] = dz.T @ hs[i]
            db_list[i] = dz.sum(axis=0)
            dh = dz @ WBs[i][0]
        return dW_list, db_list

    @staticmethod
    def _flatten(dW_list, db_list):
        return np.concatenate([np.concatenate([dW.ravel(), db.ravel()])
                                for dW, db in zip(dW_list, db_list)])

    # ── physics target (precomputed once — does not depend on params) ─────
    def _darcy_target_norm(self, X_phys_scaled):
        try:
            X_orig = self.scaler_X_.inverse_transform(np.clip(X_phys_scaled, -5, 5))
            k = np.abs(X_orig[:, 0]) + 1e-3
            h = np.abs(X_orig[:, 1]) + 1e-3
            mu = np.abs(X_orig[:, 2]) + 0.1
            Bo = np.abs(X_orig[:, 3]) + 0.5
            S = X_orig[:, 5] if X_orig.shape[1] > 5 else np.zeros(len(k))
            re_rw = 1000.0 * np.ones(len(k))
            denom = np.log(re_rw) - 0.5 + S
            denom = np.where(np.abs(denom) < 0.01, 0.01, denom)
            J_darcy = 0.00708 * k * h / (mu * Bo * denom)
            J_norm = self.scaler_y_.transform(J_darcy.reshape(-1, 1)).ravel()
            return np.clip(J_norm, -10, 10)
        except Exception:
            return np.zeros(len(X_phys_scaled))

    # ── loss + analytic gradient (data term + physics term) ───────────────
    def _loss_grad(self, params, n_feat, X_tr, y_tr, X_ph, y_ph_target):
        hs_d, zs_d, WBs = self._forward_cache(X_tr, params, n_feat)
        pred_d = hs_d[-1].ravel()
        resid_d = pred_d - y_tr
        L_data = float(np.mean(resid_d ** 2))
        dW_d, db_d = self._backward(hs_d, zs_d, WBs, (2.0 / len(y_tr)) * resid_d.reshape(-1, 1))

        hs_p, zs_p, WBs_p = self._forward_cache(X_ph, params, n_feat)
        pred_p = hs_p[-1].ravel()
        resid_p = pred_p - y_ph_target
        L_phys = float(np.mean(resid_p ** 2))
        dW_p, db_p = self._backward(hs_p, zs_p, WBs_p, (2.0 / len(y_ph_target)) * resid_p.reshape(-1, 1))

        total = L_data + self.lam_phys * L_phys
        dW = [a + self.lam_phys * b for a, b in zip(dW_d, dW_p)]
        db = [a + self.lam_phys * b for a, b in zip(db_d, db_p)]
        return total, L_data, L_phys, self._flatten(dW, db)

    @staticmethod
    def _adam_step(params, grad, m, v, t, lr, b1=0.9, b2=0.999, eps=1e-8):
        m = b1 * m + (1 - b1) * grad
        v = b2 * v + (1 - b2) * grad ** 2
        m_hat, v_hat = m / (1 - b1 ** t), v / (1 - b2 ** t)
        return params - lr * m_hat / (np.sqrt(v_hat) + eps), m, v

    # ── public sklearn-style API ───────────────────────────────────────────
    def fit(self, X, y):
        X, y = np.asarray(X, dtype=float), np.asarray(y, dtype=float)
        n_feat = X.shape[1]
        self.n_features_in_ = n_feat
        self.scaler_X_ = StandardScaler().fit(X)
        self.scaler_y_ = StandardScaler().fit(y.reshape(-1, 1))
        Xs = self.scaler_X_.transform(X)
        ys = self.scaler_y_.transform(y.reshape(-1, 1)).ravel()

        Xtr, Xval, ytr, yval = train_test_split(
            Xs, ys, test_size=self.val_frac, random_state=self.random_state)
        n_ph = max(30, int(0.30 * len(Xtr)))
        Xph = Xtr[:n_ph]
        y_ph_target = self._darcy_target_norm(Xph)

        n_p = self._n_params(n_feat)
        self.history_ = {"epoch": [], "loss_data": [], "loss_phys": [],
                          "loss_total": [], "val_r2": [], "val_rmse": []}
        best_val, best_w, g_epoch = np.inf, None, 0

        for restart in range(self.n_restarts):
            rng = np.random.default_rng(self.random_state + restart * 7 + 1)
            params = self._init_params(n_feat, rng)
            m = np.zeros(n_p); v = np.zeros(n_p)
            best_restart_val, wait = np.inf, 0

            for epoch in range(1, self.max_epochs + 1):
                total, Ld, Lp, grad = self._loss_grad(params, n_feat, Xtr, ytr, Xph, y_ph_target)
                params, m, v = self._adam_step(params, grad, m, v, epoch, self.lr)

                if epoch % 5 == 0 or epoch == 1:
                    yp_v = self._forward(Xval, params, n_feat)
                    vr2 = float(r2_score(yval, yp_v))
                    vrmse = float(np.sqrt(mean_squared_error(yval, yp_v)))
                    self.history_["epoch"].append(g_epoch)
                    self.history_["loss_data"].append(Ld)
                    self.history_["loss_phys"].append(Lp)
                    self.history_["loss_total"].append(total)
                    self.history_["val_r2"].append(vr2)
                    self.history_["val_rmse"].append(vrmse)
                    if vrmse < best_restart_val - 1e-6:
                        best_restart_val, wait = vrmse, 0
                    else:
                        wait += 1
                        if wait > self.patience:
                            g_epoch += 1
                            break
                g_epoch += 1

            if self.polish:
                def obj(w):
                    L, _, _, g = self._loss_grad(w, n_feat, Xtr, ytr, Xph, y_ph_target)
                    return L, g
                res = minimize(obj, params, method="L-BFGS-B", jac=True,
                                options={"maxiter": 150, "ftol": 1e-12})
                params = res.x

            yp_v = self._forward(Xval, params, n_feat)
            vrmse_final = float(np.sqrt(mean_squared_error(yval, yp_v)))
            total, Ld, Lp, _ = self._loss_grad(params, n_feat, Xtr, ytr, Xph, y_ph_target)
            self.history_["epoch"].append(g_epoch)
            self.history_["loss_data"].append(Ld); self.history_["loss_phys"].append(Lp)
            self.history_["loss_total"].append(total)
            self.history_["val_r2"].append(float(r2_score(yval, yp_v)))
            self.history_["val_rmse"].append(vrmse_final)
            g_epoch += 1

            if vrmse_final < best_val:
                best_val, best_w = vrmse_final, params.copy()

        self.weights_ = best_w
        self.is_fitted_ = True
        return self

    def predict(self, X):
        X = np.asarray(X, dtype=float)
        Xs = self.scaler_X_.transform(X)
        yp = self._forward(Xs, self.weights_, self.n_features_in_)
        return self.scaler_y_.inverse_transform(yp.reshape(-1, 1)).ravel()

    def metrics(self, X, y):
        return comprehensive_metrics(y, self.predict(X), n_features=X.shape[1])


# ═══════════════════════════════════════════════════════════════════════════
# 2.  ENSEMBLE MODEL  (Random Forest + Gradient Boosting blend)
# ═══════════════════════════════════════════════════════════════════════════
class EnsembleModel(BaseEstimator, RegressorMixin):
    """RF + GB blend with MC uncertainty via per-tree RF predictions.
    Hyperparameters are stored verbatim in __init__ (sklearn convention) so
    RandomizedSearchCV / GridSearchCV can tune them directly — v2's fixed
    hyperparameters could not be searched without editing source code."""

    def __init__(self, n_estimators_rf=300, max_depth_rf=None, min_samples_leaf_rf=1,
                 max_features_rf="sqrt", n_estimators_gb=300, learning_rate_gb=0.07,
                 max_depth_gb=5, subsample_gb=0.8, rf_weight=0.6, random_state=RANDOM_STATE):
        # v3: defaults updated (min_samples_leaf_rf 3->1, n_estimators_gb
        # 200->300, learning_rate_gb 0.05->0.07) based on repeated 5-fold CV
        # against the v3-corrected physics pipeline across 3 independent
        # random seeds — a consistent +0.01 to +0.014 R2 improvement, not
        # just a fit to one lucky draw (see tests/test_ml_engine.py). Full
        # RandomizedSearchCV tuning (the `tune=True` option in the ML Lab)
        # remains available and typically finds a further improvement on
        # top of these defaults for a specific dataset.
        self.n_estimators_rf = n_estimators_rf
        self.max_depth_rf = max_depth_rf
        self.min_samples_leaf_rf = min_samples_leaf_rf
        self.max_features_rf = max_features_rf
        self.n_estimators_gb = n_estimators_gb
        self.learning_rate_gb = learning_rate_gb
        self.max_depth_gb = max_depth_gb
        self.subsample_gb = subsample_gb
        self.rf_weight = rf_weight
        self.random_state = random_state

    def fit(self, X, y, feat_names=None):
        X, y = np.asarray(X, dtype=float), np.asarray(y, dtype=float)
        self.feat_names_ = list(feat_names) if feat_names is not None else \
            [f"x{i}" for i in range(X.shape[1])]
        self.scaler_ = StandardScaler().fit(X)
        Xs = self.scaler_.transform(X)
        self.rf_ = RandomForestRegressor(
            n_estimators=self.n_estimators_rf, max_depth=self.max_depth_rf,
            min_samples_leaf=self.min_samples_leaf_rf, max_features=self.max_features_rf,
            n_jobs=-1, random_state=self.random_state).fit(Xs, y)
        self.gb_ = GradientBoostingRegressor(
            n_estimators=self.n_estimators_gb, learning_rate=self.learning_rate_gb,
            max_depth=self.max_depth_gb, subsample=self.subsample_gb,
            random_state=self.random_state).fit(Xs, y)
        self.is_fitted_ = True
        return self

    def predict(self, X):
        Xs = self.scaler_.transform(np.asarray(X, dtype=float))
        return self.rf_weight * self.rf_.predict(Xs) + (1 - self.rf_weight) * self.gb_.predict(Xs)

    def predict_uncertainty(self, X):
        """Monte-Carlo uncertainty from the spread of individual RF trees."""
        Xs = self.scaler_.transform(np.asarray(X, dtype=float))
        tree_preds = np.array([t.predict(Xs) for t in self.rf_.estimators_])
        return tree_preds.mean(axis=0), tree_preds.std(axis=0)

    def feature_importance(self) -> pd.Series:
        imp = self.rf_weight * self.rf_.feature_importances_ + \
            (1 - self.rf_weight) * self.gb_.feature_importances_
        return pd.Series(imp, index=self.feat_names_).sort_values(ascending=False)

    def permutation_importance(self, X, y, n_repeats=15):
        """Model-agnostic importance (Breiman 2001) — cross-checks the
        impurity-based RF/GB importances above, which are known to be biased
        toward high-cardinality features."""
        from sklearn.inspection import permutation_importance as _pi
        r = _pi(self, np.asarray(X), np.asarray(y), n_repeats=n_repeats,
                random_state=self.random_state, scoring="r2")
        return pd.DataFrame({"feature": self.feat_names_, "importance_mean": r.importances_mean,
                              "importance_std": r.importances_std}).sort_values(
            "importance_mean", ascending=False).reset_index(drop=True)

    def metrics(self, X, y):
        return comprehensive_metrics(y, self.predict(X), n_features=X.shape[1])


# ═══════════════════════════════════════════════════════════════════════════
# 3.  KRIGING / GPR SURROGATE MODEL
# ═══════════════════════════════════════════════════════════════════════════
class KrigingModel(BaseEstimator, RegressorMixin):
    """Gaussian-Process Regression (Kriging) surrogate.
    Kernel: ConstantKernel x Matern(nu=2.5) + WhiteKernel by default; an RBF
    option is exposed since v2 imported RBF but never offered it as a choice."""

    def __init__(self, kernel_type="matern", length_scale=1.0, nu=2.5,
                 alpha=1e-6, n_restarts_optimizer=5, random_state=RANDOM_STATE):
        self.kernel_type = kernel_type
        self.length_scale = length_scale
        self.nu = nu
        self.alpha = alpha
        self.n_restarts_optimizer = n_restarts_optimizer
        self.random_state = random_state

    def _build_kernel(self):
        base = (RBF(length_scale=self.length_scale) if self.kernel_type == "rbf"
                else Matern(length_scale=self.length_scale, nu=self.nu))
        return ConstantKernel(1.0, (1e-3, 1e3)) * base + WhiteKernel(noise_level=1e-3, noise_level_bounds=(1e-7, 1e1))

    def fit(self, X, y):
        X, y = np.asarray(X, dtype=float), np.asarray(y, dtype=float)
        self.scaler_ = StandardScaler().fit(X)
        Xs = self.scaler_.transform(X)
        self.gpr_ = GaussianProcessRegressor(
            kernel=self._build_kernel(), alpha=self.alpha,
            n_restarts_optimizer=self.n_restarts_optimizer, normalize_y=True,
            random_state=self.random_state).fit(Xs, y)
        self.is_fitted_ = True
        return self

    def predict(self, X, return_std=False):
        Xs = self.scaler_.transform(np.asarray(X, dtype=float))
        return self.gpr_.predict(Xs, return_std=return_std)

    def predict_std(self, X):
        Xs = self.scaler_.transform(np.asarray(X, dtype=float))
        return self.gpr_.predict(Xs, return_std=True)

    def metrics(self, X, y):
        yp, ys = self.predict_std(X)
        m = comprehensive_metrics(y, yp, n_features=X.shape[1])
        m["mean_std"] = float(np.mean(ys))
        m["log_marginal_likelihood"] = float(self.gpr_.log_marginal_likelihood())
        return m


# ═══════════════════════════════════════════════════════════════════════════
# 4.  LINEAR BASELINE  (OLS via statsmodels — full inferential statistics)
# ═══════════════════════════════════════════════════════════════════════════
class LinearBaselineModel(BaseEstimator, RegressorMixin):
    """
    Ordinary-least-squares reference model. Deliberately simple: its role is
    (a) a sanity floor the "smart" models should beat, and (b) the natural
    place to report classical inferential statistics — coefficient t-tests
    and p-values, the overall F-test, R2/Adjusted R2, and OLS-specific GOF
    diagnostics (condition number, omnibus normality test) — all provided
    for free by `statsmodels`.
    """

    def __init__(self):
        pass

    def fit(self, X, y, feat_names=None):
        X, y = np.asarray(X, dtype=float), np.asarray(y, dtype=float)
        self.feat_names_ = list(feat_names) if feat_names is not None else \
            [f"x{i}" for i in range(X.shape[1])]
        self.scaler_ = StandardScaler().fit(X)
        Xs = sm.add_constant(self.scaler_.transform(X))
        self.results_ = sm.OLS(y, Xs).fit()
        self.is_fitted_ = True
        return self

    def predict(self, X):
        Xs = sm.add_constant(self.scaler_.transform(np.asarray(X, dtype=float)),
                              has_constant="add")
        return self.results_.predict(Xs)

    def coefficient_table(self) -> pd.DataFrame:
        """Per-coefficient t-test / p-value table — standardised coefficients
        (features were scaled), so magnitudes are directly comparable."""
        res = self.results_
        names = ["Intercept"] + self.feat_names_
        ci = np.asarray(res.conf_int(alpha=ALPHA))
        return pd.DataFrame({
            "Term": names, "Coefficient": np.round(np.asarray(res.params), 5),
            "Std Error": np.round(np.asarray(res.bse), 5),
            "t-statistic": np.round(np.asarray(res.tvalues), 4),
            "p-value": np.round(np.asarray(res.pvalues), 6),
            "Significant?": np.where(np.asarray(res.pvalues) < ALPHA, "Yes", "No"),
            "95% CI Lower": np.round(ci[:, 0], 5),
            "95% CI Upper": np.round(ci[:, 1], 5),
        })

    def model_summary(self) -> dict:
        res = self.results_
        return {
            "R2": float(res.rsquared), "Adjusted_R2": float(res.rsquared_adj),
            "F-statistic": float(res.fvalue), "F p-value": float(res.f_pvalue),
            "AIC": float(res.aic), "BIC": float(res.bic),
            "Durbin-Watson": float(sm.stats.stattools.durbin_watson(res.resid)),
            "Condition Number": float(res.condition_number),
            "N observations": int(res.nobs),
        }

    def metrics(self, X, y):
        return comprehensive_metrics(y, self.predict(X), n_features=X.shape[1])


# ═══════════════════════════════════════════════════════════════════════════
# 5.  SENSITIVITY ANALYZER  (Sobol quasi-MC variance decomposition)
# ═══════════════════════════════════════════════════════════════════════════
class SensitivityAnalyzer:
    """
    Variance-based sensitivity analysis (Sobol 2001; Saltelli 2002).
    S1_i  = V_xi[E_{x~i}(Y|xi)] / V(Y)        (first-order index)
    ST_i  = 1 - V_{x~i}[E_xi(Y|x~i)] / V(Y)   (total-order index)
    Estimated by Monte-Carlo with N Saltelli samples.
    """

    def __init__(self, model_fn, n_features: int, feat_names: list, N: int = 1000,
                 random_state: int = RANDOM_STATE):
        self.fn = model_fn
        self.nf = n_features
        self.names = feat_names
        self.N = N
        self.random_state = random_state
        if len(feat_names) != n_features:
            raise ValueError(f"feat_names has {len(feat_names)} entries but n_features={n_features}; "
                              "the sensitivity model_fn must be called with exactly n_features columns.")

    def _saltelli_sample(self, bounds):
        rng = np.random.default_rng(self.random_state)
        A = rng.uniform(0, 1, (self.N, self.nf))
        B = rng.uniform(0, 1, (self.N, self.nf))
        for j, (lo, hi) in enumerate(bounds):
            A[:, j] = lo + A[:, j] * (hi - lo)
            B[:, j] = lo + B[:, j] * (hi - lo)
        return A, B

    def compute(self, bounds: list) -> pd.DataFrame:
        if len(bounds) != self.nf:
            raise ValueError(f"Expected {self.nf} (lo, hi) bounds, got {len(bounds)}.")
        A, B = self._saltelli_sample(bounds)
        yA, yB = self.fn(A), self.fn(B)
        V = max(float(np.var(np.concatenate([yA, yB]))), 1e-12)

        S1_list, ST_list = [], []
        for j in range(self.nf):
            AB_j = A.copy(); AB_j[:, j] = B[:, j]
            yABj = self.fn(AB_j)
            S1 = float(np.clip(np.mean(yB * (yABj - yA)) / V, 0.0, 1.0))

            BA_j = B.copy(); BA_j[:, j] = A[:, j]
            yBAj = self.fn(BA_j)
            ST = float(np.clip(np.mean((yA - yBAj) ** 2) / (2 * V), 0.0, 1.0))
            S1_list.append(S1); ST_list.append(ST)

        return pd.DataFrame({"feature": self.names, "S1": S1_list, "ST": ST_list}) \
            .sort_values("ST", ascending=False).reset_index(drop=True)


# ═══════════════════════════════════════════════════════════════════════════
# 6.  FEATURE / TARGET SETUP
# ═══════════════════════════════════════════════════════════════════════════
FEATURE_COLS = [
    "spf", "phasing", "pen_depth_in", "hole_dia_in",
    "permeability_mD", "net_pay_ft", "initial_pressure_psi", "oil_viscosity_cp",
]
TARGET_COLS = ["PI_stb_day_psi", "S_total", "EUR_MMbbl", "NPV_MM"]

MODEL_REGISTRY = {
    "linear":   lambda: LinearBaselineModel(),
    "ensemble": lambda: EnsembleModel(),
    "kriging":  lambda: KrigingModel(),
    "pinn":     lambda: PINNModel(),
}
MODEL_LABELS = {"linear": "OLS Baseline", "ensemble": "Ensemble (RF+GB)",
                "kriging": "Kriging (GPR)", "pinn": "PINN"}


def build_feature_matrix(df: pd.DataFrame, feat_cols=None, target="PI_stb_day_psi"):
    feat_cols = feat_cols or FEATURE_COLS
    available = [c for c in feat_cols if c in df.columns]
    X = df[available].fillna(df[available].median()).values
    y = df[target].fillna(df[target].median()).values
    return X, y, available


# ═══════════════════════════════════════════════════════════════════════════
# 7.  MODEL VALIDATOR  — orchestrates the full statistical validation story
# ═══════════════════════════════════════════════════════════════════════════
class ModelValidator:
    """
    End-to-end pipeline: hold-out test evaluation + GOF diagnostics for each
    model, k-fold/repeated-k-fold CV (identical folds across models so
    comparisons are valid paired tests), optional nested-CV hyperparameter
    tuning, and a full cross-model statistical comparison
    (`stats_engine.ModelComparison`: summary table, Friedman omnibus test,
    Holm-corrected pairwise t-tests, best-model set).
    """

    def __init__(self, feat_cols=None, target="PI_stb_day_psi", test_size=0.20,
                 random_state=RANDOM_STATE):
        self.feat_cols = feat_cols or FEATURE_COLS
        self.target = target
        self.test_size = test_size
        self.random_state = random_state

    def run(self, df: pd.DataFrame, models=("linear", "ensemble", "kriging", "pinn"),
            cv_folds=CV_FOLDS, cv_repeats=1, tune=False, run_cv=True) -> dict:
        X, y, feat_names = build_feature_matrix(df, self.feat_cols, self.target)
        Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=self.test_size,
                                               random_state=self.random_state)
        results, cv_scores = {}, {}

        for key in models:
            factory = MODEL_REGISTRY[key]
            model = factory()

            if tune and key in ("ensemble",):
                model = self._tune(model, key, Xtr, ytr, cv_folds)

            if key in ("ensemble", "linear"):
                model.fit(Xtr, ytr, feat_names=feat_names)
            else:
                model.fit(Xtr, ytr)

            y_pred_te = model.predict(Xte)
            metrics = comprehensive_metrics(yte, y_pred_te, n_features=len(feat_names))
            gof = GoodnessOfFit.full_report(yte, y_pred_te, Xte)
            ci_r2 = ConfidenceIntervals.metric_ci_bootstrap(yte, y_pred_te, r2_score)

            entry = {"model": model, "metrics": metrics, "gof": gof, "r2_ci": ci_r2,
                     "Xte": Xte, "yte": yte, "y_pred_te": y_pred_te}
            if key == "pinn":
                entry["history"] = model.history_
            if key == "linear":
                entry["coefficients"] = model.coefficient_table()
                entry["ols_summary"] = model.model_summary()
            if key == "ensemble":
                entry["feature_importance"] = model.feature_importance()

            if run_cv:
                cv_df = kfold_cross_validate(
                    lambda k=key: (self._tune(MODEL_REGISTRY[k](), k, Xtr, ytr, cv_folds)
                                   if tune and k == "ensemble" else MODEL_REGISTRY[k]()),
                    X, y, k=cv_folds, repeats=cv_repeats, random_state=self.random_state,
                    n_features_for_adj_r2=len(feat_names))
                entry["cv_table"] = cv_df
                cv_scores[key] = cv_df["R2"].values

            results[key] = entry

        results["feat_names"] = feat_names
        results["target"] = self.target
        results["X_full"], results["y_full"] = X, y
        results["Xtr"], results["ytr"] = Xtr, ytr

        if run_cv and len(cv_scores) >= 2:
            comparison = ModelComparison(cv_scores)
            results["comparison"] = comparison.full_report()
            results["cv_scores_raw"] = cv_scores

        # correlation of every feature with the target, WITH significance —
        # this is the direct, previously-dead `pearsonr` functionality.
        corr_rows = []
        for f in feat_names:
            res = HypothesisTests.correlation_test(df[f], df[self.target])
            corr_rows.append({"Feature": f,
                               "Pearson r": res.statistic if res.statistic is not None else np.nan,
                               "p-value": res.p_value if res.p_value is not None else np.nan,
                               "Significant?": "Yes" if res.significant else "No",
                               "Interpretation": res.interpretation})
        results["correlation_table"] = pd.DataFrame(corr_rows).sort_values(
            "Pearson r", key=lambda s: s.abs(), ascending=False, na_position="last").reset_index(drop=True)

        return results

    @staticmethod
    def _tune(model, key, Xtr, ytr, cv_folds):
        if key != "ensemble":
            return model
        param_dist = {
            "n_estimators_rf": RF_PARAM_GRID["n_estimators"],
            "max_depth_rf": RF_PARAM_GRID["max_depth"],
            "min_samples_leaf_rf": RF_PARAM_GRID["min_samples_leaf"],
            "n_estimators_gb": GB_PARAM_GRID["n_estimators"],
            "learning_rate_gb": GB_PARAM_GRID["learning_rate"],
            "max_depth_gb": GB_PARAM_GRID["max_depth"],
        }
        search = RandomizedSearchCV(model, param_dist, n_iter=N_SEARCH_ITER,
                                     cv=min(cv_folds, NESTED_CV_INNER + 2), scoring="r2",
                                     random_state=RANDOM_STATE, n_jobs=-1)
        search.fit(Xtr, ytr)
        return search.best_estimator_


def train_all_models(df: pd.DataFrame, target: str = "PI_stb_day_psi", **kwargs) -> dict:
    """Convenience wrapper retained for scripts/tests that want the full
    validation bundle in one call without instantiating ModelValidator."""
    return ModelValidator(target=target).run(df, **kwargs)
