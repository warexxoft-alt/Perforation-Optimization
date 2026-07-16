"""
modules/stats_engine.py — PerfoSim Pro v3
════════════════════════════════════════════════════════════════════════════
Statistical validation engine.

This module centralises every hypothesis test, goodness-of-fit diagnostic,
confidence interval, and cross-validated model-comparison routine used
across the app, so that one consistent significance level, one consistent
multiple-comparison correction, and one consistent reporting format
(`TestResult`) is used everywhere rather than being reimplemented ad hoc on
each page.

v2 of this codebase performed zero hypothesis testing anywhere: model
"accuracy" was reported as a bare R² with no indication of whether it was
statistically distinguishable from a simpler baseline, no residual
diagnostics were run, and cross-validation existed for exactly one of four
models. This module — together with the validation pipeline in
`ml_engine.ModelValidator` — closes that gap.

References
----------
Shapiro & Wilk (1965) Biometrika 52(3/4) — normality test
Jarque & Bera (1980) Econ. Letters 6(3) — normality test (skew + kurtosis)
Durbin & Watson (1951) Biometrika 38(1/2) — residual autocorrelation
Breusch & Pagan (1979) Econometrica 47(5) — heteroscedasticity (LM test)
Nash & Sutcliffe (1970) J. Hydrol. 10(3) — model-fit efficiency (NSE)
Willmott (1981) Phys. Geog. 2(2) — index of agreement (d)
Holm (1979) Scand. J. Statist. 6(2) — step-down multiple-comparison correction
Demšar (2006) J. Mach. Learn. Res. 7 — Friedman/Nemenyi for comparing
    learners across resampled folds
Efron & Tibshirani (1993) "An Introduction to the Bootstrap" — percentile CI
"""

from __future__ import annotations

from dataclasses import dataclass, field
from itertools import combinations
from typing import Callable, Optional

import numpy as np
import pandas as pd
import statsmodels.api as sm
from scipy import stats
from sklearn.metrics import (
    explained_variance_score,
    max_error,
    mean_absolute_error,
    mean_absolute_percentage_error,
    mean_squared_error,
    r2_score,
)
from sklearn.model_selection import KFold, RepeatedKFold
from statsmodels.stats.diagnostic import het_breuschpagan
from statsmodels.stats.multitest import multipletests
from statsmodels.stats.stattools import durbin_watson, jarque_bera

from config import (
    ACCURACY_TOLERANCES,
    ALPHA,
    CV_FOLDS,
    MULTIPLE_COMPARISON_CORRECTION,
    N_BOOTSTRAP,
    RANDOM_STATE,
)


# ══════════════════════════════════════════════════════════════════════════
#  UNIFORM RESULT CONTAINER
# ══════════════════════════════════════════════════════════════════════════
@dataclass
class TestResult:
    """Uniform container for a single statistical hypothesis test."""
    name: str
    statistic: Optional[float]
    p_value: Optional[float]
    alpha: float = ALPHA
    df: Optional[float] = None
    interpretation: str = ""
    significant: bool = field(init=False, default=False)

    def __post_init__(self):
        self.significant = bool(self.p_value is not None and
                                 not np.isnan(self.p_value) and self.p_value < self.alpha)

    def to_dict(self) -> dict:
        return {
            "Test": self.name,
            "Statistic": None if self.statistic is None or (isinstance(self.statistic, float) and np.isnan(self.statistic)) else round(float(self.statistic), 4),
            "p-value": None if self.p_value is None or (isinstance(self.p_value, float) and np.isnan(self.p_value)) else round(float(self.p_value), 6),
            "df": self.df,
            "Alpha": self.alpha,
            "Significant?": "Yes" if self.significant else "No",
            "Interpretation": self.interpretation,
        }


def _clean(a) -> np.ndarray:
    a = np.asarray(a, dtype=float).ravel()
    return a[np.isfinite(a)]


# ══════════════════════════════════════════════════════════════════════════
#  1.  GOODNESS-OF-FIT / RESIDUAL DIAGNOSTICS
# ══════════════════════════════════════════════════════════════════════════
class GoodnessOfFit:
    """Residual and distributional goodness-of-fit diagnostics."""

    @staticmethod
    def shapiro_wilk(residuals) -> TestResult:
        r = _clean(residuals)
        if len(r) < 3:
            return TestResult("Shapiro-Wilk (normality)", None, None, interpretation="insufficient data (n<3)")
        if len(r) > 5000:                      # statistic is validated up to n=5000
            r = np.random.default_rng(RANDOM_STATE).choice(r, 5000, replace=False)
        W, p = stats.shapiro(r)
        interp = ("residuals are consistent with a normal distribution (fail to reject H0)" if p >= ALPHA
                  else "residuals deviate significantly from normality (reject H0)")
        return TestResult("Shapiro-Wilk (normality)", float(W), float(p), interpretation=interp)

    @staticmethod
    def kolmogorov_smirnov(residuals) -> TestResult:
        r = _clean(residuals)
        if len(r) < 3:
            return TestResult("Kolmogorov-Smirnov (vs Normal)", None, None, interpretation="insufficient data")
        z = (r - r.mean()) / (r.std(ddof=1) + 1e-12)
        D, p = stats.kstest(z, "norm")
        interp = ("consistent with normality" if p >= ALPHA else "deviates from normality")
        return TestResult("Kolmogorov-Smirnov (vs Normal)", float(D), float(p), interpretation=interp)

    @staticmethod
    def jarque_bera(residuals) -> TestResult:
        r = _clean(residuals)
        if len(r) < 8:
            return TestResult("Jarque-Bera (normality)", None, None, interpretation="insufficient data")
        jb, p, skew, kurt = jarque_bera(r)
        interp = f"skew={skew:.3f}, excess kurtosis={kurt - 3:.3f} — " + \
                 ("consistent with normality" if p >= ALPHA else "deviates from normality")
        return TestResult("Jarque-Bera (normality)", float(jb), float(p), interpretation=interp)

    @staticmethod
    def durbin_watson(residuals) -> TestResult:
        r = _clean(residuals)
        if len(r) < 3:
            return TestResult("Durbin-Watson (autocorrelation)", None, None, interpretation="insufficient data")
        dw = float(durbin_watson(r))
        if dw < 1.5:
            interp = f"DW={dw:.3f} < 1.5 — evidence of POSITIVE autocorrelation in residuals"
        elif dw > 2.5:
            interp = f"DW={dw:.3f} > 2.5 — evidence of NEGATIVE autocorrelation in residuals"
        else:
            interp = f"DW={dw:.3f} ≈ 2.0 — no strong evidence of residual autocorrelation"
        return TestResult("Durbin-Watson (autocorrelation)", dw, None, interpretation=interp)

    @staticmethod
    def breusch_pagan(residuals, X) -> TestResult:
        r = _clean(residuals)
        try:
            Xc = sm.add_constant(np.asarray(X, dtype=float))
            if len(Xc) != len(r):
                raise ValueError("X and residuals length mismatch")
            lm, lm_p, _, _ = het_breuschpagan(r, Xc)
            interp = ("residual variance appears constant across predictions (homoscedastic)" if lm_p >= ALPHA
                      else "evidence of heteroscedasticity — residual spread varies with fitted value")
            return TestResult("Breusch-Pagan (heteroscedasticity)", float(lm), float(lm_p), interpretation=interp)
        except Exception as exc:
            return TestResult("Breusch-Pagan (heteroscedasticity)", None, None,
                               interpretation=f"could not be computed ({exc})")

    @staticmethod
    def nash_sutcliffe_efficiency(y_true, y_pred) -> float:
        """NSE = 1 - SS_res/SS_tot. NSE=1 perfect, 0 = no better than the mean, <0 worse than the mean."""
        y_true, y_pred = _clean(y_true), _clean(y_pred)
        denom = np.sum((y_true - y_true.mean()) ** 2)
        return float(1 - np.sum((y_true - y_pred) ** 2) / denom) if denom > 1e-12 else float("nan")

    @staticmethod
    def willmott_index(y_true, y_pred) -> float:
        """Willmott's (1981) index of agreement d, bounded [0,1], less sensitive to outliers than R2."""
        y_true, y_pred = _clean(y_true), _clean(y_pred)
        num = np.sum((y_pred - y_true) ** 2)
        denom = np.sum((np.abs(y_pred - y_true.mean()) + np.abs(y_true - y_true.mean())) ** 2)
        return float(1 - num / denom) if denom > 1e-12 else float("nan")

    @classmethod
    def full_report(cls, y_true, y_pred, X=None) -> pd.DataFrame:
        """Complete residual-diagnostic table for a fitted model, ready to display or export."""
        y_true, y_pred = np.asarray(y_true, dtype=float), np.asarray(y_pred, dtype=float)
        resid = y_true - y_pred
        rows = [cls.shapiro_wilk(resid).to_dict(),
                cls.jarque_bera(resid).to_dict(),
                cls.kolmogorov_smirnov(resid).to_dict(),
                cls.durbin_watson(resid).to_dict()]
        if X is not None:
            rows.append(cls.breusch_pagan(resid, X).to_dict())
        return pd.DataFrame(rows)


# ══════════════════════════════════════════════════════════════════════════
#  2.  COMPREHENSIVE REGRESSION METRICS  ("accuracy and other performance
#      metrics")
# ══════════════════════════════════════════════════════════════════════════
def accuracy_within_tolerance(y_true, y_pred, tol: float) -> float:
    """Fraction of predictions within +/- tol*100% of the true value — the
    engineering-style 'accuracy' metric used when a classification accuracy
    is not meaningful for a continuous target."""
    y_true, y_pred = _clean(y_true), _clean(y_pred)
    denom = np.where(np.abs(y_true) < 1e-9, 1e-9, y_true)
    rel_err = np.abs((y_pred - y_true) / denom)
    return float(np.mean(rel_err <= tol))


def comprehensive_metrics(y_true, y_pred, n_features: Optional[int] = None) -> dict:
    """Full metric bundle: R2 (+ adjusted), RMSE, MAE, MAPE, MSE, explained
    variance, max error, NSE, Willmott's d, and tolerance-based accuracy."""
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    n = len(y_true)
    r2 = float(r2_score(y_true, y_pred))
    out = {
        "R2": r2,
        "RMSE": float(np.sqrt(mean_squared_error(y_true, y_pred))),
        "MAE": float(mean_absolute_error(y_true, y_pred)),
        "MAPE_%": float(mean_absolute_percentage_error(y_true, y_pred) * 100),
        "MSE": float(mean_squared_error(y_true, y_pred)),
        "Explained_Variance": float(explained_variance_score(y_true, y_pred)),
        "Max_Error": float(max_error(y_true, y_pred)),
        "NSE": GoodnessOfFit.nash_sutcliffe_efficiency(y_true, y_pred),
        "Willmott_d": GoodnessOfFit.willmott_index(y_true, y_pred),
    }
    if n_features is not None and n - n_features - 1 > 0:
        out["Adjusted_R2"] = float(1 - (1 - r2) * (n - 1) / (n - n_features - 1))
    else:
        out["Adjusted_R2"] = None
    for tol in ACCURACY_TOLERANCES:
        out[f"Accuracy@{int(tol*100)}%"] = accuracy_within_tolerance(y_true, y_pred, tol)
    return out


# ══════════════════════════════════════════════════════════════════════════
#  3.  CONFIDENCE INTERVALS
# ══════════════════════════════════════════════════════════════════════════
class ConfidenceIntervals:

    @staticmethod
    def t_interval(data, alpha: float = ALPHA) -> tuple:
        """Parametric CI for the mean using Student's t distribution."""
        a = _clean(data)
        if len(a) < 2:
            return (float("nan"), float("nan"))
        m, se = a.mean(), stats.sem(a)
        h = se * stats.t.ppf(1 - alpha / 2, len(a) - 1)
        return (float(m - h), float(m + h))

    @staticmethod
    def bootstrap_ci(data, statistic: Callable = np.mean, n_boot: int = N_BOOTSTRAP,
                      alpha: float = ALPHA, seed: int = RANDOM_STATE) -> dict:
        """Percentile bootstrap CI for an arbitrary statistic (Efron & Tibshirani 1993)."""
        a = _clean(data)
        rng = np.random.default_rng(seed)
        if len(a) < 2:
            return {"point": float("nan"), "lower": float("nan"), "upper": float("nan"), "n_boot": n_boot}
        idx = rng.integers(0, len(a), size=(n_boot, len(a)))
        boot_stats = statistic(a[idx], axis=1) if statistic in (np.mean, np.median, np.std) else \
            np.array([statistic(a[i]) for i in idx])
        lo, hi = np.percentile(boot_stats, [100 * alpha / 2, 100 * (1 - alpha / 2)])
        return {"point": float(statistic(a)), "lower": float(lo), "upper": float(hi), "n_boot": n_boot}

    @staticmethod
    def metric_ci_bootstrap(y_true, y_pred, metric_fn: Callable, n_boot: int = N_BOOTSTRAP,
                             alpha: float = ALPHA, seed: int = RANDOM_STATE) -> dict:
        """Bootstrap CI for any (y_true, y_pred) -> scalar metric, resampling paired indices."""
        y_true, y_pred = np.asarray(y_true), np.asarray(y_pred)
        n = len(y_true)
        rng = np.random.default_rng(seed)
        point = float(metric_fn(y_true, y_pred))
        boots = np.empty(n_boot)
        for b in range(n_boot):
            idx = rng.integers(0, n, n)
            boots[b] = metric_fn(y_true[idx], y_pred[idx])
        lo, hi = np.percentile(boots, [100 * alpha / 2, 100 * (1 - alpha / 2)])
        return {"point": point, "lower": float(lo), "upper": float(hi), "n_boot": n_boot}


# ══════════════════════════════════════════════════════════════════════════
#  4.  HYPOTHESIS TESTS
# ══════════════════════════════════════════════════════════════════════════
class HypothesisTests:
    """General-purpose t-tests, non-parametric alternatives, ANOVA, and
    correlation-significance tests, all returning a `TestResult`."""

    @staticmethod
    def one_sample_ttest(sample, popmean: float, alpha: float = ALPHA) -> TestResult:
        a = _clean(sample)
        t, p = stats.ttest_1samp(a, popmean)
        interp = (f"sample mean ({a.mean():.4g}) is significantly different from {popmean:.4g}"
                  if p < alpha else f"no significant difference from {popmean:.4g} detected")
        return TestResult(f"One-sample t-test (H0: mean = {popmean:.4g})", float(t), float(p),
                           alpha=alpha, df=len(a) - 1, interpretation=interp)

    @staticmethod
    def independent_ttest(a, b, alpha: float = ALPHA, equal_var: bool = False) -> TestResult:
        a, b = _clean(a), _clean(b)
        t, p = stats.ttest_ind(a, b, equal_var=equal_var)
        interp = "means differ significantly" if p < alpha else "no significant difference between means"
        welch = "" if equal_var else " (Welch, unequal variance)"
        return TestResult(f"Independent t-test{welch}", float(t), float(p), alpha=alpha,
                           df=len(a) + len(b) - 2, interpretation=interp)

    @staticmethod
    def paired_ttest(a, b, alpha: float = ALPHA) -> TestResult:
        a, b = np.asarray(a, dtype=float), np.asarray(b, dtype=float)
        mask = np.isfinite(a) & np.isfinite(b)
        a, b = a[mask], b[mask]
        t, p = stats.ttest_rel(a, b)
        d = float(np.mean(a - b) / (np.std(a - b, ddof=1) + 1e-12))   # Cohen's d for paired samples
        interp = (f"significant difference (paired), Cohen's d={d:.3f}" if p < alpha
                  else f"no significant paired difference (Cohen's d={d:.3f})")
        return TestResult("Paired t-test", float(t), float(p), alpha=alpha, df=len(a) - 1, interpretation=interp)

    @staticmethod
    def wilcoxon_signed_rank(a, b, alpha: float = ALPHA) -> TestResult:
        """Non-parametric alternative to the paired t-test (no normality assumption)."""
        a, b = np.asarray(a, dtype=float), np.asarray(b, dtype=float)
        mask = np.isfinite(a) & np.isfinite(b)
        a, b = a[mask], b[mask]
        if np.allclose(a, b):
            return TestResult("Wilcoxon signed-rank", 0.0, 1.0, alpha=alpha, interpretation="samples identical")
        try:
            stat, p = stats.wilcoxon(a, b)
        except ValueError as exc:
            return TestResult("Wilcoxon signed-rank", None, None, alpha=alpha, interpretation=str(exc))
        interp = "significant difference in distributions" if p < alpha else "no significant difference"
        return TestResult("Wilcoxon signed-rank", float(stat), float(p), alpha=alpha, interpretation=interp)

    @staticmethod
    def one_way_anova(*groups, alpha: float = ALPHA) -> TestResult:
        groups = [_clean(g) for g in groups]
        F, p = stats.f_oneway(*groups)
        interp = "at least one group mean differs significantly" if p < alpha else "no significant difference among group means"
        df_between = len(groups) - 1
        return TestResult("One-way ANOVA", float(F), float(p), alpha=alpha, df=df_between, interpretation=interp)

    @staticmethod
    def friedman_test(*score_arrays, alpha: float = ALPHA) -> TestResult:
        """Friedman test — the standard omnibus test for comparing >=3 models
        evaluated on the SAME cross-validation folds (Demsar 2006)."""
        arrays = [np.asarray(a) for a in score_arrays]
        stat, p = stats.friedmanchisquare(*arrays)
        interp = ("at least one model's CV performance differs significantly" if p < alpha
                  else "no significant difference detected among models' CV performance")
        return TestResult("Friedman test (across CV folds)", float(stat), float(p),
                           alpha=alpha, df=len(arrays) - 1, interpretation=interp)

    @staticmethod
    def correlation_test(x, y, method: str = "pearson", alpha: float = ALPHA) -> TestResult:
        x, y = np.asarray(x, dtype=float), np.asarray(y, dtype=float)
        mask = np.isfinite(x) & np.isfinite(y)
        x, y = x[mask], y[mask]
        if len(x) < 3 or np.std(x) < 1e-12 or np.std(y) < 1e-12:
            return TestResult(f"{method.title()} correlation", None, None, alpha=alpha,
                               interpretation="undefined (insufficient data or zero variance)")
        if method == "spearman":
            r, p = stats.spearmanr(x, y)
            label = "Spearman ρ"
        else:
            r, p = stats.pearsonr(x, y)
            label = "Pearson r"
        if r is None or np.isnan(r):
            return TestResult(label, None, None, alpha=alpha, interpretation="undefined (zero variance)")
        strength = ("negligible", "weak", "moderate", "strong", "very strong")[
            min(4, int(abs(r) // 0.2))]
        interp = (f"{strength} {'positive' if r >= 0 else 'negative'} correlation, "
                  + ("statistically significant" if p < alpha else "not statistically significant"))
        return TestResult(label, float(r), float(p), alpha=alpha, df=len(x) - 2, interpretation=interp)

    @staticmethod
    def correlation_matrix_with_significance(df: pd.DataFrame, method: str = "pearson",
                                              alpha: float = ALPHA) -> tuple:
        """Returns (correlation matrix, p-value matrix) for every numeric column pair."""
        num = df.select_dtypes("number").dropna(axis=1, how="all")
        cols = num.columns
        r_mat = pd.DataFrame(np.eye(len(cols)), index=cols, columns=cols)
        p_mat = pd.DataFrame(np.zeros((len(cols), len(cols))), index=cols, columns=cols)
        for i, c1 in enumerate(cols):
            for j, c2 in enumerate(cols):
                if j <= i:
                    continue
                res = HypothesisTests.correlation_test(num[c1], num[c2], method=method, alpha=alpha)
                r_mat.loc[c1, c2] = r_mat.loc[c2, c1] = res.statistic if res.statistic is not None else np.nan
                p_mat.loc[c1, c2] = p_mat.loc[c2, c1] = res.p_value if res.p_value is not None else np.nan
        return r_mat, p_mat


# ══════════════════════════════════════════════════════════════════════════
#  5.  MODEL-COMPARISON REPORT  (cross-validated significance testing)
# ══════════════════════════════════════════════════════════════════════════
class ModelComparison:
    """
    Turns a dict of {model_name: array-of-per-fold CV scores} — all models
    MUST be scored on the *same* folds for a valid paired comparison — into
    a full statistical comparison report: descriptive stats + CI per model,
    an omnibus Friedman test, and Holm-corrected pairwise paired t-tests /
    Wilcoxon signed-rank tests with effect sizes.
    """

    def __init__(self, fold_scores: dict, alpha: float = ALPHA,
                 correction: str = MULTIPLE_COMPARISON_CORRECTION):
        self.scores = {k: np.asarray(v, dtype=float) for k, v in fold_scores.items()}
        self.alpha = alpha
        self.correction = correction

    def summary_table(self) -> pd.DataFrame:
        rows = []
        for name, arr in self.scores.items():
            ci = ConfidenceIntervals.t_interval(arr, alpha=self.alpha)
            rows.append({
                "Model": name, "N folds": len(arr),
                "Mean": round(float(np.mean(arr)), 4), "Std": round(float(np.std(arr, ddof=1)), 4),
                f"{int((1-self.alpha)*100)}% CI Lower": round(ci[0], 4),
                f"{int((1-self.alpha)*100)}% CI Upper": round(ci[1], 4),
                "Min": round(float(np.min(arr)), 4), "Max": round(float(np.max(arr)), 4),
            })
        return pd.DataFrame(rows).sort_values("Mean", ascending=False).reset_index(drop=True)

    def omnibus_test(self) -> TestResult:
        arrays = list(self.scores.values())
        if len(arrays) < 3:
            return HypothesisTests.paired_ttest(*arrays, alpha=self.alpha) if len(arrays) == 2 else \
                TestResult("Omnibus test", None, None, interpretation="need >=2 models")
        return HypothesisTests.friedman_test(*arrays, alpha=self.alpha)

    def pairwise_table(self) -> pd.DataFrame:
        names = list(self.scores.keys())
        pairs = list(combinations(names, 2))
        raw_p, rows = [], []
        for a, b in pairs:
            t_res = HypothesisTests.paired_ttest(self.scores[a], self.scores[b], alpha=self.alpha)
            w_res = HypothesisTests.wilcoxon_signed_rank(self.scores[a], self.scores[b], alpha=self.alpha)
            diff = float(np.mean(self.scores[a]) - np.mean(self.scores[b]))
            rows.append({"Model A": a, "Model B": b, "Mean Diff (A-B)": round(diff, 4),
                         "t-statistic": t_res.statistic, "t-test p-value": t_res.p_value,
                         "Wilcoxon p-value": w_res.p_value})
            raw_p.append(t_res.p_value if t_res.p_value is not None else 1.0)
        if pairs:
            _, p_adj, _, _ = multipletests(raw_p, alpha=self.alpha, method=self.correction)
            for row, pa in zip(rows, p_adj):
                row[f"p-value ({self.correction}-adjusted)"] = round(float(pa), 6)
                row["Significant?"] = "Yes" if pa < self.alpha else "No"
        return pd.DataFrame(rows)

    def best_model_set(self) -> list:
        """Models not statistically distinguishable from the single best-mean model
        (i.e. the adjusted pairwise test against the top performer is non-significant)."""
        summ = self.summary_table()
        if len(summ) == 0:
            return []
        best = summ.iloc[0]["Model"]
        pw = self.pairwise_table()
        col = [c for c in pw.columns if c.startswith("p-value (")]
        if not col:
            return [best]
        col = col[0]
        indistinguishable = {best}
        for _, row in pw.iterrows():
            if best in (row["Model A"], row["Model B"]) and row[col] >= self.alpha:
                indistinguishable.add(row["Model A"]); indistinguishable.add(row["Model B"])
        return sorted(indistinguishable)

    def full_report(self) -> dict:
        return {
            "summary": self.summary_table(),
            "omnibus": self.omnibus_test(),
            "pairwise": self.pairwise_table(),
            "best_model_set": self.best_model_set(),
        }


# ══════════════════════════════════════════════════════════════════════════
#  6.  GENERIC K-FOLD / REPEATED-K-FOLD / NESTED CROSS-VALIDATION
# ══════════════════════════════════════════════════════════════════════════
def kfold_cross_validate(model_factory: Callable, X: np.ndarray, y: np.ndarray,
                          k: int = CV_FOLDS, repeats: int = 1,
                          random_state: int = RANDOM_STATE,
                          n_features_for_adj_r2: Optional[int] = None) -> pd.DataFrame:
    """
    Model-agnostic (repeated) k-fold cross-validation. `model_factory()` must
    return a *fresh, unfitted* object exposing `.fit(X, y)` and `.predict(X)`
    — this works for the plain sklearn estimators as well as the project's
    own PINNModel / EnsembleModel / KrigingModel classes.

    Returns one row per fold with the full `comprehensive_metrics` bundle,
    ready to feed straight into `ModelComparison`.
    """
    splitter = (RepeatedKFold(n_splits=k, n_repeats=repeats, random_state=random_state)
                if repeats > 1 else KFold(n_splits=k, shuffle=True, random_state=random_state))
    rows = []
    for fold_i, (tr_idx, te_idx) in enumerate(splitter.split(X)):
        model = model_factory()
        model.fit(X[tr_idx], y[tr_idx])
        y_pred = model.predict(X[te_idx])
        m = comprehensive_metrics(y[te_idx], y_pred, n_features=n_features_for_adj_r2)
        m["fold"] = fold_i
        rows.append(m)
    return pd.DataFrame(rows)


def nested_cross_validate(model_factory: Callable, param_distributions: dict,
                           X: np.ndarray, y: np.ndarray,
                           outer_k: int = CV_FOLDS, inner_k: int = 3,
                           n_iter: int = 15, random_state: int = RANDOM_STATE,
                           scoring: str = "r2") -> dict:
    """
    Nested CV: an inner RandomizedSearchCV tunes hyperparameters on the
    training fold only, and the outer loop scores the tuned model on a held
    -out fold it never influenced — the textbook-correct way to report
    'accuracy after tuning' without leaking test information into the
    hyperparameter search (Cawley & Talbot, 2010, JMLR).
    """
    from sklearn.model_selection import RandomizedSearchCV
    outer = KFold(n_splits=outer_k, shuffle=True, random_state=random_state)
    fold_scores, best_params_per_fold = [], []
    for tr_idx, te_idx in outer.split(X):
        search = RandomizedSearchCV(model_factory(), param_distributions,
                                     n_iter=n_iter, cv=inner_k, scoring=scoring,
                                     random_state=random_state, n_jobs=-1)
        search.fit(X[tr_idx], y[tr_idx])
        y_pred = search.predict(X[te_idx])
        m = comprehensive_metrics(y[te_idx], y_pred)
        fold_scores.append(m)
        best_params_per_fold.append(search.best_params_)
    return {"fold_metrics": pd.DataFrame(fold_scores), "best_params_per_fold": best_params_per_fold}


def learning_curve_data(model_factory: Callable, X: np.ndarray, y: np.ndarray,
                         train_sizes=np.linspace(0.2, 1.0, 6), cv: int = CV_FOLDS,
                         random_state: int = RANDOM_STATE) -> pd.DataFrame:
    """Train/validation R2 vs. training-set size — the standard bias/variance
    (under-/over-fitting) diagnostic plot."""
    from sklearn.model_selection import learning_curve
    sizes, train_scores, val_scores = learning_curve(
        model_factory(), X, y, train_sizes=train_sizes, cv=cv,
        scoring="r2", random_state=random_state, shuffle=True)
    return pd.DataFrame({
        "train_size": sizes,
        "train_mean": train_scores.mean(axis=1), "train_std": train_scores.std(axis=1),
        "val_mean": val_scores.mean(axis=1), "val_std": val_scores.std(axis=1),
    })
