"""
modules/analysis.py — PerfoSim Pro v3
════════════════════════════════════════════════════════════════════════════
Performance metrics evaluation, benchmark comparison, and results
tabulation — now with the statistical rigor v2 was missing entirely:
descriptive tables carry confidence intervals and a normality verdict,
benchmark comparisons are backed by a real one-sample t-test (not just a
raw delta), and the model-comparison table reports cross-validated,
Holm-corrected statistical significance rather than a single R² per model.

WHAT CHANGED FROM v2
────────────────────────────────────────────────────────────────────────────
- `summary_table` previously imported `numpy`, `r2_score`,
  `mean_squared_error`, and `mean_absolute_percentage_error` and used NONE
  of them anywhere in the file — the whole module computed zero inferential
  statistics. It now reports a 95% CI and a Shapiro-Wilk normality verdict
  for every metric.
- `BenchmarkComparator.compare_scalar` only ever reported a raw
  study-minus-benchmark delta with an arbitrary "within 0.5" tolerance for
  "at par". `compare_distribution` (new) runs an actual one-sample t-test
  of the study's *distribution* of results against each published
  benchmark value and reports a p-value.
- `ResultsTabulator.model_comparison_table` reported a bare R² per model
  with no uncertainty and no test of whether one model actually
  outperforms another. It now takes the `ModelValidator` output directly
  and surfaces the Holm-corrected pairwise comparison from
  `stats_engine.ModelComparison`.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats as spstats

from config import ALPHA, BENCHMARKS, LIT_STUDIES
from modules.stats_engine import ConfidenceIntervals, GoodnessOfFit, HypothesisTests


# ═══════════════════════════════════════════════════════════════════════════
# 1.  PERFORMANCE METRICS  (descriptive statistics + inferential add-ons)
# ═══════════════════════════════════════════════════════════════════════════
class PerformanceMetrics:
    """Compute all KPI metrics from a simulated-results DataFrame."""

    METRIC_DEFS = {
        "PI_stb_day_psi":    ("Productivity Index",       "STB/day/psi",  "max"),
        "S_total":           ("Total Skin Factor",         "–",            "min"),
        "EUR_MMbbl":         ("Estimated Ultimate Recov.", "MMbbl",        "max"),
        "NPV_MM":            ("Net Present Value",         "M$",           "max"),
        "FDI":               ("Fluid Divergence Index",    "–",            "min"),
        "sand_risk":         ("Sand Production Risk",      "ratio",        "min"),
        "recovery_factor":   ("Recovery Factor",           "fraction",     "max"),
        "completion_cost_MM":("Completion Cost",           "M$",           "min"),
        "damage_index":      ("Damage Index",              "–",            "min"),
    }

    def summary_table(self, df: pd.DataFrame, alpha: float = ALPHA) -> pd.DataFrame:
        """Descriptive statistics + 95% CI + a normality verdict per metric —
        v2 reported only min/median/mean/max/std with no measure of
        estimation uncertainty and no distributional diagnostic at all."""
        rows = []
        for col, (name, unit, direction) in self.METRIC_DEFS.items():
            if col not in df.columns:
                continue
            v = df[col].dropna()
            if len(v) < 2:
                continue
            ci = ConfidenceIntervals.t_interval(v, alpha=alpha)
            sw = GoodnessOfFit.shapiro_wilk(v)
            rows.append({
                "Metric":    name, "Symbol": col, "Unit": unit,
                "N":         int(len(v)),
                "Min":       round(float(v.min()), 4),
                "P25":       round(float(v.quantile(0.25)), 4),
                "Median":    round(float(v.median()), 4),
                "Mean":      round(float(v.mean()), 4),
                "P75":       round(float(v.quantile(0.75)), 4),
                "Max":       round(float(v.max()), 4),
                "Std":       round(float(v.std()), 4),
                f"{int((1-alpha)*100)}% CI": f"[{ci[0]:.4g}, {ci[1]:.4g}]",
                "Skewness":  round(float(spstats.skew(v)), 3),
                "Normality (Shapiro p)": "N/A" if sw.p_value is None else round(sw.p_value, 4),
                "Optimal":   "↑ Max" if direction == "max" else "↓ Min",
            })
        return pd.DataFrame(rows)

    def ml_metrics_table(self, ml_results: dict, target: str) -> pd.DataFrame:
        """Accepts the dict returned by `ml_engine.ModelValidator.run()`."""
        from modules.ml_engine import MODEL_LABELS
        rows = []
        for key, label in MODEL_LABELS.items():
            if key not in ml_results:
                continue
            m = ml_results[key]["metrics"]
            ci = ml_results[key].get("r2_ci", {})
            row = {
                "Model": label, "Target": target,
                "R2": round(m.get("R2", np.nan), 4),
                "R2 95% CI": f"[{ci.get('lower', float('nan')):.3f}, {ci.get('upper', float('nan')):.3f}]" if ci else "—",
                "Adj. R2": None if m.get("Adjusted_R2") is None else round(m["Adjusted_R2"], 4),
                "RMSE": round(m.get("RMSE", np.nan), 4),
                "MAE": round(m.get("MAE", np.nan), 4),
                "MAPE (%)": round(m.get("MAPE_%", np.nan), 2),
                "NSE": round(m.get("NSE", np.nan), 4),
                "Willmott d": round(m.get("Willmott_d", np.nan), 4),
                "Accuracy@10%": round(m.get("Accuracy@10%", np.nan), 3),
                "Accuracy@20%": round(m.get("Accuracy@20%", np.nan), 3),
            }
            if "cv_table" in ml_results[key]:
                row["CV R2 (mean±std)"] = (f"{ml_results[key]['cv_table']['R2'].mean():.4f} "
                                            f"± {ml_results[key]['cv_table']['R2'].std():.4f}")
            rows.append(row)
        return pd.DataFrame(rows)

    def optimization_table(self, pareto_df: pd.DataFrame) -> pd.DataFrame:
        if pareto_df is None or len(pareto_df) == 0:
            return pd.DataFrame()
        stat_cols = [c for c in pareto_df.columns if "(" in c or c in
                     ["SPF", "Pen_in", "Phasing", "CS_ft", "N_clust"]]
        rows = []
        for col in stat_cols:
            if col not in pareto_df.columns:
                continue
            v = pareto_df[col].dropna()
            rows.append({"Variable/Objective": col,
                "Min": round(float(v.min()), 3),
                "Mean": round(float(v.mean()), 3),
                "Max": round(float(v.max()), 3),
                "Std": round(float(v.std()), 3)})
        return pd.DataFrame(rows)

    def compute_improvement(self, before: float, after: float,
                             higher_is_better=True) -> dict:
        if before == 0:
            return {"absolute": after, "pct": 0.0, "improved": False}
        diff = after - before
        pct  = diff / abs(before) * 100
        improved = (diff > 0) if higher_is_better else (diff < 0)
        return {"absolute": round(diff, 4),
                "pct": round(pct, 2),
                "improved": bool(improved)}


# ═══════════════════════════════════════════════════════════════════════════
# 2.  BENCHMARK COMPARATOR  (now backed by real hypothesis tests)
# ═══════════════════════════════════════════════════════════════════════════
class BenchmarkComparator:
    """Compare study results against published benchmarks and literature."""

    OPTIMAL_DIRECTION = {"PI_stb_day_psi": "max", "EUR_MMbbl": "max", "NPV_MM": "max",
                         "recovery_factor": "max", "ML_R2": "max",
                         "skin": "min", "FDI": "min"}

    def compare_scalar(self, study_val: float, metric_key: str) -> pd.DataFrame:
        """Point-estimate comparison (kept for simple scalar contexts, e.g.
        a single well design). For a full distribution of results, prefer
        `compare_distribution`, which is statistically grounded."""
        if metric_key not in BENCHMARKS:
            return pd.DataFrame()
        bm = BENCHMARKS[metric_key]
        rows = []
        for k, v in bm.items():
            if k == "source":
                continue
            rows.append({"Benchmark Category": k.replace("_", " ").title(),
                         "Value": v, "Source": bm.get("source", "—")})
        df = pd.DataFrame(rows)
        df["This Study"] = study_val
        df["Δ (Study – Benchmark)"] = (study_val - df["Value"]).round(3)
        opt = self.OPTIMAL_DIRECTION.get(metric_key, "max")

        def status(row):
            delta = row["Δ (Study – Benchmark)"]
            if opt == "max":
                return "✅ Better" if delta > 0 else ("⚠️ At par" if abs(delta) < 0.5 else "❌ Below")
            return "✅ Better" if delta < 0 else ("⚠️ At par" if abs(delta) < 0.5 else "❌ Below")
        df["Status"] = df.apply(status, axis=1)
        return df

    def compare_distribution(self, study_series: pd.Series, metric_key: str,
                              alpha: float = ALPHA) -> pd.DataFrame:
        """
        Statistically rigorous version: for EACH published benchmark value,
        runs a one-sample t-test of H0: mean(study) = benchmark_value, and
        reports the t-statistic, p-value, and whether the study's mean is
        significantly different from (and in which direction relative to)
        the published figure — not just whether the point estimate happens
        to be numerically larger.
        """
        if metric_key not in BENCHMARKS:
            return pd.DataFrame()
        bm = BENCHMARKS[metric_key]
        s = study_series.dropna()
        opt = self.OPTIMAL_DIRECTION.get(metric_key, "max")
        rows = []
        for k, v in bm.items():
            if k == "source":
                continue
            test = HypothesisTests.one_sample_ttest(s, popmean=float(v), alpha=alpha)
            mean_diff = float(s.mean() - v)
            if not test.significant:
                verdict = "No significant difference"
            elif (mean_diff > 0) == (opt == "max"):
                verdict = "Significantly BETTER than benchmark"
            else:
                verdict = "Significantly BELOW benchmark"
            rows.append({
                "Benchmark Category": k.replace("_", " ").title(), "Benchmark Value": v,
                "Study Mean": round(float(s.mean()), 4), "Study N": len(s),
                "Mean Diff": round(mean_diff, 4),
                "t-statistic": None if test.statistic is None else round(test.statistic, 3),
                "p-value": None if test.p_value is None else round(test.p_value, 6),
                "Significant?": "Yes" if test.significant else "No",
                "Verdict": verdict, "Source": bm.get("source", "—"),
            })
        return pd.DataFrame(rows)

    def literature_table(self) -> pd.DataFrame:
        return pd.DataFrame(LIT_STUDIES).rename(columns={
            "study": "Study / Authors", "journal": "Reference",
            "PI_improvement_pct": "PI Improvement (%)",
            "skin_reduction_pct": "Skin Reduction (%)",
        })

    def skin_improvement_summary(self, df: pd.DataFrame) -> pd.DataFrame:
        """Skin improvement vs. baseline for different phasing/SPF combos."""
        rows = []
        for ph in sorted(df["phasing"].unique()):
            for spf in sorted(df["spf"].unique()):
                sub = df[(df["phasing"] == ph) & (df["spf"] == spf)]
                if len(sub) < 3:
                    continue
                rows.append({"Phasing (°)": int(ph), "SPF": int(spf),
                              "Mean S_total": round(sub["S_total"].mean(), 2),
                              "Mean PI": round(sub["PI_stb_day_psi"].mean(), 3),
                              "N": len(sub)})
        return pd.DataFrame(rows).sort_values("Mean S_total")

    def percentile_rank(self, study_val: float, benchmark_dict: dict) -> str:
        vals = sorted(v for k, v in benchmark_dict.items() if k != "source")
        n = len(vals)
        if n == 0:
            return "N/A"
        rank = sum(study_val > v for v in vals)
        pct = int(rank / n * 100)
        tier = ("Top Quartile" if pct >= 75 else "Above Median" if pct >= 50
                else "Below Median" if pct >= 25 else "Bottom Quartile")
        return f"P{pct} — {tier}"


# ═══════════════════════════════════════════════════════════════════════════
# 3.  RESULTS TABULATOR  (formatted output tables, now with significance)
# ═══════════════════════════════════════════════════════════════════════════
class ResultsTabulator:

    def best_designs_table(self, df: pd.DataFrame, n_top=10) -> pd.DataFrame:
        """Top-N designs ranked by composite score."""
        if "PI_stb_day_psi" not in df.columns:
            return df.head(n_top)
        score = (
            (df["PI_stb_day_psi"] / df["PI_stb_day_psi"].max()) * 0.30 +
            (df["EUR_MMbbl"] / df["EUR_MMbbl"].max()) * 0.25 +
            (df["NPV_MM"] / df["NPV_MM"].max()) * 0.25 +
            (1 - df["S_total"].clip(0) / df["S_total"].clip(0).max().clip(1e-9)) * 0.10 +
            (1 - df["FDI"] / df["FDI"].max()) * 0.05 +
            (1 - df["sand_risk"] / df["sand_risk"].max()) * 0.05
        )
        df2 = df.copy()
        df2["Composite Score"] = score.round(4)
        display_cols = ["spf", "phasing", "pen_depth_in", "permeability_mD", "net_pay_ft",
                        "PI_stb_day_psi", "S_total", "EUR_MMbbl", "NPV_MM",
                        "FDI", "sand_risk", "Composite Score"]
        available = [c for c in display_cols if c in df2.columns]
        return df2.nlargest(n_top, "Composite Score")[available].reset_index(drop=True)

    def sensitivity_summary(self, sobol_df: pd.DataFrame) -> pd.DataFrame:
        df = sobol_df.copy()
        df["Interaction (ST-S1)"] = (df["ST"] - df["S1"]).round(4)
        df["Rank"] = range(1, len(df) + 1)
        df["Importance"] = df["ST"].apply(lambda v:
            "Critical (>0.3)" if v > 0.3 else
            ("High (0.1–0.3)" if v > 0.1 else
             ("Moderate (0.05–0.1)" if v > 0.05 else "Low (<0.05)")))
        return df.round(4)

    def model_comparison_table(self, ml_results: dict, target: str) -> pd.DataFrame:
        pm = PerformanceMetrics()
        tbl = pm.ml_metrics_table(ml_results, target)
        tbl["Benchmark R2"] = BENCHMARKS["ML_R2"]["target"]
        tbl["Meets Target?"] = tbl["R2"].apply(
            lambda v: "✅ Yes" if v >= BENCHMARKS["ML_R2"]["target"] else "❌ No")
        return tbl

    def statistical_significance_table(self, ml_results: dict) -> dict:
        """Surfaces `stats_engine.ModelComparison`'s full report (already
        computed by `ModelValidator.run`) as ready-to-display tables. This
        is the direct, cross-validated answer to "which model is actually
        better, statistically" — v2 had no such comparison at all."""
        if "comparison" not in ml_results:
            return {"summary": pd.DataFrame(), "pairwise": pd.DataFrame(),
                    "omnibus": None, "best_model_set": []}
        return ml_results["comparison"]

    def optimization_summary(self, pareto_df: pd.DataFrame,
                              best_balanced: pd.Series = None) -> pd.DataFrame:
        if pareto_df is None or len(pareto_df) == 0:
            return pd.DataFrame()
        obj_cols = [c for c in pareto_df.columns if "(" in c]
        summary_data = []
        for col in obj_cols:
            v = pareto_df[col]
            row = {"Objective": col,
                   "Pareto Min": round(v.min(), 3),
                   "Pareto Mean": round(v.mean(), 3),
                   "Pareto Max": round(v.max(), 3)}
            if best_balanced is not None and col in best_balanced.index:
                row["Balanced Solution"] = round(best_balanced[col], 3)
            summary_data.append(row)
        return pd.DataFrame(summary_data)

    def full_results_export(self, df: pd.DataFrame) -> pd.DataFrame:
        """Clean DataFrame ready for Excel/CSV export."""
        export_cols = {
            "spf": "SPF (shots/ft)", "phasing": "Phasing (°)",
            "pen_depth_in": "Penetration Depth (in)", "hole_dia_in": "Hole Diameter (in)",
            "underbalance_psi": "Underbalance (psi)", "permeability_mD": "Permeability (mD)",
            "net_pay_ft": "Net Pay (ft)", "initial_pressure_psi": "Initial Pressure (psi)",
            "PI_stb_day_psi": "PI (STB/day/psi)", "S_total": "Skin Factor (Total)",
            "S_KT": "Skin (K&T Perf.)", "S_damage": "Skin (Damage)",
            "EUR_MMbbl": "EUR (MMbbl)", "NPV_MM": "NPV (M$)", "ROI": "ROI",
            "payback_yr": "Payback (yr)", "FDI": "Fluid Divergence Index",
            "sand_risk": "Sand Risk Factor", "damage_index": "Damage Index",
            "recovery_factor": "Recovery Factor",
        }
        available = {k: v for k, v in export_cols.items() if k in df.columns}
        return df[list(available.keys())].rename(columns=available).round(4)
