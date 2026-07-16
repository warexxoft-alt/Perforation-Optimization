"""
modules/pub_plots.py — PerfoSim Pro v3
════════════════════════════════════════════════════════════════════════════
Publication-grade, camera-ready static figures (matplotlib + seaborn),
saved at 600 DPI with a consistent thesis/journal style: serif font
(Liberation Serif — metrically compatible with Times New Roman), white
backgrounds, no gridlines, ≥12pt type, ≥1.2pt line weights, and numbered
titles. This is the module the interactive Plotly GUI (`modules/plots.py`)
hands off to whenever the user wants a "print quality" / downloadable
version of a chart, and it is also where the statistical-diagnostic plots
that v2 had no equivalent of at all (residual diagnostics, CV comparison,
learning curves, GOF Q-Q plots) live.

v2 had zero matplotlib output — every chart was an interactive Plotly
figure with no high-resolution static-export path. Plotly figures CAN be
exported via kaleido, but at on-screen resolution; this module exists
specifically to satisfy "camera-ready, 600 DPI" figures for theses/papers.
"""

from __future__ import annotations

import io

import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy import stats as spstats

from config import COLORS, PUB_DPI, PUB_FONT_SERIF, PUB_FONT_SIZE, PUB_LINEWIDTH, PUB_TITLE_SIZE

_STYLE_SET = False


def set_pub_style():
    """Idempotent — configures matplotlib once per process for consistent,
    thesis/journal-ready output across every figure in this module."""
    global _STYLE_SET
    plt.rcParams.update({
        "font.family": "serif",
        "font.serif": PUB_FONT_SERIF,
        "font.size": PUB_FONT_SIZE,
        "axes.titlesize": PUB_TITLE_SIZE,
        "axes.labelsize": PUB_FONT_SIZE,
        "xtick.labelsize": PUB_FONT_SIZE - 1,
        "ytick.labelsize": PUB_FONT_SIZE - 1,
        "legend.fontsize": PUB_FONT_SIZE - 1,
        "axes.linewidth": PUB_LINEWIDTH,
        "lines.linewidth": PUB_LINEWIDTH,
        "axes.grid": False,
        "axes.facecolor": "white",
        "figure.facecolor": "white",
        "savefig.facecolor": "white",
        "savefig.dpi": PUB_DPI,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.axisbelow": True,
    })
    _STYLE_SET = True


def _finish(fig, title=None, fig_no=None):
    if title:
        full = f"Figure {fig_no}. {title}" if fig_no is not None else title
        fig.suptitle(full, fontsize=PUB_TITLE_SIZE, y=1.02, fontweight="bold")
    fig.tight_layout()
    return fig


def save_fig_bytes(fig, dpi: int = PUB_DPI) -> bytes:
    """Render a figure to 600 DPI PNG bytes, ready for st.download_button."""
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=dpi, bbox_inches="tight", facecolor="white")
    buf.seek(0)
    return buf.getvalue()


PALETTE = [COLORS["primary"], COLORS["accent"], COLORS["secondary"],
           COLORS["good"], COLORS["danger"], COLORS["warning"]]


# ═══════════════════════════════════════════════════════════════════════════
# 1.  PREDICTED VS ACTUAL
# ═══════════════════════════════════════════════════════════════════════════
def fig_predicted_vs_actual(y_true, y_pred, model_name="Model", target_label="Target",
                             r2=None, rmse=None, p_value=None, fig_no=None):
    set_pub_style()
    y_true, y_pred = np.asarray(y_true), np.asarray(y_pred)
    fig, ax = plt.subplots(figsize=(6.0, 5.4))
    ax.scatter(y_true, y_pred, s=26, color=COLORS["primary"], alpha=0.65,
               edgecolor="white", linewidth=0.4, zorder=3)
    lo, hi = min(y_true.min(), y_pred.min()), max(y_true.max(), y_pred.max())
    pad = 0.04 * (hi - lo + 1e-9)
    ax.plot([lo - pad, hi + pad], [lo - pad, hi + pad], "--", color="black",
            linewidth=PUB_LINEWIDTH, label="1:1 line", zorder=2)
    slope, intercept, r, p, se = spstats.linregress(y_true, y_pred)
    xs = np.linspace(lo - pad, hi + pad, 50)
    ax.plot(xs, slope * xs + intercept, color=COLORS["accent"],
            linewidth=PUB_LINEWIDTH, label=f"Fit (r={r:.3f})", zorder=2)
    ax.set_xlim(lo - pad, hi + pad); ax.set_ylim(lo - pad, hi + pad)
    ax.set_xlabel(f"Actual {target_label}"); ax.set_ylabel(f"Predicted {target_label}")
    txt = []
    if r2 is not None: txt.append(f"$R^2$ = {r2:.3f}")
    if rmse is not None: txt.append(f"RMSE = {rmse:.4g}")
    if p_value is not None: txt.append(f"corr. p = {p_value:.2g}")
    if txt:
        ax.text(0.04, 0.96, "\n".join(txt), transform=ax.transAxes, va="top", ha="left",
                fontsize=PUB_FONT_SIZE - 1, bbox=dict(boxstyle="round", fc="white",
                ec=COLORS["muted"], alpha=0.9))
    ax.legend(loc="lower right", frameon=False)
    return _finish(fig, f"{model_name}: Predicted vs. Actual {target_label}", fig_no)


# ═══════════════════════════════════════════════════════════════════════════
# 2.  RESIDUAL DIAGNOSTICS  (4-panel — the classic regression GOF bundle)
# ═══════════════════════════════════════════════════════════════════════════
def fig_residual_diagnostics(y_true, y_pred, model_name="Model", fig_no=None):
    set_pub_style()
    y_true, y_pred = np.asarray(y_true), np.asarray(y_pred)
    resid = y_true - y_pred
    std_resid = (resid - resid.mean()) / (resid.std(ddof=1) + 1e-12)

    fig, axes = plt.subplots(2, 2, figsize=(9.6, 8.0))

    ax = axes[0, 0]
    ax.scatter(y_pred, resid, s=20, color=COLORS["primary"], alpha=0.6, edgecolor="white", linewidth=0.3)
    ax.axhline(0, color="black", linewidth=PUB_LINEWIDTH, linestyle="--")
    ax.set_xlabel("Fitted value"); ax.set_ylabel("Residual"); ax.set_title("(a) Residuals vs. Fitted")

    ax = axes[0, 1]
    spstats.probplot(std_resid, dist="norm", plot=ax)
    ax.get_lines()[0].set_markerfacecolor(COLORS["primary"])
    ax.get_lines()[0].set_markeredgecolor("white")
    ax.get_lines()[0].set_markersize(5)
    ax.get_lines()[1].set_color("black")
    ax.get_lines()[1].set_linewidth(PUB_LINEWIDTH)
    ax.set_title("(b) Normal Q-Q"); ax.set_ylabel("Ordered residuals (standardised)")

    ax = axes[1, 0]
    ax.hist(resid, bins=min(30, max(8, len(resid)//8)), color=COLORS["primary"],
            alpha=0.75, edgecolor="white", density=True)
    xs = np.linspace(resid.min(), resid.max(), 200)
    ax.plot(xs, spstats.norm.pdf(xs, resid.mean(), resid.std(ddof=1)), color=COLORS["accent"],
            linewidth=PUB_LINEWIDTH + 0.3, label="Normal fit")
    ax.set_xlabel("Residual"); ax.set_ylabel("Density"); ax.set_title("(c) Residual Distribution")
    ax.legend(frameon=False)

    ax = axes[1, 1]
    ax.scatter(y_pred, np.sqrt(np.abs(std_resid)), s=20, color=COLORS["primary"],
               alpha=0.6, edgecolor="white", linewidth=0.3)
    if len(y_pred) > 2:
        order = np.argsort(y_pred)
        z = np.polyfit(y_pred[order], np.sqrt(np.abs(std_resid))[order], 1)
        ax.plot(y_pred[order], np.polyval(z, y_pred[order]), color=COLORS["accent"], linewidth=PUB_LINEWIDTH)
    ax.set_xlabel("Fitted value"); ax.set_ylabel(r"$\sqrt{|Standardised\ residual|}$")
    ax.set_title("(d) Scale-Location")

    return _finish(fig, f"{model_name}: Residual Diagnostics", fig_no)


# ═══════════════════════════════════════════════════════════════════════════
# 3.  CROSS-VALIDATION MODEL COMPARISON (boxplot + significance brackets)
# ═══════════════════════════════════════════════════════════════════════════
def fig_cv_comparison(scores_dict: dict, metric_name="R2", pairwise_df=None, fig_no=None):
    set_pub_style()
    names = list(scores_dict.keys())
    data = [np.asarray(scores_dict[n]) for n in names]
    fig, ax = plt.subplots(figsize=(max(6.0, 1.3 * len(names) + 2), 5.4))
    bp = ax.boxplot(data, labels=names, patch_artist=True, widths=0.55,
                     medianprops=dict(color="black", linewidth=PUB_LINEWIDTH),
                     boxprops=dict(linewidth=PUB_LINEWIDTH), whiskerprops=dict(linewidth=PUB_LINEWIDTH),
                     capprops=dict(linewidth=PUB_LINEWIDTH))
    for patch, c in zip(bp["boxes"], PALETTE * 3):
        patch.set_facecolor(c); patch.set_alpha(0.55)
    for i, d in enumerate(data, start=1):
        jitter = np.random.default_rng(0).normal(0, 0.04, len(d))
        ax.scatter(np.full(len(d), i) + jitter, d, s=16, color="black", alpha=0.5, zorder=5)

    if pairwise_df is not None and len(pairwise_df) > 0:
        sig_col = [c for c in pairwise_df.columns if c.startswith("p-value (")]
        sig_col = sig_col[0] if sig_col else None
        if sig_col:
            ymax = max(d.max() for d in data)
            step = 0.07 * (ymax - min(d.min() for d in data) + 1e-9)
            level = 1
            for _, row in pairwise_df.iterrows():
                if row[sig_col] < 0.05:
                    ia, ib = names.index(row["Model A"]) + 1, names.index(row["Model B"]) + 1
                    y = ymax + step * level
                    ax.plot([ia, ia, ib, ib], [y, y + step*0.15, y + step*0.15, y], color="black", linewidth=1.0)
                    ax.text((ia+ib)/2, y + step*0.18, "*", ha="center", va="bottom", fontsize=14)
                    level += 1

    ax.set_ylabel(f"Cross-validated {metric_name}")
    ax.set_xlabel("Model")
    plt.setp(ax.get_xticklabels(), rotation=15, ha="right")
    return _finish(fig, f"Cross-Validated {metric_name} by Model (* = significant pairwise difference, p<0.05)", fig_no)


# ═══════════════════════════════════════════════════════════════════════════
# 4.  LEARNING CURVE
# ═══════════════════════════════════════════════════════════════════════════
def fig_learning_curve(lc_df: pd.DataFrame, model_name="Model", fig_no=None):
    set_pub_style()
    fig, ax = plt.subplots(figsize=(6.4, 5.0))
    ax.plot(lc_df["train_size"], lc_df["train_mean"], "-o", color=COLORS["primary"],
            label="Training score", markersize=4)
    ax.fill_between(lc_df["train_size"], lc_df["train_mean"]-lc_df["train_std"],
                     lc_df["train_mean"]+lc_df["train_std"], color=COLORS["primary"], alpha=0.15)
    ax.plot(lc_df["train_size"], lc_df["val_mean"], "-s", color=COLORS["accent"],
            label="Validation score", markersize=4)
    ax.fill_between(lc_df["train_size"], lc_df["val_mean"]-lc_df["val_std"],
                     lc_df["val_mean"]+lc_df["val_std"], color=COLORS["accent"], alpha=0.15)
    ax.set_xlabel("Training set size"); ax.set_ylabel("$R^2$ score")
    ax.legend(frameon=False, loc="lower right")
    return _finish(fig, f"{model_name}: Learning Curve", fig_no)


# ═══════════════════════════════════════════════════════════════════════════
# 5.  FEATURE IMPORTANCE
# ═══════════════════════════════════════════════════════════════════════════
def fig_feature_importance(importance, names=None, errors=None, title="Feature Importance", fig_no=None):
    set_pub_style()
    if isinstance(importance, pd.Series):
        names, vals = list(importance.index), importance.values
    elif isinstance(importance, pd.DataFrame):
        col = [c for c in importance.columns if "importance" in c.lower()][0]
        names, vals = list(importance.iloc[:, 0]), importance[col].values
        if "importance_std" in importance.columns:
            errors = importance["importance_std"].values
    else:
        vals = np.asarray(importance)
    order = np.argsort(vals)
    fig, ax = plt.subplots(figsize=(6.6, max(3.2, 0.42 * len(vals) + 1)))
    yerr = np.asarray(errors)[order] if errors is not None else None
    ax.barh(np.array(names)[order], np.array(vals)[order], xerr=yerr,
            color=COLORS["primary"], edgecolor="white", height=0.65,
            error_kw=dict(ecolor="black", capsize=3, linewidth=1.0))
    ax.set_xlabel("Relative importance")
    return _finish(fig, title, fig_no)


# ═══════════════════════════════════════════════════════════════════════════
# 6.  SOBOL SENSITIVITY TORNADO
# ═══════════════════════════════════════════════════════════════════════════
def fig_sobol_tornado(sobol_df: pd.DataFrame, fig_no=None):
    set_pub_style()
    df = sobol_df.sort_values("ST")
    fig, ax = plt.subplots(figsize=(6.8, max(3.2, 0.5 * len(df) + 1)))
    y = np.arange(len(df))
    ax.barh(y - 0.18, df["S1"], height=0.32, color=COLORS["primary"], label="First-order $S_1$")
    ax.barh(y + 0.18, df["ST"], height=0.32, color=COLORS["accent"], label="Total-order $S_T$")
    ax.set_yticks(y); ax.set_yticklabels(df["feature"])
    ax.set_xlabel("Sobol sensitivity index")
    ax.legend(frameon=False, loc="lower right")
    return _finish(fig, "Sobol Variance-Based Sensitivity Analysis", fig_no)


# ═══════════════════════════════════════════════════════════════════════════
# 7.  CORRELATION HEATMAP WITH SIGNIFICANCE
# ═══════════════════════════════════════════════════════════════════════════
def fig_correlation_heatmap(corr: pd.DataFrame, pvals: pd.DataFrame = None, fig_no=None):
    set_pub_style()
    fig, ax = plt.subplots(figsize=(max(6, 0.55*len(corr)+2), max(5, 0.55*len(corr)+1.5)))
    im = ax.imshow(corr.values, cmap="RdBu_r", vmin=-1, vmax=1, aspect="auto")
    ax.set_xticks(range(len(corr.columns))); ax.set_xticklabels(corr.columns, rotation=45, ha="right")
    ax.set_yticks(range(len(corr.index))); ax.set_yticklabels(corr.index)
    for i in range(len(corr)):
        for j in range(len(corr.columns)):
            val = corr.values[i, j]
            star = ""
            if pvals is not None and i != j:
                p = pvals.values[i, j]
                star = "*" if p < 0.05 else ""
            txt_color = "white" if abs(val) > 0.55 else "black"
            ax.text(j, i, f"{val:.2f}{star}", ha="center", va="center",
                    fontsize=PUB_FONT_SIZE - 2, color=txt_color)
    cbar = fig.colorbar(im, ax=ax, shrink=0.85)
    cbar.set_label("Pearson r")
    return _finish(fig, "Feature Correlation Matrix (* p<0.05)", fig_no)


# ═══════════════════════════════════════════════════════════════════════════
# 8.  PARETO FRONT (2D projection with color-coded 3rd objective)
# ═══════════════════════════════════════════════════════════════════════════
def fig_pareto_2d(df: pd.DataFrame, x_col: str, y_col: str, color_col: str = None, fig_no=None):
    set_pub_style()
    fig, ax = plt.subplots(figsize=(6.6, 5.4))
    if color_col and color_col in df.columns:
        sca = ax.scatter(df[x_col], df[y_col], c=df[color_col], cmap="viridis",
                          s=42, edgecolor="white", linewidth=0.4)
        cbar = fig.colorbar(sca, ax=ax); cbar.set_label(color_col)
    else:
        ax.scatter(df[x_col], df[y_col], color=COLORS["primary"], s=42, edgecolor="white", linewidth=0.4)
    if "Feasible?" in df.columns:
        infeas = df[df["Feasible?"] == "No"]
        if len(infeas):
            ax.scatter(infeas[x_col], infeas[y_col], facecolor="none", edgecolor=COLORS["danger"],
                       s=70, linewidth=1.2, label="Infeasible")
            ax.legend(frameon=False)
    ax.set_xlabel(x_col); ax.set_ylabel(y_col)
    return _finish(fig, f"NSGA-II Pareto Front: {y_col} vs. {x_col}", fig_no)


# ═══════════════════════════════════════════════════════════════════════════
# 9.  PINN TRAINING HISTORY (now meaningful — v2's was fabricated)
# ═══════════════════════════════════════════════════════════════════════════
def fig_training_history(history: dict, fig_no=None):
    set_pub_style()
    fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.6))
    ep = history["epoch"]
    axes[0].plot(ep, history["loss_data"], color=COLORS["primary"], label="Data loss")
    axes[0].plot(ep, history["loss_phys"], color=COLORS["accent"], label="Physics loss")
    axes[0].plot(ep, history["loss_total"], color="black", linestyle="--", label="Total loss", linewidth=1.2)
    axes[0].set_yscale("log"); axes[0].set_xlabel("Epoch"); axes[0].set_ylabel("Loss (log scale)")
    axes[0].set_title("(a) Training Loss"); axes[0].legend(frameon=False)

    axes[1].plot(ep, history["val_r2"], color=COLORS["good"])
    axes[1].set_xlabel("Epoch"); axes[1].set_ylabel("Validation $R^2$")
    axes[1].set_title("(b) Validation Performance")

    return _finish(fig, "PINN Training Convergence (Adam + analytic backprop)", fig_no)


# ═══════════════════════════════════════════════════════════════════════════
# 10.  MODEL COMPARISON SUMMARY (grouped bar with CI whiskers)
# ═══════════════════════════════════════════════════════════════════════════
def fig_model_metric_bars(summary_df: pd.DataFrame, fig_no=None):
    set_pub_style()
    fig, ax = plt.subplots(figsize=(max(6, 1.1*len(summary_df)+2), 5.0))
    x = np.arange(len(summary_df))
    lo_col = [c for c in summary_df.columns if "CI Lower" in c][0]
    hi_col = [c for c in summary_df.columns if "CI Upper" in c][0]
    yerr = np.vstack([summary_df["Mean"] - summary_df[lo_col], summary_df[hi_col] - summary_df["Mean"]])
    ax.bar(x, summary_df["Mean"], yerr=yerr, color=PALETTE[:len(summary_df)],
           edgecolor="white", capsize=5, error_kw=dict(linewidth=1.2, ecolor="black"))
    ax.set_xticks(x); ax.set_xticklabels(summary_df["Model"], rotation=10, ha="right")
    ax.set_ylabel("Mean CV score (95% CI)")
    return _finish(fig, "Cross-Validated Model Performance with 95% Confidence Intervals", fig_no)
