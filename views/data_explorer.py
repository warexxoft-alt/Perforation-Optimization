"""views/data_explorer.py — Filterable table view + rigorous descriptive statistics."""
import pandas as pd
import streamlit as st

from modules import pub_plots
from modules.analysis import PerformanceMetrics
from modules.stats_engine import HypothesisTests
from views.common import caption, info_card, kpi_row, register_figure, section_header


def render(ctx=None):
    ss = st.session_state
    st.title("🔍 Data Explorer")
    caption("Filter the working dataset and inspect rigorous descriptive statistics "
            "(95% CI, skewness, Shapiro-Wilk normality) for every KPI.")
    st.divider()

    if ss.get("reservoir_df") is None:
        info_card("No dataset yet. Head to <b>📂 Data Hub</b> first.")
        return
    df = ss["reservoir_df"]

    section_header("Filters", "🧰", help="Narrow the dataset to a range of one column before viewing statistics or running a comparison.")
    numeric_cols = df.select_dtypes("number").columns.tolist()
    filt_col = st.selectbox("Filter by column (optional)", ["— none —"] + numeric_cols,
                            help="Restrict every table/chart below to a range of this column.")
    df_f = df
    if filt_col != "— none —":
        lo, hi = float(df[filt_col].min()), float(df[filt_col].max())
        if lo < hi:
            rng = st.slider(f"{filt_col} range", lo, hi, (lo, hi))
            df_f = df[(df[filt_col] >= rng[0]) & (df[filt_col] <= rng[1])]

    kpi_row([("Rows (filtered)", f"{len(df_f):,}"), ("Rows (total)", f"{len(df):,}"),
            ("Columns", f"{df.shape[1]}")])
    st.dataframe(df_f, width='stretch', height=320)

    st.divider()
    section_header("Descriptive Statistics with Inferential Add-ons", "📐", help="Same core stats as a textbook summary table, plus a 95% CI and a normality test most spreadsheets skip.")
    pm = PerformanceMetrics()
    summ = pm.summary_table(df_f)
    st.dataframe(summ, width='stretch', hide_index=True)
    caption("95% CI is the parametric t-interval for the mean. Shapiro-Wilk p<0.05 means the "
            "distribution significantly deviates from normal — common for skewed EUR/NPV metrics.")

    st.divider()
    section_header("Two-Sample Comparison", "⚖️", help="Splits the (filtered) dataset in two by a threshold and tests whether a metric genuinely differs between the groups.")
    st.caption("Split the dataset by a categorical/binary condition and test whether a metric "
               "differs significantly between the two groups (independent t-test + Wilcoxon).")
    c1, c2, c3 = st.columns(3)
    split_col = c1.selectbox("Split by", numeric_cols, index=numeric_cols.index("spf") if "spf" in numeric_cols else 0)
    threshold = c2.number_input("Threshold", value=float(df_f[split_col].median()))
    metric_col = c3.selectbox("Compare metric", [c for c in numeric_cols if c != split_col])
    if st.button("Run comparison"):
        group_a = df_f[df_f[split_col] <= threshold][metric_col].dropna()
        group_b = df_f[df_f[split_col] > threshold][metric_col].dropna()
        if len(group_a) > 1 and len(group_b) > 1:
            test = HypothesisTests.independent_ttest(group_a, group_b)
            kpi_row([(f"{split_col} ≤ {threshold:g} (n={len(group_a)})", f"{group_a.mean():.4g}"),
                    (f"{split_col} > {threshold:g} (n={len(group_b)})", f"{group_b.mean():.4g}"),
                    ("t-statistic", f"{test.statistic:.3f}"), ("p-value", f"{test.p_value:.4g}")])
            st.write(test.interpretation)
            fig = pub_plots.fig_cv_comparison({f"≤{threshold:g}": group_a.values, f">{threshold:g}": group_b.values},
                                              metric_name=metric_col, fig_no=1)
            png = pub_plots.save_fig_bytes(fig)
            register_figure(f"comparison_{split_col}_{metric_col}", png)
            st.download_button("⬇️ Download 600 DPI comparison figure", png, "group_comparison_600dpi.png", "image/png")
        else:
            st.warning("Not enough data in one of the groups to run the test.")
