"""
views/statistical_validation.py — Dedicated statistical-validation console.

A single place that gathers every hypothesis test, GOF diagnostic, and
significance report generated elsewhere in the app, plus an ad-hoc test
runner — the closest analogue to what v2 was missing entirely.
"""
import numpy as np
import pandas as pd
import streamlit as st

from config import ALPHA, tip
from modules.stats_engine import ConfidenceIntervals, GoodnessOfFit, HypothesisTests
from views.common import caption, info_card, kpi_row, register_figure, section_header, significance_pill


def render(ctx=None):
    ss = st.session_state
    st.title("🧪 Statistical Validation Console")
    caption(f"Every test on this page is run at significance level α = {ALPHA} (95% confidence). "
            "v2 performed zero hypothesis tests, GOF diagnostics, or confidence intervals anywhere "
            "in the codebase; this console — and the tests embedded in the ML Lab and Benchmarks "
            "pages — is the direct fix.")
    st.divider()

    if ss.get("reservoir_df") is None:
        info_card("No dataset yet. Head to <b>📂 Data Hub</b> first.")
        return
    df = ss["reservoir_df"]

    tab1, tab2, tab3 = st.tabs(["📊 Model Comparison Report", "🧮 Ad-hoc Hypothesis Test", "📐 GOF Reference"])

    with tab1:
        if ss.get("ml_results") is None or "comparison" not in ss["ml_results"]:
            st.info("Train models with cross-validation enabled on the **ML & Statistical Lab** "
                    "page to populate this report.")
        else:
            comp = ss["ml_results"]["comparison"]
            section_header("Cross-Validated Model Summary", "📈", help="Populated automatically after you run cross-validated training on the ML & Statistical Lab page.")
            st.dataframe(comp["summary"], width='stretch', hide_index=True)
            omni = comp["omnibus"]
            st.markdown(f"**{omni.name}:** statistic={omni.statistic:.4f}, p={omni.p_value:.4g} — "
                       + significance_pill(omni.significant, omni.p_value), unsafe_allow_html=True)
            st.write(omni.interpretation)
            section_header("Pairwise Comparisons (Holm-corrected)", "🔗", help="Every model-vs-model paired t-test, with p-values adjusted for the number of comparisons made.")
            st.dataframe(comp["pairwise"], width='stretch', hide_index=True)
            st.success(f"Statistically indistinguishable top tier: {', '.join(comp['best_model_set'])}")

    with tab2:
        section_header("Run a Custom Test", "🧮", help="Pick any two columns from the working dataset and run a standard hypothesis test on them.")
        numeric_cols = df.select_dtypes("number").columns.tolist()
        test_type = st.selectbox("Test", [
            "One-sample t-test (mean = value)",
            "Independent two-sample t-test",
            "Paired t-test",
            "Wilcoxon signed-rank (non-parametric paired)",
            "One-way ANOVA (3+ groups by category)",
            "Pearson / Spearman correlation",
        ])

        if test_type == "One-sample t-test (mean = value)":
            col = st.selectbox("Column", numeric_cols)
            popmean = st.number_input("Hypothesized mean (H0)", value=float(df[col].mean()))
            if st.button("Run test"):
                res = HypothesisTests.one_sample_ttest(df[col].dropna(), popmean)
                st.dataframe(pd.DataFrame([res.to_dict()]), width='stretch', hide_index=True)
                st.write(res.interpretation)

        elif test_type == "Independent two-sample t-test":
            col = st.selectbox("Metric column", numeric_cols)
            split_col = st.selectbox("Split column", numeric_cols, index=min(1, len(numeric_cols)-1))
            thresh = st.number_input("Split threshold", value=float(df[split_col].median()))
            if st.button("Run test"):
                a = df[df[split_col] <= thresh][col].dropna()
                b = df[df[split_col] > thresh][col].dropna()
                res = HypothesisTests.independent_ttest(a, b)
                st.dataframe(pd.DataFrame([res.to_dict()]), width='stretch', hide_index=True)
                st.write(res.interpretation)

        elif test_type in ("Paired t-test", "Wilcoxon signed-rank (non-parametric paired)"):
            c1, c2 = st.columns(2)
            col_a = c1.selectbox("Column A", numeric_cols)
            col_b = c2.selectbox("Column B", numeric_cols, index=min(1, len(numeric_cols)-1))
            if st.button("Run test"):
                fn = HypothesisTests.paired_ttest if "Paired t" in test_type else HypothesisTests.wilcoxon_signed_rank
                res = fn(df[col_a].dropna(), df[col_b].dropna())
                st.dataframe(pd.DataFrame([res.to_dict()]), width='stretch', hide_index=True)
                st.write(res.interpretation)

        elif test_type == "One-way ANOVA (3+ groups by category)":
            col = st.selectbox("Metric column", numeric_cols)
            group_col = st.selectbox("Grouping column", numeric_cols, index=min(1, len(numeric_cols)-1))
            if st.button("Run test"):
                groups = [g.dropna().values for _, g in df.groupby(pd.qcut(df[group_col], 4, duplicates="drop"))[col]]
                res = HypothesisTests.one_way_anova(*groups)
                st.dataframe(pd.DataFrame([res.to_dict()]), width='stretch', hide_index=True)
                st.write(res.interpretation)
                st.caption(f"Grouping column split into quartiles of {group_col}.")

        elif test_type == "Pearson / Spearman correlation":
            c1, c2, c3 = st.columns(3)
            col_a = c1.selectbox("Column A", numeric_cols)
            col_b = c2.selectbox("Column B", numeric_cols, index=min(1, len(numeric_cols)-1))
            method = c3.selectbox("Method", ["pearson", "spearman"])
            if st.button("Run test"):
                res = HypothesisTests.correlation_test(df[col_a], df[col_b], method=method)
                st.dataframe(pd.DataFrame([res.to_dict()]), width='stretch', hide_index=True)
                st.write(res.interpretation)

    with tab3:
        section_header("Goodness-of-Fit Reference", "📐", help=tip("gof"))
        col = st.selectbox("Column to diagnose", df.select_dtypes("number").columns.tolist())
        v = df[col].dropna()
        rows = [GoodnessOfFit.shapiro_wilk(v).to_dict(), GoodnessOfFit.jarque_bera(v).to_dict(),
                GoodnessOfFit.kolmogorov_smirnov(v).to_dict()]
        st.dataframe(pd.DataFrame(rows), width='stretch', hide_index=True)
        st.markdown("""
**Method notes** (for methodology sections):
- **Shapiro-Wilk** — most powerful normality test for small-to-moderate samples (Shapiro & Wilk, 1965).
- **Jarque-Bera** — normality test based on sample skewness and kurtosis (Jarque & Bera, 1980).
- **Kolmogorov-Smirnov** — compares the empirical CDF to a fitted normal CDF.
- **Durbin-Watson** — detects first-order autocorrelation in residuals (Durbin & Watson, 1951); used
  throughout the ML Lab's residual diagnostics.
- **Breusch-Pagan** — Lagrange-multiplier test for heteroscedasticity (Breusch & Pagan, 1979).
- All p-values are two-sided unless otherwise noted; α = 0.05 throughout, with Holm's step-down
  correction (Holm, 1979) applied whenever more than two models/groups are compared pairwise.
""")
