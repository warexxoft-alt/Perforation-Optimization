"""views/results_analysis.py — Best designs, sensitivity summary, tabulated results."""
import numpy as np
import pandas as pd
import streamlit as st

from modules import plots, pub_plots
from modules.analysis import ResultsTabulator
from modules.ml_engine import FEATURE_COLS, SensitivityAnalyzer
from views.common import caption, info_card, kpi_row, register_figure, section_header


def render(ctx=None):
    ss = st.session_state
    st.title("🔬 Results Analysis")
    caption("Composite-score ranking of top designs, and Sobol variance-based sensitivity "
            "analysis — fixed in v3 to check the feature count matches the model before "
            "calling `.predict()` (v2 crashed here with a silent feature-mismatch error).")
    st.divider()

    if ss.get("reservoir_df") is None:
        info_card("No dataset yet. Head to <b>📂 Data Hub</b> first.")
        return
    df = ss["reservoir_df"]
    rt = ResultsTabulator()

    section_header("Top-Ranked Designs (Composite Score)", "🏅", help="A weighted blend of PI, EUR, NPV, skin, FDI, and sand risk (see modules/analysis.py for the exact weights).")
    n_top = st.slider("Show top N", 5, 50, 10, help="How many top-ranked designs to display.")
    best = rt.best_designs_table(df, n_top=n_top)
    st.dataframe(best, width='stretch', hide_index=True)
    st.download_button("⬇️ Download top designs (CSV)", best.to_csv(index=False).encode(),
                       "top_designs.csv", "text/csv")

    st.divider()
    section_header("Sobol Sensitivity Analysis", "🎛️", help="Which design/reservoir variables actually drive the trained model's predictions, and how much of that is from interactions between variables.")
    if ss.get("ml_results") is None or "ensemble" not in ss["ml_results"]:
        st.info("Train the Ensemble model on the **ML & Statistical Lab** page first to enable sensitivity analysis.")
        return
    model = ss["ml_results"]["ensemble"]["model"]
    feat_names = ss["ml_results"]["feat_names"]   # exact features the model was trained on
    st.caption(f"Analyzing sensitivity w.r.t. these {len(feat_names)} features exactly as trained: "
              f"{', '.join(feat_names)} — v2 silently truncated this list to 6 regardless of how "
              f"many features the model actually used, which crashed `StandardScaler`.")
    n_samples = st.slider("Sobol sample size (higher = more accurate, slower)", 200, 3000, 800, step=200,
                          help="Number of Saltelli Monte-Carlo samples per feature — more samples reduce estimation noise in S1/ST at the cost of runtime.")

    if st.button("Run Sobol sensitivity analysis"):
        bounds = [(float(df[f].min()), float(df[f].max())) for f in feat_names]
        sa = SensitivityAnalyzer(lambda X: model.predict(X), len(feat_names), feat_names, N=n_samples)
        with st.spinner("Computing Sobol indices (Saltelli sampling)..."):
            sobol_df = sa.compute(bounds)
        ss["sobol_df"] = sobol_df

    if ss.get("sobol_df") is not None:
        sobol_df = ss["sobol_df"]
        st.dataframe(rt.sensitivity_summary(sobol_df), width='stretch', hide_index=True)
        st.plotly_chart(plots.plot_sobol_indices(sobol_df), width='stretch')
        if st.button("📸 Generate 600 DPI sensitivity tornado"):
            fig = pub_plots.fig_sobol_tornado(sobol_df, fig_no=1)
            png = pub_plots.save_fig_bytes(fig)
            register_figure("sobol_tornado", png)
            st.image(png)
            st.download_button("⬇️ Download PNG (600 DPI)", png, "sobol_tornado_600dpi.png", "image/png")
