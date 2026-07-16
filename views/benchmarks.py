"""views/benchmarks.py — Industry benchmark comparison, now backed by real t-tests."""
import pandas as pd
import streamlit as st

from config import BENCHMARKS
from modules import plots, pub_plots
from modules.analysis import BenchmarkComparator
from views.common import caption, info_card, register_figure, section_header, significance_pill


def render(ctx=None):
    ss = st.session_state
    st.title("🏆 Benchmark Comparison")
    caption("v2 only reported a raw delta against each published benchmark. v3 runs a one-sample "
            "t-test of this study's full distribution of results against each benchmark value.")
    st.divider()

    if ss.get("reservoir_df") is None:
        info_card("No dataset yet. Head to <b>📂 Data Hub</b> first.")
        return
    df = ss["reservoir_df"]
    bc = BenchmarkComparator()

    metric = st.selectbox("Metric to benchmark", [k for k in BENCHMARKS if k in df.columns],
                          help="Runs a one-sample t-test of this metric's full distribution against each published benchmark value.")
    st.dataframe(bc.compare_distribution(df[metric], metric), width='stretch', hide_index=True)
    caption(f"Source: {BENCHMARKS[metric].get('source', '—')}. Significance level α = 0.05.")

    st.plotly_chart(plots.plot_benchmark_bar(df[metric].mean(),
        {k: v for k, v in BENCHMARKS[metric].items() if k != "source"}, metric),
        width='stretch')

    if st.button("📸 Generate 600 DPI benchmark figure"):
        import matplotlib.pyplot as plt

        from config import COLORS
        from modules.pub_plots import _finish, save_fig_bytes, set_pub_style
        set_pub_style()
        bm = {k: v for k, v in BENCHMARKS[metric].items() if k != "source"}
        fig, ax = plt.subplots(figsize=(6.6, 5.0))
        labels = list(bm.keys()) + ["This Study"]
        vals = list(bm.values()) + [df[metric].mean()]
        colors = [COLORS["primary"]] * len(bm) + [COLORS["accent"]]
        ax.bar(labels, vals, color=colors, edgecolor="white")
        ax.set_ylabel(metric); plt.setp(ax.get_xticklabels(), rotation=25, ha="right")
        _finish(fig, f"Benchmark Comparison — {metric}")
        png = save_fig_bytes(fig)
        register_figure(f"benchmark_{metric}", png)
        st.image(png)
        st.download_button("⬇️ Download PNG (600 DPI)", png, f"benchmark_{metric}_600dpi.png", "image/png")

    st.divider()
    section_header("Literature Portfolio", "📚", help="Comparable published studies' reported improvements, for context.")
    st.dataframe(bc.literature_table(), width='stretch', hide_index=True)

    st.divider()
    section_header("Radar: Multi-Metric Position vs. Industry Average", "🕸️", help="Each axis is normalised 0-1 against that metric's best published benchmark value.")
    radar_metrics = [k for k in ["PI_stb_day_psi", "EUR_MMbbl", "NPV_MM", "recovery_factor"] if k in df.columns and k in BENCHMARKS]
    if radar_metrics:
        study_vals, bench_vals = [], []
        for m in radar_metrics:
            bm = BENCHMARKS[m]
            ref = bm.get("industry_avg", bm.get("target", list(bm.values())[0]))
            mx = max(v for k, v in bm.items() if k != "source")
            study_vals.append(float(df[m].mean()) / mx)
            bench_vals.append(float(ref) / mx)
        st.plotly_chart(plots.plot_radar_benchmark(radar_metrics, study_vals, bench_vals), width='stretch')
