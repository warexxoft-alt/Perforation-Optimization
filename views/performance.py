"""views/performance.py — Performance-metrics deep dive with statistical rigor."""
import numpy as np
import pandas as pd
import streamlit as st

from modules import pub_plots
from modules.analysis import PerformanceMetrics
from modules.stats_engine import ConfidenceIntervals, GoodnessOfFit
from views.common import caption, info_card, kpi_row, register_figure, section_header


def render(ctx=None):
    ss = st.session_state
    st.title("📈 Performance Metrics")
    caption("Deep dive into any KPI: distribution shape, normality diagnostics, and confidence "
            "intervals — the statistical grounding v2's single-number summaries lacked.")
    st.divider()

    if ss.get("reservoir_df") is None:
        info_card("No dataset yet. Head to <b>📂 Data Hub</b> first.")
        return
    df = ss["reservoir_df"]
    pm = PerformanceMetrics()

    section_header("Full Metric Summary", "📏", help="Descriptive statistics with 95% confidence intervals and a normality test for every KPI.")
    summ = pm.summary_table(df)
    st.dataframe(summ, width='stretch', hide_index=True)

    st.divider()
    section_header("Single-Metric Deep Dive", "🔎", help="Full distributional diagnostics — parametric and bootstrap confidence intervals, three normality tests — for one metric at a time.")
    metric_options = [c for c in pm.METRIC_DEFS if c in df.columns]
    metric = st.selectbox("Metric", metric_options,
                          format_func=lambda c: pm.METRIC_DEFS[c][0])
    v = df[metric].dropna()

    kpi_row([("Mean", f"{v.mean():.4g}"), ("Median", f"{v.median():.4g}"),
            ("Std Dev", f"{v.std():.4g}"), ("N", f"{len(v):,}")])

    ci = ConfidenceIntervals.t_interval(v)
    boot_ci = ConfidenceIntervals.bootstrap_ci(v, np.mean)
    c1, c2 = st.columns(2)
    c1.info(f"**Parametric 95% CI (t-dist):** [{ci[0]:.4g}, {ci[1]:.4g}]")
    c2.info(f"**Bootstrap 95% CI ({boot_ci['n_boot']} resamples):** [{boot_ci['lower']:.4g}, {boot_ci['upper']:.4g}]")

    sw = GoodnessOfFit.shapiro_wilk(v)
    ks = GoodnessOfFit.kolmogorov_smirnov(v)
    jb = GoodnessOfFit.jarque_bera(v)
    st.dataframe(pd.DataFrame([sw.to_dict(), ks.to_dict(), jb.to_dict()]), width='stretch', hide_index=True)

    import plotly.graph_objects as go

    from config import COLORS, PLOTLY_THEME
    fig = go.Figure()
    fig.add_trace(go.Histogram(x=v, histnorm="probability density", marker_color=COLORS["primary"],
                               opacity=0.75, name="Empirical"))
    from scipy import stats as spstats
    xs = np.linspace(v.min(), v.max(), 200)
    fig.add_trace(go.Scatter(x=xs, y=spstats.norm.pdf(xs, v.mean(), v.std()), line=dict(color=COLORS["accent"], width=2.5),
                             name="Normal fit"))
    fig.update_layout(template=PLOTLY_THEME, height=420, title=f"Distribution: {pm.METRIC_DEFS[metric][0]}",
                      xaxis_title=pm.METRIC_DEFS[metric][1])
    st.plotly_chart(fig, width='stretch')

    if st.button("📸 Generate 600 DPI distribution figure"):
        import matplotlib.pyplot as plt

        from modules.pub_plots import _finish, save_fig_bytes, set_pub_style
        set_pub_style()
        f, ax = plt.subplots(figsize=(6.6, 5.0))
        ax.hist(v, bins=min(40, max(10, len(v)//10)), density=True, color=COLORS["primary"], alpha=0.75, edgecolor="white")
        ax.plot(xs, spstats.norm.pdf(xs, v.mean(), v.std()), color=COLORS["accent"], linewidth=1.6)
        ax.set_xlabel(pm.METRIC_DEFS[metric][1]); ax.set_ylabel("Density")
        _finish(f, f"Distribution of {pm.METRIC_DEFS[metric][0]} (n={len(v)})")
        png = save_fig_bytes(f)
        register_figure(f"distribution_{metric}", png)
        st.image(png)
        st.download_button("⬇️ Download PNG (600 DPI)", png, f"{metric}_distribution_600dpi.png", "image/png")
