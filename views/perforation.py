"""views/perforation.py — Single-well perforation simulation: skin breakdown, IPR, decline."""
import numpy as np
import pandas as pd
import streamlit as st

from config import BENCHMARKS
from modules import plots, pub_plots
from modules.simulator import CoupledSimulator
from views.common import badge, caption, kpi_row, register_figure, section_header


def render(ctx=None):
    ss = st.session_state
    st.title("⚙️ Perforation Simulation")
    caption("Single-well coupled simulation: Karakas-Tariq skin, Darcy/Vogel IPR, Arps decline, "
            "sand-control drawdown check, and NPV — using the reservoir/perforation parameters "
            "in the sidebar.")
    st.divider()

    p = ss["res_params"]
    if st.button("▶️ Run simulation", type="primary"):
        with st.spinner("Running coupled simulation (iterating rate ↔ non-Darcy skin to convergence)..."):
            result = CoupledSimulator().run_well(p)
        ss["perf_result"] = result

    if ss.get("perf_result") is None:
        st.info("Click **Run simulation** to see results.")
        return
    r = ss["perf_result"]

    section_header("Key Results", "📊", help="Outputs of the coupled physics simulation for the design currently set in the sidebar.")
    kpi_row([
        ("Productivity Index", f"{r['PI_stb_day_psi']:.4f} STB/d/psi"),
        ("Total Skin", f"{r['S_total']:.3f}"),
        ("Initial Rate", f"{r['q_initial_stbd']:.0f} STB/d"),
        ("EUR", f"{r['EUR_MMbbl']:.3f} MMbbl"),
    ])
    kpi_row([
        ("NPV", f"${r['NPV_MM']:.2f} MM"),
        ("Payback", f"{r['payback_yr']:.2f} yr" if np.isfinite(r['payback_yr']) else "N/A"),
        ("Sand Risk", f"{r['sand_risk']:.3f}"),
        ("Rate/Skin Iterations", f"{r['rate_skin_iterations']} ({'converged' if r['rate_skin_converged'] else 'capped'})"),
    ])
    if r["sand_risk"] > 1.0:
        badge("⚠️ Sand-control risk: drawdown exceeds critical threshold", "bad")
    else:
        badge("✅ Within sand-control limits", "good")

    st.divider()
    section_header("Skin Component Breakdown", "🧩", help="How each mechanism (perforation geometry, damage, non-Darcy turbulence) contributes to total skin.")
    c1, c2 = st.columns(2)
    with c1:
        st.plotly_chart(plots.plot_skin_waterfall(r), width='stretch')
    with c2:
        st.plotly_chart(plots.plot_2d_ipr(r), width='stretch')

    st.divider()
    section_header("Production Forecast", "📈")
    c1, c2 = st.columns(2)
    with c1:
        st.plotly_chart(plots.plot_2d_decline(r), width='stretch')
    with c2:
        st.plotly_chart(plots.plot_2d_cumulative(r), width='stretch')

    with st.expander("📸 Generate camera-ready 600 DPI figures for this well", expanded=False):
        if st.button("Generate figures"):
            import matplotlib.pyplot as plt

            from modules.pub_plots import _finish, save_fig_bytes, set_pub_style
            set_pub_style()
            fig, axes = plt.subplots(1, 2, figsize=(11, 4.6))
            axes[0].plot(r["t_arr"], r["q_arr"], color="#2E75B6", linewidth=1.5)
            axes[0].set_xlabel("Time (yr)"); axes[0].set_ylabel("Rate (STB/d)"); axes[0].set_title("(a) Decline")
            axes[1].plot(r["q_ipr"], r["Pwf_ipr"], color="#2E75B6", linewidth=1.5)
            axes[1].set_xlabel("Rate (STB/d)"); axes[1].set_ylabel("BHFP (psi)"); axes[1].set_title("(b) IPR")
            _finish(fig, "Single-Well Simulation Summary")
            png = save_fig_bytes(fig)
            register_figure("perforation_summary", png)
            st.image(png)
            st.download_button("⬇️ Download PNG (600 DPI)", png, "perforation_summary_600dpi.png", "image/png")

    st.divider()
    section_header("Industry Benchmark Position", "🏆", help="Where this single design sits relative to published Niger Delta benchmarks.")
    from modules.analysis import BenchmarkComparator
    bc = BenchmarkComparator()
    st.write(f"This design's PI percentile rank: **{bc.percentile_rank(r['PI_stb_day_psi'], BENCHMARKS['PI_stb_day_psi'])}**")
    st.dataframe(bc.compare_scalar(r["PI_stb_day_psi"], "PI_stb_day_psi"), width='stretch', hide_index=True)
