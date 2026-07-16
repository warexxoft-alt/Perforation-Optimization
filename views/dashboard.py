"""views/dashboard.py — Landing page: onboarding checklist, KPIs, session overview."""
import numpy as np
import pandas as pd
import streamlit as st

from config import BENCHMARKS, VERSION
from modules.data_gen import NigerDeltaDataGenerator
from modules.simulator import CoupledSimulator
from views.common import badge, caption, info_card, kpi_row, progress_tracker, section_header, workflow_status


def render(ctx=None):
    ss = st.session_state
    st.title("🛢️ PerfoSim Pro v3")
    st.markdown("##### Niger Delta Perforation &amp; Hydraulic Fracturing Optimization Framework")
    caption("Real-Data Upload · Coupled Simulation · PINNs · NSGA-II · Statistical Validation · 2D–9D Visualization")
    st.divider()

    has_data = ss.get("reservoir_df") is not None
    steps = workflow_status()
    n_done = sum(steps.values())

    # ── First-time / in-progress onboarding checklist ─────────────────────
    if n_done < len(steps):
        with st.container(border=True):
            c1, c2 = st.columns([3, 1])
            with c1:
                st.markdown("#### 👋 New here? Follow the checklist")
                st.caption("PerfoSim Pro walks through a typical workflow in six steps. Complete them "
                          "in order, or jump straight to any page from the sidebar — nothing here is "
                          "mandatory. Full walkthroughs and definitions live on the **📖 User Guide & "
                          "Tutorial** page.")
            with c2:
                if st.button("📖 Open User Guide", width='stretch'):
                    st.session_state["_jump_to_guide_hint"] = True
                    st.info("Select **📖 User Guide & Tutorial** in the sidebar navigation (under Reporting).")
            progress_tracker(steps)

    if not has_data:
        info_card(
            "**Get started in one click.** Generate a synthetic Niger Delta dataset now, "
            "or head to <b>📂 Data Hub</b> to upload real field data (LAS, Excel, CSV, GRDECL, images)."
        )
        col1, col2 = st.columns([1, 3])
        with col1:
            if st.button("⚡ Generate sample dataset", type="primary", width='stretch',
                        help="Creates a calibrated synthetic Niger Delta dataset and runs the coupled "
                             "physics simulator on every well — usually a few seconds."):
                with st.spinner("Generating synthetic wells and running the coupled simulator..."):
                    df = NigerDeltaDataGenerator(ss["n_samples"], seed=ss["random_seed"]).generate()
                    df = CoupledSimulator().run_dataset(df)
                    ss["reservoir_df"] = df
                st.rerun()
        st.stop()

    df = ss["reservoir_df"]

    section_header("Session Overview", "📊", help="Live summary of the dataset currently loaded in this session.")
    kpi_row([
        ("Wells in dataset", f"{len(df):,}", None, "Total number of wells in the current working dataset."),
        ("Mean PI", f"{df['PI_stb_day_psi'].mean():.3f} STB/d/psi" if "PI_stb_day_psi" in df else "—",
         None, "Average Productivity Index across all wells — higher is better."),
        ("Mean EUR", f"{df['EUR_MMbbl'].mean():.2f} MMbbl" if "EUR_MMbbl" in df else "—",
         None, "Average Estimated Ultimate Recovery per well."),
        ("Mean NPV", f"${df['NPV_MM'].mean():.1f} MM" if "NPV_MM" in df else "—",
         None, "Average Net Present Value per well at the sidebar's economic assumptions."),
    ])

    st.write("")
    col1, col2, col3 = st.columns(3)
    with col1:
        st.markdown("**🧠 Machine Learning**")
        if steps["Train & validate ML models"]:
            badge("Models trained", "good")
            best = max(ss["ml_results"].get("comparison", {}).get("best_model_set", ["—"]))
            st.caption(f"Best model: {best}")
        else:
            badge("Not run yet", "warn")
            st.caption("Visit **ML & Statistical Lab** to train and cross-validate models.")
    with col2:
        st.markdown("**🎯 NSGA-II Optimization**")
        if steps["Run NSGA-II optimization"]:
            n_feas = int((ss["opt_results"]["pareto_df"]["Feasible?"] == "Yes").sum()) \
                if "pareto_df" in ss["opt_results"] else 0
            badge(f"{n_feas} feasible designs", "good")
        else:
            badge("Not run yet", "warn")
            st.caption("Visit **NSGA-II Optimization** to search the design space.")
    with col3:
        st.markdown("**💧 HF Design**")
        if steps["Design a hydraulic fracture"]:
            badge("Design computed", "good")
        else:
            badge("Not run yet", "warn")
            st.caption("Visit **HF Design & DFIT** to size a treatment.")

    st.divider()
    section_header("Distribution Snapshot", "📈",
                   help="A quick look at how skin and productivity vary across the current dataset. "
                        "For the full 2D-9D exploration (including ML/optimisation results), see the "
                        "2D–9D Visualization page.")
    from modules import plots
    c1, c2 = st.columns(2)
    with c1:
        st.plotly_chart(plots.plot_2d_skin_vs_spf(df), width='stretch')
    with c2:
        st.plotly_chart(plots.plot_2d_heatmap(df), width='stretch')

    st.divider()
    caption(f"PerfoSim Pro v{VERSION} · All statistical tests run at α = 0.05 significance level "
            f"· Benchmarks: {BENCHMARKS['PI_stb_day_psi']['source']}")
