"""views/hf_design_view.py — Hydraulic fracturing design, pump schedule, DFIT."""
import numpy as np
import pandas as pd
import streamlit as st

from modules import plots, pub_plots
from modules.hf_design import HFDesignEngine
from views.common import caption, info_card, kpi_row, register_figure, section_header


def render(ctx=None):
    ss = st.session_state
    st.title("💧 HF Design & DFIT")
    caption("PKN/KGD/Radial fracture geometry, proppant pump schedule (integrated ramp), "
            "post-frac productivity, and G-function DFIT closure analysis.")
    st.divider()

    section_header("Treatment Design Parameters", "⚙️",
                   help="These describe the planned hydraulic-fracturing treatment, separate from "
                        "the sidebar's perforation design parameters.")
    hp = ss["hf_params"]
    c1, c2, c3, c4 = st.columns(4)
    hp["model"] = c1.selectbox("Fracture model", ["PKN", "KGD", "Radial"],
                               index=["PKN", "KGD", "Radial"].index(hp.get("model", "PKN")),
                               help="PKN: long, contained fractures (height << length). KGD: short, "
                                    "wide fractures. Radial: unconfined, disc-shaped growth.")
    hp["pump_rate_bpm"] = c2.number_input("Pump Rate (bpm)", 5.0, 100.0, float(hp["pump_rate_bpm"]),
                                          help="Injection rate in barrels per minute — higher rates "
                                               "create wider, faster-growing fractures.")
    hp["total_fluid_bbl"] = c3.number_input("Total Fluid (bbl)", 200.0, 20000.0, float(hp["total_fluid_bbl"]),
                                            help="Total treatment fluid volume (pad + proppant-laden stages + flush).")
    hp["proppant_mass_klbs"] = c4.number_input("Target Proppant (klbs)", 20.0, 2000.0, float(hp["proppant_mass_klbs"]),
                                               help="Total proppant mass (thousands of lbs) to place in the fracture.")
    c5, c6, c7 = st.columns(3)
    hp["fluid_viscosity_cp"] = c5.number_input("Fluid Viscosity (cp)", 1.0, 1000.0, float(hp["fluid_viscosity_cp"]),
                                               help="Higher-viscosity fluids carry more proppant but generate more friction pressure.")
    hp["target_xf_ft"] = c6.number_input("Target Fracture Half-Length (ft)", 50.0, 3000.0, float(hp["target_xf_ft"]),
                                         help="Design target for the propped fracture half-length, used for the Fcd/PI-improvement estimate.")
    hp["target_hf_ft"] = c7.number_input("Fracture Height (ft)", 10.0, 400.0, float(hp["target_hf_ft"]),
                                         help="Fracture height, typically constrained by bounding shale/stress barriers above and below the pay zone.")
    ss["hf_params"] = hp

    if st.button("▶️ Run HF Design", type="primary"):
        with st.spinner("Solving implicit fracture-length material balance (Brent's method)..."):
            result = HFDesignEngine().run(hp, ss["res_params"])
        ss["hf_results"] = result

    if ss.get("hf_results") is None:
        st.info("Click **Run HF Design** to see results.")
        return
    r = ss["hf_results"]

    section_header("Design Summary", "📊")
    kpi_row([
        ("Fracture Half-Length", f"{r['xf_ft']:.0f} ft"),
        ("Max Width", f"{r['w_max_in']:.3f} in"),
        ("Dimensionless Conductivity (Fcd)", f"{r['Fcd']:.2f}"),
        ("PI Improvement", f"{r['PI_improvement_pct']:.1f}%"),
    ])
    kpi_row([
        ("Net Pressure", f"{r['net_pressure_psi']:.0f} psi"),
        ("Fluid Efficiency", f"{r['fluid_efficiency']:.1%}"),
        ("ISIP", f"{r['ISIP_psi']:.0f} psi"),
        ("Closure Stress", f"{r['closure_stress_psi']:.0f} psi"),
    ])

    st.divider()
    section_header("Fracture Geometry &amp; Growth", "📐")
    c1, c2 = st.columns(2)
    with c1:
        st.plotly_chart(plots.plot_hf_geometry(r), width='stretch')
    with c2:
        st.plotly_chart(plots.plot_hf_time_evolution(r["time_evolution"]), width='stretch')
    st.plotly_chart(plots.plot_hf_3d(r), width='stretch')

    st.divider()
    section_header("Pump Schedule (Integrated Proppant Ramp)", "🚰")
    st.caption("v3 fix: proppant mass per stage now integrates the linear concentration ramp "
               "(`scipy.integrate.quad`) instead of treating each stage as an instant jump to its "
               "target concentration.")
    st.dataframe(r["schedule_df"], width='stretch', hide_index=True)
    st.plotly_chart(plots.plot_pump_schedule(r["schedule_df"]), width='stretch')

    st.divider()
    section_header("DFIT G-Function Analysis", "📉")
    st.plotly_chart(plots.plot_dfit_gfunction(r["dfit_df"]), width='stretch')

    st.divider()
    section_header("Conductivity Sensitivity", "🌊")
    c1, c2 = st.columns(2)
    with c1:
        st.plotly_chart(plots.plot_fcd_sensitivity(r["xf_table"]), width='stretch')

    with st.expander("📸 Export 600 DPI HF design figure", expanded=False):
        if st.button("Generate figure"):
            import matplotlib.pyplot as plt

            from modules.pub_plots import _finish, save_fig_bytes, set_pub_style
            set_pub_style()
            fig, axes = plt.subplots(1, 2, figsize=(11, 4.6))
            te = r["time_evolution"]
            axes[0].plot(te["t_min"], te["xf_ft"], color="#2E75B6", linewidth=1.5)
            axes[0].set_xlabel("Pump time (min)"); axes[0].set_ylabel("Fracture half-length (ft)")
            axes[0].set_title("(a) Implicit PKN Growth (Brent solve)")
            axes[1].plot(r["schedule_df"]["cum_vol_bbl"], r["schedule_df"]["cum_prop_klbs"],
                        color="#FF7F0E", linewidth=1.5, marker="o", markersize=3)
            axes[1].set_xlabel("Cumulative volume (bbl)"); axes[1].set_ylabel("Cumulative proppant (klbs)")
            axes[1].set_title("(b) Integrated Proppant Schedule")
            _finish(fig, "Hydraulic Fracturing Design Summary")
            png = save_fig_bytes(fig)
            register_figure("hf_design_summary", png)
            st.image(png)
            st.download_button("⬇️ Download PNG (600 DPI)", png, "hf_design_600dpi.png", "image/png")
