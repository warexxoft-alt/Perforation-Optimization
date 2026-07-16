"""views/optimization.py — Constrained NSGA-II multi-objective optimization."""
import numpy as np
import pandas as pd
import streamlit as st

from config import CONSTRAINTS, OPT_BOUNDS
from modules import plots, pub_plots
from modules.optimizer import NSGA2, ObjectiveFunctions, ParetoAnalyzer
from views.common import badge, caption, info_card, kpi_row, register_figure, section_header


def render(ctx=None):
    ss = st.session_state
    st.title("🎯 NSGA-II Optimization")
    caption("8 objectives (max PI, EUR, NPV, SRV; min skin, cost, FDI, sand risk) subject to "
            "6 real constraints — v2's optimizer ran fully unconstrained despite documenting 6 "
            "constraints that were never implemented.")
    st.divider()

    with st.expander("📋 Constraint definitions", expanded=False):
        obj_preview = ObjectiveFunctions(ss["res_params"])
        for name in obj_preview.constraint_names():
            st.write(f"- {name}")
        st.caption(f"Configurable in `config.CONSTRAINTS`: {CONSTRAINTS}")

    section_header("Optimizer Settings", "⚙️",
                   help="Larger population/more generations explore the design space more "
                        "thoroughly but take longer to run.")
    c1, c2, c3, c4 = st.columns(4)
    pop_size = c1.number_input("Population size", 20, 300, 80, step=10,
                               help="How many candidate designs are evaluated in parallel each generation.")
    n_gen = c2.number_input("Generations", 10, 300, 60, step=10,
                            help="How many rounds of selection/crossover/mutation to run. Watch the "
                                 "hypervolume convergence chart after running — a flat curve means "
                                 "more generations won't help much further.")
    use_constraints = c3.checkbox("Enforce constraints", value=True,
                                  help="Uncheck to see the raw, unconstrained trade-off surface — "
                                       "useful for understanding how much the constraints cost you.")
    do_refine = c4.checkbox("Memetic local refinement", value=True,
                            help="After NSGA-II converges, polishes a handful of representative "
                                 "designs with local optimisation (L-BFGS-B) — can only improve the "
                                 "front, never make it worse.")

    if st.button("🚀 Run NSGA-II", type="primary"):
        obj = ObjectiveFunctions(ss["res_params"])
        bounds = list(OPT_BOUNDS.values())
        cfn = obj.constraint_violation if use_constraints else None
        opt = NSGA2(obj.evaluate, bounds, constraint_fn=cfn, pop_size=int(pop_size), n_gen=int(n_gen), seed=ss["random_seed"])
        prog = st.progress(0.0, text="Evolving population...")
        pop, F, pareto_idx = opt.run(progress_cb=lambda f: prog.progress(f, text=f"Generation {int(f*n_gen)}/{n_gen}"))
        prog.empty()
        if do_refine:
            with st.spinner("Memetic local-search polish on representative Pareto solutions..."):
                pop, F, pareto_idx = opt.refine(pop, F, pareto_idx, n_refine=8)
        pa = ParetoAnalyzer()
        pareto_df = pa.decode(F[pareto_idx], pop[pareto_idx], constraint_fn=cfn)
        ss["opt_results"] = {"pop": pop, "F": F, "pareto_idx": pareto_idx, "obj": obj,
                             "history": opt.history, "pareto_df": pareto_df, "analyzer": pa}
        st.success(f"Converged: {len(pareto_idx)} Pareto-optimal designs found "
                  f"({int((pareto_df['Feasible?']=='Yes').sum())} feasible).")

    if ss.get("opt_results") is None:
        return
    res = ss["opt_results"]
    pareto_df, history = res["pareto_df"], res["history"]

    st.divider()
    section_header("Convergence Diagnostics", "📈", help="Hypervolume measures how much of the normalized objective space the Pareto front dominates — should rise then plateau as the optimizer converges.")
    hist_df = pd.DataFrame(history)
    c1, c2 = st.columns(2)
    with c1:
        import plotly.graph_objects as go

        from config import COLORS, PLOTLY_THEME
        fig = go.Figure(go.Scatter(x=hist_df["gen"], y=hist_df["hypervolume"], line=dict(color=COLORS["primary"])))
        fig.update_layout(template=PLOTLY_THEME, title="Normalized Hypervolume vs. Generation",
                          xaxis_title="Generation", yaxis_title="Hypervolume (normalized)", height=380)
        st.plotly_chart(fig, width='stretch')
    with c2:
        fig2 = go.Figure(go.Scatter(x=hist_df["gen"], y=hist_df["feasible_fraction"], line=dict(color=COLORS["good"])))
        fig2.update_layout(template=PLOTLY_THEME, title="Feasible Fraction of Population vs. Generation",
                           xaxis_title="Generation", yaxis_title="Feasible fraction", height=380)
        st.plotly_chart(fig2, width='stretch')

    n_feas = int((pareto_df["Feasible?"] == "Yes").sum())
    if n_feas == 0:
        badge("⚠️ No feasible designs found — relax constraints or bounds", "bad")
    else:
        badge(f"✅ {n_feas}/{len(pareto_df)} Pareto designs are fully feasible", "good")

    st.divider()
    section_header("Pareto Front", "🌐", help="Every point is a distinct design where you cannot improve one objective without worsening another. Hollow red markers are infeasible (violate a constraint).")
    c1, c2 = st.columns([1, 1])
    obj_cols = ["PI (STB/d/psi)", "EUR (MMbbl)", "NPV (M$)", "SRV (MMft3)",
               "Skin S_total", "Cost (M$)", "FDI", "Sand Risk"]
    x_col = c1.selectbox("X-axis", obj_cols, index=2)
    y_col = c2.selectbox("Y-axis", obj_cols, index=0)
    st.plotly_chart(plots.plot_pareto_2d(pareto_df, x_col, y_col), width='stretch')
    st.plotly_chart(plots.plot_pareto_3d(pareto_df), width='stretch')

    with st.expander("📸 Export 600 DPI Pareto front figure", expanded=False):
        if st.button("Generate figure"):
            fig = pub_plots.fig_pareto_2d(pareto_df, x_col, y_col, fig_no=1)
            png = pub_plots.save_fig_bytes(fig)
            register_figure("pareto_front", png)
            st.image(png)
            st.download_button("⬇️ Download PNG (600 DPI)", png, "pareto_front_600dpi.png", "image/png")

    st.divider()
    section_header("Recommended Designs", "🏅", help="Three natural picks from the feasible Pareto front: highest NPV, highest PI, and the balanced (TOPSIS closest-to-ideal) compromise.")
    pa = res["analyzer"]
    feas_df = pareto_df[pareto_df["Feasible?"] == "Yes"] if "Feasible?" in pareto_df.columns else pareto_df
    if len(feas_df) == 0:
        st.warning("No feasible designs to recommend — showing best available (infeasible) design.")
        feas_df = pareto_df
    tabs = st.tabs(["💰 Best NPV", "⚡ Best PI", "⚖️ Best Balanced"])
    with tabs[0]:
        st.dataframe(pa.best_npv(feas_df).astype(str).to_frame("Value"), width='stretch')
    with tabs[1]:
        st.dataframe(pa.best_pi(feas_df).astype(str).to_frame("Value"), width='stretch')
    with tabs[2]:
        st.dataframe(pa.best_balanced(feas_df).astype(str).to_frame("Value"), width='stretch')

    st.divider()
    section_header("Full Pareto Table", "📋")
    st.dataframe(pareto_df, width='stretch')
    st.download_button("⬇️ Download Pareto front (CSV)", pareto_df.to_csv(index=False).encode(),
                       "pareto_front.csv", "text/csv")
