"""
views/visualization.py — 2D through 9D interactive results exploration.

v3 enhancement: dimensional visualization is no longer limited to the raw
synthetic/uploaded dataset. A data-source selector lets the same 2D-9D
toolkit explore ML model predictions (multi-model predicted-vs-actual),
the NSGA-II Pareto front (all 8 objectives via parallel coordinates/SPLOM),
or Sobol sensitivity indices — i.e. it now visualizes *results*, not just
raw inputs.
"""
import pandas as pd
import streamlit as st

from modules import plots
from views.common import caption, info_card, section_header


def render(ctx=None):
    ss = st.session_state
    st.title("🌐 2D–9D Visualization")
    caption("Dimensional encoding: 3D = x,y,z · 4D = +colour · 5D = +size · 6D = +symbol · "
            "7D = +animation · 8D = +facets · 9D = parallel coordinates / SPLOM. Choose a data "
            "source below — raw data, ML predictions, the optimisation Pareto front, or "
            "sensitivity results all use the same toolkit.")
    st.divider()

    if ss.get("reservoir_df") is None:
        info_card("No dataset yet. Head to <b>📂 Data Hub</b> first.")
        return
    df = ss["reservoir_df"]

    sources = ["📊 Raw Dataset"]
    if ss.get("ml_results") is not None:
        sources.append("🧠 ML Predictions")
    if ss.get("opt_results") is not None:
        sources.append("🎯 Optimization Pareto Front")
    if ss.get("sobol_df") is not None:
        sources.append("🎛️ Sensitivity Results")

    source = st.radio("Data source", sources, horizontal=True,
                      help="Results from other pages become available here automatically once "
                           "you've run them (ML Lab, NSGA-II Optimization, Results Analysis).")
    st.divider()

    if source == "📊 Raw Dataset":
        _render_raw_dataset(df)
    elif source == "🧠 ML Predictions":
        _render_ml_predictions(ss["ml_results"])
    elif source == "🎯 Optimization Pareto Front":
        _render_pareto(ss["opt_results"])
    elif source == "🎛️ Sensitivity Results":
        _render_sensitivity(ss["sobol_df"])


def _render_raw_dataset(df):
    dim = st.select_slider("Dimensionality", options=[3, 4, 5, 6, 7, 8, 9], value=4,
                           help="How many variables to encode simultaneously — position, colour, "
                                "size, symbol, animation, facets, and (at 9D) every numeric column at once.")

    if dim == 3:
        c1, c2 = st.columns(2)
        with c1: st.plotly_chart(plots.plot_3d_surface(df), width='stretch')
        with c2: st.plotly_chart(plots.plot_3d_scatter(df), width='stretch')
    elif dim == 4:
        st.plotly_chart(plots.plot_4d_scatter(df), width='stretch')
    elif dim == 5:
        st.plotly_chart(plots.plot_5d_scatter(df), width='stretch')
    elif dim == 6:
        st.plotly_chart(plots.plot_6d_scatter(df), width='stretch')
    elif dim == 7:
        st.plotly_chart(plots.plot_7d_animated(df), width='stretch')
    elif dim == 8:
        st.plotly_chart(plots.plot_8d_faceted(df), width='stretch')
    elif dim == 9:
        tab1, tab2 = st.tabs(["Parallel Coordinates", "Scatter Plot Matrix"])
        with tab1: st.plotly_chart(plots.plot_9d_parallel(df), width='stretch')
        with tab2: st.plotly_chart(plots.plot_9d_splom(df), width='stretch')

    st.divider()
    with st.expander("🪵 Well Log Cross-Plot & Grid Slice (if data available)", expanded=False):
        bundle = st.session_state.get("uploaded_bundle")
        if bundle is not None and bundle.well_logs is not None and len(bundle.well_logs) > 0:
            st.plotly_chart(plots.plot_log_view(bundle.well_logs), width='stretch')
        elif bundle is not None and bundle.grid is not None:
            st.plotly_chart(plots.plot_grid_slice(bundle.grid), width='stretch')
        else:
            st.caption("Upload a LAS log or GRDECL grid in the Data Hub to unlock this view.")


def _render_ml_predictions(results):
    section_header("ML Prediction Space", "🧠",
                   help="Compare every trained model's predicted-vs-actual quality in one "
                        "multi-dimensional view: position=actual/predicted, colour=which model, "
                        "size=how wrong that prediction was.")
    models_available = [k for k in results if k in ("linear", "ensemble", "kriging", "pinn")]
    picked = st.multiselect("Models to include", models_available, default=models_available)
    st.plotly_chart(plots.plot_ml_predictions_multid(results, picked), width='stretch')
    caption("Larger bubbles = larger absolute residual for that prediction. Hover to see the "
            "well, model, actual, and predicted values.")

    st.divider()
    section_header("Feature Correlation Matrix (SPLOM)", "🔗")
    feat_names = results.get("feat_names", [])
    if len(feat_names) >= 3 and "X_full" in results:
        Xdf = pd.DataFrame(results["X_full"], columns=feat_names)
        Xdf["target"] = results.get("y_full")
        cols = st.multiselect("Columns", list(Xdf.columns), default=list(Xdf.columns)[:5])
        if len(cols) >= 2:
            st.plotly_chart(plots.plot_nd_splom(Xdf, cols, color_col="target" if "target" in cols else None),
                            width='stretch')


def _render_pareto(opt_results):
    pareto_df = opt_results["pareto_df"]
    obj_cols = [c for c in pareto_df.columns if "(" in c or c == "Skin S_total" or c in ("FDI", "Sand Risk")]

    section_header("Full Pareto Front — All Objectives at Once", "🎯",
                   help="Every one of the 8 NSGA-II objectives simultaneously, colour-coded by "
                        "whichever objective you choose — the 'true' 9-dimensional (8 objectives + "
                        "colour) view of the trade-off space, versus picking one 2D/3D projection "
                        "at a time on the Optimization page.")
    color_by = st.selectbox("Colour by", obj_cols, index=obj_cols.index("NPV (M$)") if "NPV (M$)" in obj_cols else 0)
    tab1, tab2 = st.tabs(["Parallel Coordinates", "Scatter Plot Matrix"])
    with tab1:
        st.plotly_chart(plots.plot_nd_parallel(pareto_df, obj_cols, color_col=color_by,
                        title="NSGA-II Pareto Front — 8 Objectives (Parallel Coordinates)"), width='stretch')
    with tab2:
        show_cols = st.multiselect("Objectives to include in the matrix", obj_cols, default=obj_cols[:4])
        if len(show_cols) >= 2:
            st.plotly_chart(plots.plot_nd_splom(pareto_df, show_cols, color_col=color_by), width='stretch')

    st.divider()
    section_header("Custom Multi-D Projection", "🔬")
    c1, c2, c3, c4 = st.columns(4)
    x = c1.selectbox("X", obj_cols, index=0)
    y = c2.selectbox("Y", obj_cols, index=1)
    size_c = c3.selectbox("Size (optional)", ["— none —"] + obj_cols, index=0)
    color_c = c4.selectbox("Colour (optional)", ["— none —"] + obj_cols, index=0)
    st.plotly_chart(plots.plot_nd_bubble(pareto_df, x, y,
                    size_col=None if size_c == "— none —" else size_c,
                    color_col=None if color_c == "— none —" else color_c), width='stretch')


def _render_sensitivity(sobol_df):
    section_header("Sensitivity Results in Multi-D Space", "🎛️",
                   help="S1 (first-order) vs ST (total-order, including interactions) Sobol "
                        "indices for every feature — the gap between them (ST-S1) shows how much "
                        "of that feature's influence comes from interacting with other features.")
    d = sobol_df.copy()
    d["Interaction (ST-S1)"] = d["ST"] - d["S1"]
    # Theoretically ST >= S1 always; Monte-Carlo sampling noise can very
    # occasionally push the estimated difference slightly negative, which
    # Plotly's marker-size encoding rejects outright — clip for the plot
    # only (the raw, unclipped values are still shown in the table below).
    size_vals = d["Interaction (ST-S1)"].clip(lower=0)
    st.plotly_chart(plots.plot_nd_bubble(d.assign(**{"Interaction (size)": size_vals}),
                    "S1", "ST", size_col="Interaction (size)",
                    color_col="feature", title="Sobol Sensitivity: S1 vs ST (size=interaction effect)"),
                    width='stretch')
    st.dataframe(d, width='stretch', hide_index=True)
