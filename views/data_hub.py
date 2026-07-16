"""views/data_hub.py — Synthetic data generation + real-data upload/ingestion."""
import numpy as np
import pandas as pd
import streamlit as st

from config import SUPPORTED_FORMATS
from modules.data_gen import NigerDeltaDataGenerator
from modules.file_ingestion import FileIngestionEngine
from modules.simulator import CoupledSimulator
from views.common import badge, caption, info_card, kpi_row, section_header


def render(ctx=None):
    ss = st.session_state
    st.title("📂 Data Hub")
    caption("Generate a synthetic Niger Delta dataset, or upload real field data "
            "(LAS logs, Excel/CSV, Eclipse GRDECL grids, core/CT images).")
    st.divider()

    tab1, tab2 = st.tabs(["⚡ Synthetic Generator", "📤 Upload Real Data"])

    with tab1:
        section_header("Synthetic Niger Delta Dataset", "🧪", help="Instantly generates a full, physically calibrated dataset — the fastest way to explore the app without real field data.")
        st.write("Uses calibrated distributions for permeability, net pay, API gravity, GOR, "
                 "geomechanical properties, and completion design — see `modules/data_gen.py` "
                 "for the full parameter ranges and literature sources.")
        c1, c2, c3 = st.columns(3)
        n = c1.number_input("Number of wells", 50, 5000, ss["n_samples"], step=50)
        seed = c2.number_input("Random seed", 0, 99999, ss["random_seed"])
        run_sim = c3.checkbox("Run coupled simulator immediately", value=True)
        if st.button("Generate dataset", type="primary"):
            with st.spinner("Generating synthetic wells..."):
                df = NigerDeltaDataGenerator(int(n), seed=int(seed)).generate()
                if run_sim:
                    df = CoupledSimulator().run_dataset(df)
                ss["reservoir_df"] = df
                ss["n_samples"], ss["random_seed"] = int(n), int(seed)
            st.success(f"Generated {len(df):,} wells" + (" and ran the coupled simulator." if run_sim else "."))

    with tab2:
        section_header("Upload Real Field Data", "📤", help="Auto-detects the format and shows you exactly what was extracted before you commit to using it.")
        st.write(f"Supported formats: {', '.join(sorted(set(SUPPORTED_FORMATS.values())))}"
                 if isinstance(SUPPORTED_FORMATS, dict) else "LAS · CSV/Excel · GRDECL · DICOM · Images")
        upl = st.file_uploader("Drop a file here",
                               type=sorted({e.lstrip(".").lower() for e in SUPPORTED_FORMATS.keys()}),
                               accept_multiple_files=False)
        if upl is not None:
            with st.spinner(f"Ingesting {upl.name}..."):
                try:
                    engine = FileIngestionEngine()
                    bundle = engine.ingest(upl, upl.name)
                    ss["uploaded_bundle"] = bundle
                except Exception as exc:
                    st.error(f"Could not read this file: {exc}")
                    bundle = None
            if bundle is not None:
                st.success(f"Ingested {upl.name} as **{bundle.source}** data.")
                for w in bundle.warnings:
                    st.warning(w)
                if bundle.metadata:
                    with st.expander("File metadata", expanded=True):
                        st.json(bundle.metadata)
                if bundle.reservoir is not None and len(bundle.reservoir) > 0:
                    st.write("**Extracted reservoir table (preview)**")
                    st.dataframe(bundle.reservoir.head(20), width='stretch')
                    if st.button("Use this data as the working dataset", type="primary"):
                        df = bundle.reservoir.copy()
                        # fill anything the simulator needs but the upload didn't provide
                        defaults = ss["res_params"]
                        for k, v in defaults.items():
                            if k not in df.columns:
                                df[k] = v
                        with st.spinner("Running coupled simulator on uploaded data..."):
                            df = CoupledSimulator().run_dataset(df)
                        ss["reservoir_df"] = df
                        st.success(f"Working dataset updated: {len(df):,} rows.")
                if bundle.images:
                    st.write("**Extracted image(s)**")
                    for i, img in enumerate(bundle.images[:3]):
                        st.image(np.clip(img, 0, 255).astype("uint8"), caption=f"Image {i+1}", width=350)

    st.divider()
    if ss.get("reservoir_df") is not None:
        df = ss["reservoir_df"]
        section_header("Current Working Dataset", "✅")
        kpi_row([("Rows", f"{len(df):,}"), ("Columns", f"{df.shape[1]}"),
                ("Missing values", f"{int(df.isna().sum().sum()):,}")])
        st.dataframe(df.head(50), width='stretch')
    else:
        badge("No working dataset yet", "warn")
