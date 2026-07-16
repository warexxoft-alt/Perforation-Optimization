"""
views/export_center.py — Export & Print Center.

v2 had no export path at all. This page bundles every table and every
figure generated elsewhere in the session (via `views.common.register_figure`)
into a formatted Excel workbook, a print-ready PDF report, a Word summary,
or a zip of every 600 DPI figure — plus a browser-printable summary view.
"""
from datetime import datetime

import pandas as pd
import streamlit as st

from config import APP_TITLE, VERSION
from modules import report_export as rex
from modules.analysis import BenchmarkComparator, PerformanceMetrics, ResultsTabulator
from views.common import caption, info_card, kpi_row, section_header


def _gather_tables() -> dict:
    ss = st.session_state
    tables = {}
    if ss.get("reservoir_df") is not None:
        df = ss["reservoir_df"]
        pm, rt = PerformanceMetrics(), ResultsTabulator()
        tables["Summary Statistics"] = pm.summary_table(df)
        tables["Top Designs"] = rt.best_designs_table(df, n_top=20)
        tables["Full Dataset"] = rt.full_results_export(df)
    if ss.get("ml_results") is not None:
        pm = PerformanceMetrics()
        tables["ML Metrics"] = pm.ml_metrics_table(ss["ml_results"], ss.get("ml_target", ""))
        if "comparison" in ss["ml_results"]:
            tables["Model Significance"] = ss["ml_results"]["comparison"]["summary"]
            tables["Pairwise Tests"] = ss["ml_results"]["comparison"]["pairwise"]
        tables["Feature Correlation"] = ss["ml_results"].get("correlation_table", pd.DataFrame())
    if ss.get("opt_results") is not None:
        tables["Pareto Front"] = ss["opt_results"]["pareto_df"]
    if ss.get("sobol_df") is not None:
        tables["Sobol Sensitivity"] = ss["sobol_df"]
    return {k: v for k, v in tables.items() if v is not None and len(v) > 0}


def render(ctx=None):
    ss = st.session_state
    st.title("📤 Export & Print Center")
    caption("Bundle every table and every 600 DPI figure generated this session into a single "
            "downloadable Excel workbook, PDF report, or Word document — or use your browser's "
            "print function on the print-ready view below.")
    st.divider()

    tables = _gather_tables()
    figures = ss.get("figure_registry", {})

    kpi_row([("Tables available", len(tables)), ("Figures generated", len(figures))])

    if not tables and not figures:
        info_card("Nothing to export yet — run a simulation, train models, or optimize a design "
                  "on any other page first. Figures generated via any \"📸 Generate 600 DPI figure\" "
                  "button anywhere in the app show up here automatically.")
        return

    st.divider()
    section_header("What will be included", "📦", help="Every table and 600 DPI figure gathered from elsewhere in this session.")
    if tables:
        st.write("**Tables:** " + ", ".join(tables.keys()))
    if figures:
        st.write("**Figures:** " + ", ".join(figures.keys()))

    st.divider()
    section_header("Export", "⬇️", help="Choose a format — each button bundles everything listed above.")
    c1, c2, c3, c4 = st.columns(4)

    with c1:
        if st.button("📗 Build Excel Workbook", width='stretch'):
            path = f"/tmp/perfosim_report_{datetime.now():%Y%m%d_%H%M%S}.xlsx"
            rex.export_excel(tables, path)
            with open(path, "rb") as f:
                st.download_button("⬇️ Download .xlsx", f.read(), "PerfoSim_Pro_Report.xlsx",
                                   "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                                   width='stretch', key="dl_xlsx")

    with c2:
        if st.button("📕 Build PDF Report", width='stretch'):
            sections = [{"heading": name, "table": tbl} for name, tbl in tables.items()]
            for name, png in figures.items():
                sections.append({"heading": name.replace("_", " ").title(), "figure_bytes": png,
                                 "caption": f"Figure: {name}"})
            path = f"/tmp/perfosim_report_{datetime.now():%Y%m%d_%H%M%S}.pdf"
            rex.export_pdf_report(path, APP_TITLE, sections, subtitle=f"PerfoSim Pro v{VERSION} Technical Report")
            with open(path, "rb") as f:
                st.download_button("⬇️ Download .pdf", f.read(), "PerfoSim_Pro_Report.pdf",
                                   "application/pdf", width='stretch', key="dl_pdf")

    with c3:
        if st.button("📘 Build Word Report", width='stretch'):
            sections = [{"heading": name, "table": tbl} for name, tbl in tables.items()]
            for name, png in figures.items():
                sections.append({"heading": name.replace("_", " ").title(), "figure_bytes": png,
                                 "caption": f"Figure: {name}"})
            path = f"/tmp/perfosim_report_{datetime.now():%Y%m%d_%H%M%S}.docx"
            rex.export_word_report(path, APP_TITLE, sections, subtitle=f"PerfoSim Pro v{VERSION} Technical Report")
            with open(path, "rb") as f:
                st.download_button("⬇️ Download .docx", f.read(), "PerfoSim_Pro_Report.docx",
                                   "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
                                   width='stretch', key="dl_docx")

    with c4:
        if figures and st.button("🗜️ Build Figures ZIP", width='stretch'):
            z = rex.figures_to_zip(figures)
            st.download_button("⬇️ Download .zip", z, "PerfoSim_Pro_Figures_600dpi.zip",
                               "application/zip", width='stretch', key="dl_zip")

    st.divider()
    section_header("Print-Ready Summary View", "🖨️", help="Sidebar and navigation chrome are automatically hidden when you print this page.")
    st.caption("Use your browser's print dialog (Ctrl/Cmd+P) on this section — the sidebar and "
               "navigation are automatically hidden in print layout.")
    with st.container(border=True):
        st.markdown(f"## {APP_TITLE}")
        st.markdown(f"*Generated {datetime.now():%Y-%m-%d %H:%M} · v{VERSION}*")
        for name, tbl in tables.items():
            st.markdown(f"### {name}")
            st.dataframe(tbl, width='stretch', hide_index=True)
        for name, png in figures.items():
            st.markdown(f"### {name.replace('_', ' ').title()}")
            st.image(png, width='stretch')

    st.divider()
    if st.button("🗑️ Clear figure registry"):
        st.session_state["figure_registry"] = {}
        st.rerun()
