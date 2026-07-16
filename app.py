"""
app.py — PerfoSim Pro v3 entry point
════════════════════════════════════════════════════════════════════════════
v2 crammed page config, sidebar controls, and every page's business logic
into a single 1,110-line file with an if/elif dispatch on a sidebar radio
widget. v3 splits business logic into `views/*.py` (one module per page,
each exposing `render()`) and uses Streamlit's native multipage navigation
(`st.navigation` + `st.Page`) for a real sidebar nav with icons and
sectioned grouping — this file is now just wiring.

Run locally:  streamlit run app.py
"""
import streamlit as st

from config import VERSION
from views import (
                    benchmarks,
                    common,
                    dashboard,
                    data_explorer,
                    data_hub,
                    export_center,
                    hf_design_view,
                    ml_lab,
                    optimization,
                    perforation,
                    performance,
                    results_analysis,
                    statistical_validation,
                    user_guide,
                    visualization,
)

st.set_page_config(page_title=f"PerfoSim Pro v{VERSION}", page_icon="🛢️",
                   layout="wide", initial_sidebar_state="expanded")

common.init_session_state()
common.inject_theme_css()

pages = {
    "Overview": [
        st.Page(dashboard.render, title="Dashboard", icon="🏠", default=True, url_path="dashboard"),
    ],
    "Data": [
        st.Page(data_hub.render, title="Data Hub", icon="📂", url_path="data-hub"),
        st.Page(data_explorer.render, title="Data Explorer", icon="🔍", url_path="data-explorer"),
    ],
    "Simulation": [
        st.Page(perforation.render, title="Perforation Simulation", icon="⚙️", url_path="perforation"),
        st.Page(hf_design_view.render, title="HF Design & DFIT", icon="💧", url_path="hf-design"),
    ],
    "Machine Learning": [
        st.Page(ml_lab.render, title="ML & Statistical Lab", icon="🧠", url_path="ml-lab"),
        st.Page(statistical_validation.render, title="Statistical Validation", icon="🧪", url_path="stats-validation"),
    ],
    "Optimization": [
        st.Page(optimization.render, title="NSGA-II Optimization", icon="🎯", url_path="optimization"),
    ],
    "Analysis": [
        st.Page(visualization.render, title="2D–9D Visualization", icon="🌐", url_path="visualization"),
        st.Page(performance.render, title="Performance Metrics", icon="📈", url_path="performance"),
        st.Page(benchmarks.render, title="Benchmark Comparison", icon="🏆", url_path="benchmarks"),
        st.Page(results_analysis.render, title="Results Analysis", icon="🔬", url_path="results"),
    ],
    "Reporting": [
        st.Page(export_center.render, title="Export & Print Center", icon="📤", url_path="export"),
        st.Page(user_guide.render, title="User Guide & Tutorial", icon="📖", url_path="user-guide"),
    ],
}

pg = st.navigation(pages)
common.render_sidebar()
pg.run()
