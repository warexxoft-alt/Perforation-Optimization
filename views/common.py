"""
views/common.py — PerfoSim Pro v3
════════════════════════════════════════════════════════════════════════════
Shared GUI infrastructure used by every page: session-state initialisation,
the sidebar (data source + reservoir/perf/geo/econ parameter controls), the
light/dark theme system, KPI-card / section-header helpers, and a small
"figure registry" that the Export Center page reads from to bundle every
chart the user has generated during the session into one download.

v2 kept all of this — plus every page's business logic — inline inside a
single 1,110-line `app.py`. Splitting it out is what makes the multipage
navigation in the new `app.py` possible, and makes each page independently
testable (see tests/test_app_smoke.py).
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import streamlit as st

from config import (
    APP_TITLE,
    COLORS,
    DEFAULT_ECON,
    DEFAULT_GEO,
    DEFAULT_HF,
    DEFAULT_PERF,
    DEFAULT_RES,
    DEFAULT_THEME,
    THEMES,
    VERSION,
    tip,
)


# ═══════════════════════════════════════════════════════════════════════════
# SESSION STATE
# ═══════════════════════════════════════════════════════════════════════════
def init_session_state():
    ss = st.session_state
    ss.setdefault("theme", DEFAULT_THEME)
    ss.setdefault("n_samples", 300)
    ss.setdefault("data_source", "Synthetic (Niger Delta generator)")
    ss.setdefault("res_params", {**DEFAULT_RES, **DEFAULT_PERF, **DEFAULT_GEO, **DEFAULT_ECON})
    ss.setdefault("hf_params", dict(DEFAULT_HF))
    ss.setdefault("reservoir_df", None)          # raw + simulated dataframe
    ss.setdefault("uploaded_bundle", None)        # DataBundle from file_ingestion
    ss.setdefault("ml_results", None)             # ModelValidator.run() output
    ss.setdefault("hf_results", None)             # HFDesignEngine.run() output
    ss.setdefault("opt_results", None)            # dict: pop, F, pareto_idx, obj, history, df
    ss.setdefault("figure_registry", {})          # {label: png_bytes} for Export Center
    ss.setdefault("random_seed", 42)


def get_params() -> dict:
    return st.session_state["res_params"]


def register_figure(label: str, png_bytes: bytes):
    """Stash a 600 DPI figure so the Export Center can bundle it later."""
    st.session_state["figure_registry"][label] = png_bytes


# ═══════════════════════════════════════════════════════════════════════════
# THEME  (runtime light/dark CSS-variable toggle)
# ═══════════════════════════════════════════════════════════════════════════
def inject_theme_css():
    t = THEMES[st.session_state.get("theme", DEFAULT_THEME)]
    st.markdown(f"""
    <style>
      html, body, [class*="css"] {{
        font-family: -apple-system, "Segoe UI", Roboto, "Inter", sans-serif;
      }}
      .stApp {{
        background: {t['bg_grad']};
        color: {t['text']};
      }}
      section[data-testid="stSidebar"] {{
        background-color: {t['sidebar_bg']};
        border-right: 1px solid {t['card_border']};
      }}
      [data-testid="stMetric"] {{
        background: {t['card']};
        border: 1px solid {t['card_border']};
        border-radius: 12px;
        padding: 14px 16px 10px 16px;
        box-shadow: {t['shadow']};
        transition: transform 0.15s ease, box-shadow 0.15s ease;
      }}
      [data-testid="stMetric"]:hover {{
        transform: translateY(-2px);
        box-shadow: 0 6px 20px rgba(0,0,0,0.18);
      }}
      [data-testid="stMetricLabel"] {{ color: {t['muted']}; font-weight: 500; }}
      [data-testid="stMetricValue"] {{ font-weight: 700; }}
      div[data-testid="stExpander"] {{
        background: {t['card']};
        border: 1px solid {t['card_border']};
        border-radius: 10px;
      }}
      div[data-testid="stTabs"] button[role="tab"] {{
        font-weight: 600;
      }}
      .psim-card {{
        background: {t['card']};
        border: 1px solid {t['card_border']};
        border-radius: 12px;
        padding: 18px 20px;
        box-shadow: {t['shadow']};
        margin-bottom: 14px;
      }}
      .psim-section-title {{
        font-size: 1.28rem; font-weight: 700; color: {COLORS['primary']};
        border-bottom: 2px solid {COLORS['primary']}33;
        padding-bottom: 6px; margin: 6px 0 14px 0;
        display: flex; align-items: center; gap: 8px;
      }}
      .psim-badge {{
        display:inline-block; padding: 3px 12px; border-radius: 999px;
        font-size: 0.75rem; font-weight: 600; margin-right: 6px;
        letter-spacing: 0.01em;
      }}
      .psim-badge-good {{ background:{COLORS['good']}26; color:{COLORS['good']}; }}
      .psim-badge-warn {{ background:{COLORS['warning']}26; color:{COLORS['warning']}; }}
      .psim-badge-bad  {{ background:{COLORS['danger']}26;  color:{COLORS['danger']}; }}
      .psim-caption {{ color: {t['muted']}; font-size: 0.85rem; line-height: 1.5; }}
      div[data-testid="stDataFrame"] {{ border-radius: 10px; overflow: hidden; }}
      .stButton>button, .stDownloadButton>button {{
        border-radius: 8px; border: 1px solid {COLORS['primary']}55;
        transition: all 0.15s ease;
      }}
      .stButton>button:hover, .stDownloadButton>button:hover {{
        border-color: {COLORS['primary']}; transform: translateY(-1px);
        box-shadow: 0 4px 12px {COLORS['primary']}33;
      }}
      .psim-step {{
        display: flex; align-items: center; gap: 12px; padding: 10px 14px;
        border-radius: 10px; margin-bottom: 6px; border: 1px solid {t['card_border']};
        background: {t['card']};
      }}
      .psim-step-done {{ border-left: 4px solid {COLORS['good']}; }}
      .psim-step-todo {{ border-left: 4px solid {t['muted']}; opacity: 0.75; }}
      .psim-step-num {{
        width: 26px; height: 26px; border-radius: 50%; display:flex;
        align-items:center; justify-content:center; font-weight:700; font-size:0.85rem;
        background: {COLORS['primary']}22; color: {COLORS['primary']}; flex-shrink: 0;
      }}
      .psim-glossary-term {{
        border-bottom: 1px dotted {COLORS['muted']}; cursor: help;
      }}
      #MainMenu {{visibility: hidden;}}
      footer {{visibility: hidden;}}
      @media print {{
        section[data-testid="stSidebar"], header, #MainMenu, footer {{ display:none !important; }}
      }}
    </style>
    """, unsafe_allow_html=True)


def theme_toggle_widget():
    t = st.session_state.get("theme", DEFAULT_THEME)
    label = "🌙 Dark mode" if t == "light" else "☀️ Light mode"
    if st.button(label, width='stretch', key="theme_toggle_btn"):
        st.session_state["theme"] = "dark" if t == "light" else "light"
        st.rerun()


# ═══════════════════════════════════════════════════════════════════════════
# LAYOUT HELPERS
# ═══════════════════════════════════════════════════════════════════════════
def section_header(title: str, icon: str = "", help: str = None):
    tip_html = (f'<span title="{help}" style="cursor help; font-size:0.8em; '
                f'color:{COLORS["muted"]}; margin-left:4px;">&#9432;</span>') if help else ""
    st.markdown(f'<div class="psim-section-title">{icon} {title}{tip_html}</div>', unsafe_allow_html=True)


def kpi_row(items: list):
    """items: list of (label, value, delta_or_None, help_or_None)."""
    cols = st.columns(len(items))
    for c, item in zip(cols, items):
        label, value = item[0], item[1]
        delta = item[2] if len(item) > 2 else None
        help_ = item[3] if len(item) > 3 else None
        with c:
            st.metric(label, value, delta=delta, help=help_)


def badge(text: str, kind: str = "good"):
    st.markdown(f'<span class="psim-badge psim-badge-{kind}">{text}</span>', unsafe_allow_html=True)


def caption(text: str):
    st.markdown(f'<div class="psim-caption">{text}</div>', unsafe_allow_html=True)


def info_card(body_markdown: str):
    st.markdown(f'<div class="psim-card">{body_markdown}</div>', unsafe_allow_html=True)


def significance_pill(is_significant: bool, p_value: float = None) -> str:
    txt = f"p = {p_value:.4g}" if p_value is not None else ""
    if is_significant:
        return f'<span class="psim-badge psim-badge-good">Significant {txt}</span>'
    return f'<span class="psim-badge psim-badge-warn">Not significant {txt}</span>'


def glossary_term(term: str, definition: str) -> str:
    """Inline HTML span with a native hover tooltip — use inside st.markdown
    bodies to gloss a technical term the first time it appears, e.g.
    `st.markdown(f"...the {glossary_term('FDI', tip('FDI'))} shows...")`."""
    return f'<span class="psim-glossary-term" title="{definition}">{term}</span>'


# ═══════════════════════════════════════════════════════════════════════════
# ONBOARDING / GETTING-STARTED PROGRESS TRACKER
# ═══════════════════════════════════════════════════════════════════════════
def workflow_status() -> dict:
    """Returns {step_name: bool_done} reflecting how far through the typical
    workflow (data -> simulate -> train -> optimize -> export) this session
    has progressed — drives the Dashboard's onboarding checklist."""
    ss = st.session_state
    return {
        "Load or generate data":        ss.get("reservoir_df") is not None,
        "Run a single-well simulation": ss.get("perf_result") is not None,
        "Train & validate ML models":   ss.get("ml_results") is not None,
        "Run NSGA-II optimization":     ss.get("opt_results") is not None,
        "Design a hydraulic fracture":  ss.get("hf_results") is not None,
        "Export or print results":      len(ss.get("figure_registry", {})) > 0,
    }


def progress_tracker(steps: dict, page_targets: dict = None):
    """Renders a step-by-step checklist. `steps`: {label: done_bool}.
    `page_targets`: optional {label: url_path} to render a jump-to link."""
    n_done = sum(steps.values())
    st.progress(n_done / max(len(steps), 1),
               text=f"Getting started: {n_done}/{len(steps)} steps complete")
    for label, done in steps.items():
        cls = "psim-step-done" if done else "psim-step-todo"
        icon = "✅" if done else "○"
        st.markdown(f'<div class="psim-step {cls}">'
                    f'<span class="psim-step-num">{icon}</span>'
                    f'<span>{label}</span></div>', unsafe_allow_html=True)


# ═══════════════════════════════════════════════════════════════════════════
# SIDEBAR  (data source + reservoir/perf/geo/econ parameters)
# ═══════════════════════════════════════════════════════════════════════════
def render_sidebar():
    ss = st.session_state
    with st.sidebar:
        st.markdown(f"### 🛢️ PerfoSim Pro v{VERSION}")
        caption("Niger Delta Perf &amp; HF Optimization")
        theme_toggle_widget()
        st.divider()

        st.markdown("**Data Source**")
        ss["data_source"] = st.radio("Data source", ["Synthetic (Niger Delta generator)", "Upload real data"],
                                     index=0 if ss["data_source"].startswith("Synthetic") else 1,
                                     label_visibility="collapsed",
                                     help="Generate a calibrated synthetic Niger Delta dataset instantly, "
                                          "or upload your own LAS/Excel/CSV/GRDECL/image data on the Data Hub page.")
        if ss["data_source"].startswith("Synthetic"):
            ss["n_samples"] = st.slider("Number of synthetic wells", 50, 2000, ss["n_samples"], step=50,
                                        help="More wells give more statistical power for cross-validation and "
                                             "sensitivity analysis, at the cost of slower simulation/training runs.")
            ss["random_seed"] = st.number_input("Random seed", 0, 9999, ss["random_seed"],
                                                help="Fixes the random number generator so the same seed always "
                                                     "reproduces the same synthetic dataset — useful for reproducible results.")

        st.divider()
        with st.expander("🪨 Reservoir Parameters", expanded=False):
            p = ss["res_params"]
            p["permeability_mD"] = st.number_input("Permeability (mD)", 1.0, 5000.0, float(p["permeability_mD"]),
                                                    help=tip("permeability_mD"))
            p["net_pay_ft"] = st.number_input("Net Pay (ft)", 5.0, 300.0, float(p["net_pay_ft"]), help=tip("net_pay_ft"))
            p["porosity"] = st.slider("Porosity", 0.05, 0.35, float(p["porosity"]), help=tip("porosity"))
            p["initial_pressure_psi"] = st.number_input("Initial Pressure (psi)", 1000.0, 10000.0,
                                                          float(p["initial_pressure_psi"]), help=tip("initial_pressure_psi"))
            p["oil_viscosity_cp"] = st.number_input("Oil Viscosity (cp)", 0.2, 20.0, float(p["oil_viscosity_cp"]),
                                                     help=tip("oil_viscosity_cp"))
            p["oil_fvf"] = st.number_input("Oil FVF (rb/stb)", 1.0, 2.5, float(p["oil_fvf"]), help=tip("oil_fvf"))
            p["drainage_radius_ft"] = st.number_input("Drainage Radius (ft)", 200.0, 5000.0,
                                                        float(p["drainage_radius_ft"]), help=tip("drainage_radius_ft"))
            p["damage_skin"] = st.number_input("Formation Damage Skin", 0.0, 15.0, float(p["damage_skin"]),
                                               help=tip("damage_skin"))

        with st.expander("🔩 Perforation Design", expanded=False):
            p["spf"] = st.slider("Shots per Foot (SPF)", 4, 20, int(p["spf"]), help=tip("spf"))
            p["phasing"] = st.select_slider("Phasing (deg)", options=[60, 90, 120, 135, 150, 180],
                                            value=int(p["phasing"]) if int(p["phasing"]) in
                                            [60, 90, 120, 135, 150, 180] else 120, help=tip("phasing"))
            p["pen_depth_in"] = st.slider("Penetration Depth (in)", 6, 36, int(p["pen_depth_in"]), help=tip("pen_depth_in"))
            p["hole_dia_in"] = st.slider("Hole Diameter (in)", 0.25, 1.0, float(p["hole_dia_in"]), help=tip("hole_dia_in"))
            p["underbalance_psi"] = st.slider("Underbalance (psi)", -2000, 500, int(p["underbalance_psi"]),
                                              help=tip("underbalance_psi"))

        with st.expander("⛰️ Geomechanics", expanded=False):
            p["UCS_MPa"] = st.number_input("UCS (MPa)", 5.0, 80.0, float(p["UCS_MPa"]), help=tip("UCS_MPa"))
            p["min_h_stress_psi"] = st.number_input("Min Horizontal Stress (psi)", 1000.0, 10000.0,
                                                      float(p["min_h_stress_psi"]), help=tip("min_h_stress_psi"))
            p["max_h_stress_psi"] = st.number_input("Max Horizontal Stress (psi)", 1000.0, 12000.0,
                                                      float(p["max_h_stress_psi"]), help=tip("max_h_stress_psi"))
            p["friction_angle_deg"] = st.slider("Friction Angle (deg)", 15, 45, int(p["friction_angle_deg"]),
                                                help=tip("friction_angle_deg"))

        with st.expander("💰 Economics", expanded=False):
            for k, label, lo, hi in [("oil_price_usd", "Oil Price ($/bbl)", 20.0, 150.0),
                                      ("opex_per_stb", "OPEX ($/bbl)", 1.0, 30.0),
                                      ("discount_rate", "Discount Rate", 0.02, 0.25)]:
                if k in p:
                    p[k] = st.number_input(label, lo, hi, float(p[k]), help=tip(k))

        ss["res_params"] = p
        st.divider()
        n_done = sum(workflow_status().values())
        st.caption(f"📋 Workflow: {n_done}/6 steps complete — see the Dashboard for your checklist.")
        caption("📖 Full documentation: **User Guide & Tutorial** page.")
        caption(f"PerfoSim Pro v{VERSION} · Statistical validation enabled")
    return ss
