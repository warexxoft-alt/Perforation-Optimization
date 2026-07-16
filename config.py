"""
config.py — PerfoSim Pro v3 Global Configuration
Niger Delta Perforation & Hydraulic Fracturing Optimization Framework

v3.0.0 adds: rigorous statistical validation (k-fold/repeated/nested CV,
hypothesis testing, GOF diagnostics), corrected + constrained NSGA-II,
a real (gradient-trained) PINN, publication-grade 600 DPI export figures,
a modular multipage GUI, and CI/CD scaffolding.
v3.1.0 adds: an in-app User Guide & Tutorial with a getting-started
checklist, tooltips throughout every page, a further-corrected
Karakas-Tariq skin model (validated against a published worked example),
2D-9D visualization of ML/optimization *results* (not just raw data),
and retuned Ensemble-model defaults. See CHANGELOG.md.
"""

APP_TITLE   = "PerfoSim Pro v3 — Niger Delta Perf & HF Optimization"
APP_SUB     = "Real-Data Upload · Coupled Simulation · PINNs · NSGA-II · Statistical Validation · 2D–9D Visualization"
VERSION     = "3.1.0"

# ─── Supported Upload Formats ────────────────────────────────────────────────
SUPPORTED_FORMATS = {
    ".csv":    "CSV Tabular Data",
    ".xlsx":   "Excel 2007+ Workbook",
    ".xls":    "Excel Legacy Workbook",
    ".las":    "LAS Well Log (CWLS 2.0/3.0)",
    ".grdecl": "Eclipse / INTERSECT Grid (GRDECL)",
    ".GRDECL": "Eclipse / INTERSECT Grid (GRDECL)",
    ".npy":    "NumPy Binary Array",
    ".npz":    "NumPy Compressed Archive",
    ".dcm":    "DICOM CT Core Scan",
    ".dcom":   "DICOM CT Core Scan",
    ".jpg":    "JPEG Core / Seismic Image",
    ".jpeg":   "JPEG Core / Seismic Image",
    ".png":    "PNG Core / Seismic Image",
    ".img":    "Raw Binary Image",
    ".nic":    "Nexus Reservoir Simulator Input",
    ".txt":    "Whitespace-Delimited Text Table",
    ".dat":    "Whitespace-Delimited Data File",
}

# ─── LAS Mnemonic → Standard Column Mapping ──────────────────────────────────
LAS_MAP = {
    "DEPT": "depth_ft",   "DEPTH": "depth_ft",   "MD": "depth_ft",
    "GR":   "GR_api",     "GRGC": "GR_api",      "SGR": "GR_api",
    "RT":   "Rt_ohmm",    "ILD":  "Rt_ohmm",     "AT90": "Rt_ohmm",
    "NPHI": "NPHI_pu",    "CNLS": "NPHI_pu",     "TNPH": "NPHI_pu",
    "RHOB": "RHOB_gcc",   "ZDEN": "RHOB_gcc",    "RHOZ": "RHOB_gcc",
    "DT":   "DT_usft",    "DTCO": "DT_usft",     "AC":   "DT_usft",
    "CAL":  "CAL_in",     "CALI": "CAL_in",
    "SP":   "SP_mv",
    "PORO": "porosity",   "PHIE": "porosity",    "PHI": "porosity",
    "PERM": "permeability_mD", "PERMX": "permeability_mD",
    "SW":   "Sw_frac",    "SWAT": "Sw_frac",
}

# ─── CSV / Excel Standard Column Aliases ────────────────────────────────────
CSV_MAP = {
    # Reservoir
    "K":         "permeability_mD",   "PERM": "permeability_mD",
    "PHI":       "porosity",          "PORO": "porosity",
    "H":         "net_pay_ft",        "PAY":  "net_pay_ft",
    "PI":        "initial_pressure_psi", "PRES": "initial_pressure_psi",
    "T":         "temperature_F",     "TEMP": "temperature_F",
    "MUO":       "oil_viscosity_cp",  "VISC": "oil_viscosity_cp",
    "API":       "oil_API",
    "GOR":       "GOR_scf_stb",
    # Perforation
    "SPF":       "spf",               "SHOTS": "spf",
    "PHASE":     "phasing",           "PHASING": "phasing",
    "PEN":       "pen_depth_in",      "PENETRATION": "pen_depth_in",
    "DIAM":      "hole_dia_in",       "HOLE_DIA": "hole_dia_in",
    "UB":        "underbalance_psi",
    # Production
    "QO":        "q_oil_stbd",        "QOIL": "q_oil_stbd",
    "QW":        "q_water_stbd",
    "QG":        "q_gas_mscfd",
    "WC":        "water_cut",
    "BHFP":      "BHFP_psi",
    # Geomechanics
    "UCS":       "UCS_MPa",
    "E":         "youngs_modulus_GPa",
    "NU":        "poisson_ratio",
    "SIGMA_H":   "min_h_stress_psi",  "SH": "min_h_stress_psi",
    "SIGMA_V":   "overburden_psi",
}

# ─── Karakas-Tariq (1992) Coefficients — SPE-18247 ───────────────────────────
KT_COEFF = {
    # Karakas & Tariq (1988), SPE 18247 / SPE Prod. Eng. 6(1), 1991, Table 2 —
    # as reproduced in Bell, Snider & Barker, "Perforating", SPE Monograph 16,
    # Ch.13, Table 13-4.
    #
    # v3 FIX: the v2 table below was not the published Karakas-Tariq table at
    # all — every coefficient (a1, a2, b1, b2, c1, c2) was different from the
    # source, the phasing-dependent "a" coefficient used to compute the
    # effective wellbore radius (rw') was missing entirely, and the skin
    # formulas built around it (see SkinCalculator.kt_skin) used a different,
    # non-standard structure. The vertical-skin term (sV) — "potentially the
    # largest contributor to sp" per the source text — was, as a direct
    # result, numerically wrong by up to 4 orders of magnitude AND
    # insensitive to shots-per-foot (verified: SPF=1 through SPF=20 gave an
    # identical sV to 3 decimal places with the old table/formulas).
    #
    # This table + the rewritten kt_skin() are validated against the
    # source's fully worked example (rw=0.328ft, rperf=0.25in, Lperf=8in,
    # 2 SPF, kH/kV=10, phasing=180): expected sH=-0.4, sV=4.3, swb=0.1,
    # sp=4.0 — reproduced exactly in tests/test_simulator.py — and against
    # its Table 13-5 (sV vs. shots/ft at kH/kV=10/5/1).
    0:   {"a": 0.250, "a1": -2.091, "a2": 0.0453, "b1": 5.1313, "b2": 1.8672, "c1": 1.6e-1, "c2": 2.675},
    180: {"a": 0.500, "a1": -2.025, "a2": 0.0943, "b1": 3.0373, "b2": 1.8115, "c1": 2.6e-2, "c2": 4.532},
    120: {"a": 0.648, "a1": -2.018, "a2": 0.0634, "b1": 1.6136, "b2": 1.7770, "c1": 6.6e-3, "c2": 5.320},
    90:  {"a": 0.726, "a1": -1.905, "a2": 0.1038, "b1": 1.5674, "b2": 1.6935, "c1": 1.9e-3, "c2": 6.155},
    60:  {"a": 0.813, "a1": -1.898, "a2": 0.1023, "b1": 1.3654, "b2": 1.6490, "c1": 3.0e-4, "c2": 7.509},
    45:  {"a": 0.860, "a1": -1.788, "a2": 0.2398, "b1": 1.1915, "b2": 1.6392, "c1": 4.6e-5, "c2": 8.791},
}

# ─── Default Simulation Parameters ───────────────────────────────────────────
DEFAULT_RES = {
    "permeability_mD": 150.0,    "k_vertical_mD": 15.0,
    "porosity": 0.22,            "net_pay_ft": 25.0,
    "initial_pressure_psi": 3500.0, "pore_pressure_psi": 3325.0,
    "temperature_F": 180.0,      "oil_viscosity_cp": 2.5,
    "oil_fvf": 1.25,             "drainage_radius_ft": 1320.0,
    "wellbore_radius_ft": 0.354, "total_compressibility": 1.5e-5,
    "damage_skin": 4.0,          "oil_API": 35.0,  "GOR_scf_stb": 400.0,
}
DEFAULT_PERF = {
    "spf": 12.0, "phasing": 60.0, "pen_depth_in": 18.0,
    "hole_dia_in": 0.45, "underbalance_psi": -500.0,
    "cluster_spacing_ft": 30.0, "n_clusters": 4.0, "n_stages": 20.0,
}
DEFAULT_GEO = {
    "UCS_MPa": 25.0, "youngs_modulus_GPa": 15.0, "poisson_ratio": 0.28,
    "min_h_stress_psi": 4200.0, "max_h_stress_psi": 5000.0,
    "overburden_psi": 6000.0,   "biot_coefficient": 0.85,
    "friction_angle_deg": 30.0,
}
DEFAULT_ECON = {
    "oil_price_usd": 70.0, "opex_per_stb": 8.0, "discount_rate": 0.10,
}

# ─── Hydraulic Fracturing Defaults ───────────────────────────────────────────
DEFAULT_HF = {
    # Treatment design
    "pump_rate_bpm": 30.0,          # bbl/min
    "pad_volume_bbl": 500.0,        # barrels of pad fluid
    "total_fluid_bbl": 3000.0,      # total slurry volume
    "fluid_viscosity_cp": 200.0,    # fracturing fluid viscosity (crosslinked gel)
    "fluid_density_ppg": 8.5,       # lb/gal
    # Proppant
    "proppant_mass_klbs": 400.0,    # klb total proppant
    "proppant_density_gcc": 2.65,   # specific gravity (Ottawa sand ≈ 2.65)
    "proppant_diam_in": 0.025,      # grain diameter (20/40 mesh ≈ 0.025 in)
    "proppant_pack_perm_mD": 80000.0, # pack permeability at reference stress
    "proppant_pack_phi": 0.35,      # pack porosity
    # Geometry targets
    "target_xf_ft": 500.0,         # target fracture half-length
    "target_hf_ft": 80.0,          # target fracture height
    "closure_stress_psi": 4200.0,  # minimum horizontal stress (= closure)
    # Leak-off
    "leakoff_coeff_ft_sqrtmin": 0.002,  # CL — total leak-off coefficient
    "spurt_loss_gal_ft2": 0.05,
    # Model selection
    "model": "PKN",                 # PKN / KGD / Radial
}

# ─── Optimisation Bounds ─────────────────────────────────────────────────────
OPT_BOUNDS = {
    "spf":               (4,   20),
    "phasing":           (60, 180),
    "pen_depth_in":      (12,  30),
    "hole_dia_in":       (0.32, 0.75),
    "underbalance_psi": (-1500, 300),
    "cluster_spacing_ft":(15,   50),
    "n_clusters":        (2,     8),
}

# ─── Industry Benchmarks ─────────────────────────────────────────────────────
BENCHMARKS = {
    "skin":           {"optimal":-2.0,"good":0.0,"industry_avg":5.0,"poor":15.0,
                       "source":"SPE-196009; SPE-162990; Baigelan et al.(2023)"},
    "PI_stb_day_psi": {"excellent":8.0,"good":4.0,"industry_avg":2.5,"poor":0.8,
                       "source":"SPE-140437 (SPDC Agbada Fm)"},
    "EUR_MMbbl":      {"top_quartile":18.0,"median":8.0,"p10":2.5,
                       "source":"Rystad Energy Nigeria 2023"},
    "NPV_MM":         {"top_quartile":45.0,"median":22.0,"p10":8.0,
                       "source":"Wood Mackenzie Nigeria 2023"},
    "FDI":            {"target":0.10,"acceptable":0.25,"poor":0.50,
                       "source":"Wen et al.(2025) SPE-222015"},
    "recovery_factor":{"optimized":0.42,"water_flood":0.35,"primary":0.20,
                       "source":"NUPRC 2023"},
    "ML_R2":          {"target":0.95,"acceptable":0.85,
                       "source":"Nemati et al.(2025) SPE-221145"},
    # HF benchmarks
    "Fcd":            {"optimal":1.6,"good":5.0,"poor":0.1,
                       "source":"Cinco-Ley & Samaniego(1981); Prats(1961)"},
    "PI_frac_mult":   {"excellent":4.0,"good":2.5,"typical":1.8,"poor":1.2,
                       "source":"SPE-18247; Economides & Nolte (2000)"},
}

LIT_STUDIES = [
    {"study":"Karakas & Tariq (1992)","journal":"SPE-18247","PI_improvement_pct":22,"skin_reduction_pct":38},
    {"study":"McLeod (1983)",          "journal":"JPT",       "PI_improvement_pct":15,"skin_reduction_pct":60},
    {"study":"Baigelan et al.(2023)", "journal":"SPE-212235","PI_improvement_pct":31,"skin_reduction_pct":72},
    {"study":"Wen et al.(2025)",       "journal":"SPE-222015","PI_improvement_pct":41,"skin_reduction_pct":83},
    {"study":"Nemati et al.(2025)",    "journal":"SPE-221145","PI_improvement_pct":28,"skin_reduction_pct":65},
    {"study":"Zhou et al.(2025)",      "journal":"SPE-221893","PI_improvement_pct":19,"skin_reduction_pct":44},
    {"study":"Tan et al.(2024)",       "journal":"SPE-218906","PI_improvement_pct":35,"skin_reduction_pct":70},
    {"study":"This Study (PerfoSim Pro v3)","journal":"In Progress","PI_improvement_pct":None,"skin_reduction_pct":None},
]

# ─── Accent Palette (theme-invariant; used by both light & dark modes) ───────
COLORS = {
    "primary":"#2E75B6","secondary":"#1B8A8A","accent":"#FF7F0E",
    "good":"#28A745","warning":"#FFC107","danger":"#DC3545",
    "bg":"#0E1117","card":"#1E2130",
    "text":"#FAFAFA","muted":"#9AA1B2","grid":"#2A2E3E",
}
PLOTLY_THEME = "plotly_dark"
CSCALES = {"PI":"Plasma","EUR":"Viridis","NPV":"RdYlGn","skin":"RdYlGn_r",
           "risk":"Reds","hf":"Turbo","pressure":"Hot"}

# ─── Light / Dark Theme Tokens (runtime-toggleable CSS variables) ───────────
# Streamlit's own [theme] block in .streamlit/config.toml only sets the
# *initial* look; it cannot be swapped at runtime. The GUI's theme toggle
# (views/common.py) instead injects one of these token sets as CSS variables,
# so surface/text colours change live while the accent palette (COLORS,
# above) stays constant for brand consistency.
THEMES = {
    "dark": {
        "bg": "#0B0E14", "bg_grad": "linear-gradient(160deg,#0B0E14 0%,#111726 100%)",
        "card": "#1A1F2E", "card_border": "#2A3040", "text": "#F2F4F8",
        "muted": "#9AA3B5", "sidebar_bg": "#0B0E14", "input_bg": "#161C2A",
        "shadow": "0 2px 14px rgba(0,0,0,0.35)", "plotly_template": "plotly_dark",
    },
    "light": {
        "bg": "#F5F7FB", "bg_grad": "linear-gradient(160deg,#F5F7FB 0%,#ECEFF6 100%)",
        "card": "#FFFFFF", "card_border": "#E1E5EE", "text": "#161A22",
        "muted": "#5B6270", "sidebar_bg": "#FFFFFF", "input_bg": "#F3F5FA",
        "shadow": "0 2px 14px rgba(20,30,60,0.08)", "plotly_template": "plotly_white",
    },
}
DEFAULT_THEME = "dark"

# ─── NSGA-II Constraint Configuration ────────────────────────────────────────
# The v2 NSGA-II docstring promised six design constraints that were never
# actually implemented (the optimiser ran fully unconstrained). v3 makes each
# one concrete and enforces it with a standard constrained-dominance rule
# (Deb et al. 2002, "constraint-domination"): feasible designs always beat
# infeasible ones; among two infeasible designs the one with the smaller
# total constraint violation wins; among two feasible designs, ordinary
# Pareto dominance decides. Interpretations made explicit here where the
# original spec was underspecified (shot-clearance, interval length, "CDI")
# are documented inline and are configurable.
CONSTRAINTS = {
    "sand_risk_max":         1.0,     # g1: DP_actual / DP_critical <= 1  (no sand production)
    "casing_yield_psi":      80_000.0,  # g2: API N-80 minimum yield strength
    "casing_od_in":          7.0,        # 7-in nominal casing OD
    "casing_id_in":          6.184,      # 7-in 29 lb/ft casing nominal ID, used for shot clearance
    "min_shot_clearance_in": 0.50,      # g3: min arc length between adjacent perforations
    "max_interval_ft":       400.0,     # g4: N_clusters x cluster_spacing must fit lateral
    "cdi_limit":             0.50,      # g5: Completion/Fluid Divergence Index ceiling (= BENCHMARKS["FDI"]["poor"])
    "min_pi_ratio":          1.0,       # g6: optimized PI must be >= damaged-baseline PI (no-regret)
}

# ─── Statistical Validation Configuration ────────────────────────────────────
ALPHA               = 0.05     # significance level (95% confidence) for every hypothesis test in the app
N_BOOTSTRAP         = 2000     # bootstrap resamples used for percentile confidence intervals
CV_FOLDS            = 5        # k in k-fold cross-validation
CV_REPEATS          = 3        # repeats for RepeatedKFold (reduces fold-split variance in the estimate)
NESTED_CV_INNER     = 3        # inner-loop folds for nested CV hyperparameter search
RANDOM_STATE        = 42
ACCURACY_TOLERANCES = (0.10, 0.20)   # "% of predictions within +-10% / +-20% of actual" engineering accuracy bands
MULTIPLE_COMPARISON_CORRECTION = "holm"   # correction applied when many models are compared pairwise

# ─── Publication-Quality Figure Configuration (matplotlib, 600 DPI) ─────────
PUB_DPI          = 600
PUB_FONT_FAMILY  = "serif"
PUB_FONT_SERIF   = ["Liberation Serif", "Times New Roman", "Nimbus Roman", "DejaVu Serif"]
PUB_FONT_SIZE    = 12
PUB_TITLE_SIZE   = 14
PUB_LINEWIDTH    = 1.5
PUB_FIGSIZE      = (7.2, 5.0)   # inches; ~half a US-letter page at 600 DPI

# ─── ML Hyperparameter Search Grids (used by RandomizedSearchCV) ────────────
RF_PARAM_GRID = {
    "n_estimators":     [200, 300, 400, 500],
    "max_depth":        [None, 8, 14, 20, 30],
    "min_samples_leaf": [1, 2, 3, 5, 8],
    "max_features":     ["sqrt", "log2", 1.0],
}
GB_PARAM_GRID = {
    "n_estimators":  [100, 150, 200, 300],
    "learning_rate": [0.02, 0.05, 0.08, 0.12],
    "max_depth":     [2, 3, 4, 5],
    "subsample":     [0.6, 0.8, 0.9, 1.0],
}
N_SEARCH_ITER = 25   # RandomizedSearchCV draws per model (kept modest for interactive Streamlit use)

# ─── Tooltips / Glossary (single source of truth) ────────────────────────────
# Reused both as Streamlit widget `help=` text throughout views/*.py AND as
# the glossary content on the in-app User Guide page, so definitions never
# drift out of sync between a tooltip and the written documentation.
TOOLTIPS = {
    # reservoir
    "permeability_mD": "Reservoir rock permeability (mD). Higher permeability means oil flows more easily through the rock toward the wellbore.",
    "net_pay_ft": "Net pay thickness (ft) — the vertical interval of reservoir rock capable of producing oil, excluding shale/non-reservoir layers.",
    "porosity": "Fraction of rock volume that is void space (0-1). Affects both storage capacity and the non-Darcy flow calculation.",
    "initial_pressure_psi": "Original reservoir pressure (psi) before any production — the main driver of the pressure drawdown available to produce fluid.",
    "oil_viscosity_cp": "Live oil viscosity (cp) at reservoir conditions. Higher viscosity = more resistance to flow = lower productivity index.",
    "oil_fvf": "Oil Formation Volume Factor (rb/stb) — reservoir barrels of oil (with dissolved gas) needed to yield one stock-tank barrel at surface.",
    "drainage_radius_ft": "Radius (ft) of the reservoir area draining into this well — sets the 're' term in the productivity index equation.",
    "damage_skin": "Pre-existing formation damage skin (dimensionless) from drilling/mud filtrate invasion, before any perforation or stimulation effects are added.",
    "UCS_MPa": "Unconfined Compressive Strength (MPa) of the rock — the main rock-strength input to the sand-control and casing-stress checks.",
    "min_h_stress_psi": "Minimum horizontal in-situ stress (psi) — typically close to the hydraulic fracture closure stress.",
    "max_h_stress_psi": "Maximum horizontal in-situ stress (psi) — used with the minimum stress for wellbore/casing stress-concentration checks.",
    "friction_angle_deg": "Rock internal friction angle (degrees) in the Mohr-Coulomb failure criterion — higher angle means stronger, more stable rock.",
    # perforation design
    "spf": "Shots Per Foot — how many perforations are fired per foot of casing. More shots generally improve connectivity but cost more.",
    "phasing": "Angular spacing between perforations around the casing circumference (degrees). Affects flow convergence and casing integrity.",
    "pen_depth_in": "Perforation tunnel penetration depth (in) past the casing/cement — deeper penetration reaches past near-wellbore damage.",
    "hole_dia_in": "Diameter (in) of each perforation tunnel entry hole.",
    "underbalance_psi": "Pressure differential (psi) at the moment of perforating; negative = underbalanced (wellbore pressure below reservoir pressure), which helps clean up perforation debris.",
    # economics
    "oil_price_usd": "Assumed oil sales price ($/bbl) used in the NPV calculation.",
    "opex_per_stb": "Operating expense ($/bbl produced) subtracted from revenue in the cash-flow model.",
    "discount_rate": "Annual discount rate used to convert future cash flows to present value (NPV).",
    # results / metrics
    "PI": "Productivity Index (STB/day/psi) — production rate per unit of pressure drawdown. Higher is better.",
    "skin": "Total skin factor (dimensionless) — the net flow restriction (positive) or enhancement (negative) caused by the completion, relative to an idealised open hole.",
    "EUR": "Estimated Ultimate Recovery (MMbbl) — total oil expected to be produced over the well's life, from Arps decline-curve integration.",
    "NPV": "Net Present Value (M$) — discounted lifetime cash flow minus capital cost.",
    "FDI": "Fluid/Completion Divergence Index (0-1) — how evenly flow is distributed across perforation clusters. Lower is more uniform (better).",
    "sand_risk": "Ratio of actual to critical drawdown for sand production. Values above 1.0 mean the design risks producing formation sand.",
    "Fcd": "Dimensionless fracture conductivity — fracture permeability x width, divided by reservoir permeability x fracture half-length. Higher generally means a more effective fracture, up to a point of diminishing returns.",
    # statistics
    "alpha": "Significance level (α) — the probability of a false positive the test tolerates. α=0.05 is the conventional 95%-confidence threshold used everywhere in this app.",
    "p_value": "Probability of seeing a result this extreme (or more) if there were truly no effect. Smaller p-values are stronger evidence against the 'no effect' null hypothesis. p < α is conventionally called 'statistically significant'.",
    "r2": "Coefficient of determination — fraction of the variance in the target that the model explains. 1.0 is a perfect fit; 0 is no better than predicting the mean.",
    "cv_fold": "One held-out subset of the data used to test a model that was trained on the remaining data — repeated k times (k-fold cross-validation) so every row gets tested exactly once, giving an honest estimate of how the model performs on data it hasn't seen.",
    "gof": "Goodness-of-Fit — statistical tests (Shapiro-Wilk, Durbin-Watson, Breusch-Pagan, ...) that check whether a model's residuals behave the way a well-specified model's residuals should (normally distributed, no autocorrelation, constant variance).",
    "confidence_interval": "A range that would contain the true value in roughly 95% of repeated samples, given the observed data's variability — wider intervals mean more uncertainty.",
}


def tip(key: str) -> str:
    """Look up a tooltip by key; returns an empty string (not an error) for
    an unrecognised key so a missing tooltip never breaks a page."""
    return TOOLTIPS.get(key, "")
