"""
views/user_guide.py — User Guide & Tutorial

A single, comprehensive in-app reference: a step-by-step getting-started
walkthrough, a page-by-page feature reference, a searchable glossary (built
from the same `config.TOOLTIPS` dictionary that powers every tooltip in the
app, so definitions never drift out of sync), a plain-language explanation
of the statistical methodology used throughout, an FAQ, and citations.
"""
import pandas as pd
import streamlit as st

from config import ALPHA, CV_FOLDS, CV_REPEATS, TOOLTIPS, VERSION
from views.common import caption, info_card, kpi_row, progress_tracker, section_header, workflow_status


def render(ctx=None):
    st.title("📖 User Guide & Tutorial")
    caption(f"Everything you need to use PerfoSim Pro v{VERSION} — a step-by-step walkthrough, "
            "a page-by-page reference, a glossary, and a plain-language explanation of the statistics.")
    st.divider()

    tabs = st.tabs(["🚀 Getting Started", "🗺️ Page Reference", "📚 Glossary",
                    "🔬 Statistical Methods", "❓ FAQ", "📑 References"])

    # ═══════════════════════════════════════════════════════════════════
    with tabs[0]:
        st.markdown("### Your first five minutes")
        steps = workflow_status()
        progress_tracker(steps)
        st.write("")

        with st.expander("**Step 1 — Load or generate data**", expanded=not steps["Load or generate data"]):
            st.markdown("""
Go to **📂 Data Hub**. You have two options:
- **Synthetic Generator** (fastest way to explore the app): pick a number of
  wells and a random seed, click *Generate dataset*. This uses calibrated
  Niger Delta distributions for permeability, net pay, API gravity, GOR,
  and geomechanical properties (see `modules/data_gen.py` for exact
  ranges and literature sources), then runs the full coupled physics
  simulation on every well automatically.
- **Upload Real Data**: drop a LAS well log, Excel/CSV table, Eclipse
  GRDECL grid, or a core/CT image (DICOM/JPEG/PNG). The app auto-detects
  the format, extracts what it can, and shows you exactly what was found
  (with any warnings) before you commit to using it as the working dataset.

Either way, the result is a **working dataset** — visible and editable on
**🔍 Data Explorer** — that every other page draws from.
""")

        with st.expander("**Step 2 — Explore & simulate a single well**", expanded=False):
            st.markdown("""
**🔍 Data Explorer** gives you filterable tables and full descriptive
statistics (mean, 95% confidence interval, skewness, a normality test) for
every metric.

**⚙️ Perforation Simulation** runs the *coupled* physics model (perforation
skin → productivity index → production forecast → NPV → sand-control check)
for a single design — the one currently set in the sidebar's Reservoir/
Perforation/Geomechanics/Economics panels. Adjust any sidebar parameter and
re-run to see the effect immediately.

**💧 HF Design & DFIT** does the same for a hydraulic-fracturing
treatment: fracture geometry, an integrated proppant pump schedule, and a
G-function DFIT closure analysis.
""")

        with st.expander("**Step 3 — Train & validate machine-learning models**", expanded=False):
            st.markdown(f"""
**🧠 ML & Statistical Lab** trains four models — an interpretable OLS
baseline, a Random-Forest+Gradient-Boosting ensemble, a Gaussian-Process
(Kriging) surrogate, and a Physics-Informed Neural Network — on your working
dataset, to predict any target (Productivity Index, skin, EUR, or NPV) from
the design and reservoir parameters.

Every model is evaluated the same rigorous way: a held-out test set, {CV_FOLDS}-fold
(optionally repeated) cross-validation, goodness-of-fit diagnostics on the
residuals, and a statistical significance comparison against every other
model (with a multiple-comparison correction, since comparing many models
pairwise inflates the chance of a false "this one's better" finding). See
the **🔬 Statistical Methods** tab for what all of that means.
""")

        with st.expander("**Step 4 — Optimize the completion design**", expanded=False):
            st.markdown("""
**🎯 NSGA-II Optimization** searches the full design space (shots/ft,
phasing, penetration, hole diameter, underbalance, cluster spacing, cluster
count) for the best trade-offs across 8 competing objectives — maximise
productivity, recovery, and value; minimise skin, cost, and risk — subject
to 6 real engineering constraints (casing integrity, sand control, shot
clearance, ...). The result is a **Pareto front**: a set of designs where
you cannot improve one objective without worsening another. The page
recommends a best-NPV, a best-PI, and a best-balanced design from that front.
""")

        with st.expander("**Step 5 — Dig into results & run custom statistics**", expanded=False):
            st.markdown("""
**📈 Performance Metrics**, **🏆 Benchmark Comparison**, and **🔬 Results
Analysis** let you drill into any metric's distribution, compare your
results against published industry benchmarks with a real hypothesis test
(not just "is my number bigger"), rank your top designs, and run a Sobol
sensitivity analysis to see which design variable actually drives the
outcome.

**🧪 Statistical Validation Console** is a dedicated page for ad-hoc
hypothesis testing (t-tests, ANOVA, correlation, ...) on any two columns
you choose, plus an aggregated view of every significance test run
elsewhere in the app.

**🌐 2D–9D Visualization** can plot the raw dataset, your ML predictions,
or the optimisation Pareto front, scaling from a simple scatter plot up to
9-dimensional parallel-coordinates / scatter-matrix views.
""")

        with st.expander("**Step 6 — Export & print**", expanded=False):
            st.markdown("""
Every "📸 Generate 600 DPI figure" button anywhere in the app stashes a
camera-ready PNG in a session-wide figure registry. **📤 Export & Print
Center** bundles everything generated so far — tables and figures — into a
formatted Excel workbook, a print-ready PDF report, a Word document, or a
ZIP of every figure, and offers a browser-printable summary view.
""")

    # ═══════════════════════════════════════════════════════════════════
    with tabs[1]:
        st.markdown("### What each page does")
        pages_ref = [
            ("🏠 Dashboard", "Session overview, onboarding checklist, and quick KPIs. Your home base."),
            ("📂 Data Hub", "Generate a synthetic dataset or upload real field data (LAS/Excel/CSV/GRDECL/images)."),
            ("🔍 Data Explorer", "Filterable data table plus rigorous descriptive statistics (CI, normality) for every KPI, and a two-sample comparison tool."),
            ("⚙️ Perforation Simulation", "Single-well coupled simulation: skin breakdown, IPR, decline curve, NPV, sand-control check."),
            ("💧 HF Design & DFIT", "Fracture geometry (PKN/KGD/Radial), integrated proppant pump schedule, DFIT G-function closure analysis."),
            ("🧠 ML & Statistical Lab", "Train/cross-validate 4 models, compare them statistically, inspect residual diagnostics and feature importance."),
            ("🧪 Statistical Validation", "Ad-hoc hypothesis-test runner plus aggregated significance reports from elsewhere in the app."),
            ("🎯 NSGA-II Optimization", "Constrained multi-objective optimisation of the completion design; explore and export the Pareto front."),
            ("🌐 2D–9D Visualization", "Interactive multi-dimensional exploration of the dataset, ML predictions, or the Pareto front."),
            ("📈 Performance Metrics", "Deep dive into any single KPI's distribution, confidence intervals, and normality diagnostics."),
            ("🏆 Benchmark Comparison", "Compares your results against published industry benchmarks with a one-sample t-test, not just a raw delta."),
            ("🔬 Results Analysis", "Composite-score ranking of top designs; Sobol variance-based sensitivity analysis."),
            ("📤 Export & Print Center", "Bundle every table/figure generated this session into Excel, PDF, Word, or a print view."),
        ]
        for name, desc in pages_ref:
            st.markdown(f"**{name}** — {desc}")

    # ═══════════════════════════════════════════════════════════════════
    with tabs[2]:
        st.markdown("### Glossary")
        search = st.text_input("Search terms", placeholder="e.g. skin, p-value, Fcd, cross-validation")
        rows = [{"Term": k, "Definition": v} for k, v in TOOLTIPS.items()]
        df_gloss = pd.DataFrame(rows).sort_values("Term").reset_index(drop=True)
        if search:
            mask = (df_gloss["Term"].str.contains(search, case=False)
                    | df_gloss["Definition"].str.contains(search, case=False))
            df_gloss = df_gloss[mask]
        st.dataframe(df_gloss, width='stretch', hide_index=True, height=420)
        caption("These are the exact same definitions shown in the tooltips throughout the app "
                "(single source of truth: `config.TOOLTIPS`).")

    # ═══════════════════════════════════════════════════════════════════
    with tabs[3]:
        st.markdown(f"""
### How to read the statistics in this app

All hypothesis tests in PerfoSim Pro run at **significance level α = {ALPHA}**
(95% confidence) unless stated otherwise. Here's what the recurring terms mean,
in plain language:

**p-value.** The probability of seeing a result this extreme (or more extreme)
*if there were truly no real effect*. A small p-value (< α) is evidence
*against* "no effect" — conventionally called "statistically significant".
A p-value is **not** the probability that the null hypothesis is true, and it
says nothing about how *large* or *practically important* an effect is —
that's what the effect size / confidence interval is for.

**Confidence interval (CI).** A range built from your sample that would
contain the true value in roughly 95% of repeated samples. Wider intervals
mean more uncertainty, usually from a smaller sample.

**Cross-validation (k-fold).** Instead of trusting one lucky (or unlucky)
train/test split, the data is split into k roughly-equal folds; the model
trains on k-1 folds and is tested on the held-out fold, k times, so every
row gets tested exactly once by a model that never saw it during training.
This app uses **k={CV_FOLDS}** by default, and can repeat the whole process
**{CV_REPEATS}** times with different random splits (*repeated* k-fold) for
a more stable estimate.

**Goodness-of-fit (GOF).** After fitting a model, its residuals (actual
minus predicted) should look like unstructured noise if the model is
well-specified. GOF tests check specific ways that can fail:
- *Shapiro-Wilk / Jarque-Bera / Kolmogorov-Smirnov* — are the residuals
  normally distributed?
- *Durbin-Watson* — are consecutive residuals correlated with each other
  (they shouldn't be)?
- *Breusch-Pagan* — does the residual spread change with the predicted
  value (it shouldn't — that's called heteroscedasticity)?

**Comparing more than two models.** Running many pairwise t-tests inflates
the chance that *at least one* comes back "significant" by pure chance. This
app first runs a **Friedman test** (an omnibus test across all models at
once); only if that's significant does it look at pairwise comparisons, and
even then the p-values get a **Holm correction** that accounts for how many
comparisons were made.

**"Accuracy" for a continuous prediction.** Classification accuracy doesn't
apply to a continuous target like Productivity Index. Instead this app
reports **R²** (variance explained), and an engineering-style **"accuracy
within ±10%/±20%"**: the fraction of predictions that landed within that
tolerance of the true value.
""")

    # ═══════════════════════════════════════════════════════════════════
    with tabs[4]:
        st.markdown("### Frequently asked questions")
        faqs = [
            ("Do I need real data to use this app?",
             "No. The Synthetic Generator on the Data Hub page produces a full, physically "
             "calibrated dataset instantly, which is enough to explore every feature."),
            ("Why do my results change when I change a sidebar parameter?",
             "The sidebar's Reservoir/Perforation/Geomechanics/Economics values are the *current "
             "design point* used by the single-well pages (Perforation Simulation, HF Design) and "
             "by NSGA-II's fixed reservoir context. They don't retroactively change an already-"
             "generated dataset — regenerate the dataset on Data Hub if you want the new values "
             "reflected in every synthetic well."),
            ("Why did cross-validation give a different R² than the plain train/test split?",
             "The held-out test-set R² is a single number from one particular split. Cross-"
             "validation averages performance over several different splits, which is a more "
             "robust estimate and the more defensible number to report."),
            ("A p-value shows as 'None' — why?",
             "Some statistics (e.g. Durbin-Watson, or a correlation on a zero-variance column) "
             "don't have an associated p-value, or there wasn't enough data to compute one "
             "reliably. The app reports 'None'/'undefined' rather than a misleading number."),
            ("Can I trust the NSGA-II Pareto front if 'no feasible designs' shows up?",
             "That message means the box bounds on the design variables don't contain any point "
             "satisfying all 6 constraints — widen the bounds, relax a constraint in "
             "`config.CONSTRAINTS`, or check the reservoir parameters (e.g. a very low UCS makes "
             "sand control very hard to satisfy at any completion design)."),
            ("What's the difference between the interactive charts and the '📸 Generate 600 DPI figure' buttons?",
             "The interactive Plotly charts are for on-screen exploration (zoom, hover, pan). The "
             "600 DPI buttons render a separate, static, publication-quality matplotlib version of "
             "the same result — camera-ready for a thesis, paper, or report — and stash it in the "
             "Export & Print Center."),
        ]
        for q, a in faqs:
            with st.expander(q):
                st.write(a)

    # ═══════════════════════════════════════════════════════════════════
    with tabs[5]:
        st.markdown("### References & further reading")
        refs = [
            "Karakas, M. & Tariq, S. (1991). Semianalytical Productivity Models for Perforated Completions. SPE Production Engineering, 6(1).",
            "Bell, W.T., Sukup, R.A. & Tariq, S.M. (1995). Perforating. SPE Monograph Series, Vol. 16.",
            "Vogel, J.V. (1968). Inflow Performance Relationships for Solution-Gas Drive Wells. JPT, 20(1).",
            "Arps, J.J. (1945). Analysis of Decline Curves. Trans. AIME, 160(1).",
            "Nordgren, R.P. (1972). Propagation of a Vertical Hydraulic Fracture. SPE Journal, 12(4). [PKN model]",
            "Nolte, K.G. (1979). Determination of Fracture Parameters from Fracturing Pressure Decline. SPE 8341.",
            "Barree, R.D., Barree, V.L. & Craig, D.P. (2009). Holistic Fracture Diagnostics. SPE 122208.",
            "Deb, K., Pratap, A., Agarwal, S. & Meyarivan, T. (2002). A Fast and Elitist Multiobjective GA: NSGA-II. IEEE Trans. Evol. Comput., 6(2).",
            "Deb, K. (2000). An Efficient Constraint Handling Method for Genetic Algorithms. Comput. Methods Appl. Mech. Eng., 186.",
            "Raissi, M., Perdikaris, P. & Karniadakis, G.E. (2019). Physics-Informed Neural Networks. J. Comput. Phys., 378.",
            "Sobol, I.M. (2001). Global Sensitivity Indices for Nonlinear Mathematical Models. Math. Comput. Simul., 55(1-3).",
            "Beggs, H.D. & Robinson, J.R. (1975). Estimating the Viscosity of Crude Oil Systems. JPT, 27(9).",
            "Standing, M.B. (1947). A Pressure-Volume-Temperature Correlation for Mixtures of California Oils and Gases. API Drilling and Production Practice.",
            "Demšar, J. (2006). Statistical Comparisons of Classifiers over Multiple Data Sets. JMLR, 7.",
            "Holm, S. (1979). A Simple Sequentially Rejective Multiple Test Procedure. Scand. J. Statist., 6(2).",
        ]
        for r in refs:
            st.markdown(f"- {r}")
        st.caption("See CHANGELOG.md in the project repository for the full list of v2→v3 corrections "
                  "and the specific literature each fix was validated against.")
