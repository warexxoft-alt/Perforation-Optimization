# PerfoSim Pro v3

**Niger Delta Perforation & Hydraulic Fracturing Optimization Framework**

Real-data upload · Coupled physics simulation · Physics-Informed Neural
Networks · Constrained NSGA-II multi-objective optimization · Rigorous
statistical validation (k-fold/repeated/nested cross-validation, hypothesis
testing, goodness-of-fit diagnostics) · 2D–9D interactive visualization ·
600 DPI publication-quality export.

---

## What's new in v3

v3 is a full statistical, architectural, and UX rebuild of v2. A comprehensive
audit of v2 turned up several critical, previously-invisible bugs — a fake
PINN training loop that never updated its weights, an optimizer decode
constant wrong by 14 orders of magnitude, an optimizer that documented six
design constraints but implemented none of them, a perforation-skin model
that didn't actually implement its own namesake correlation, and two
"reservoir properties" in the synthetic data generator that were
mathematically guaranteed to be constant regardless of input. All of these
are fixed and covered by regression tests. See **[CHANGELOG.md](CHANGELOG.md)**
for the complete, itemized list with before/after evidence.

Headline additions:

| Area | v2 | v3 |
|---|---|---|
| Cross-validation | 1 of 4 models, single split | All 4 models, k-fold **and** repeated k-fold, identical folds for valid paired comparisons |
| Hypothesis testing | None anywhere in the codebase | t-tests, Wilcoxon, ANOVA, Friedman, Holm-corrected pairwise comparisons, correlation significance |
| Goodness-of-fit | None | Shapiro-Wilk, Jarque-Bera, Kolmogorov-Smirnov, Durbin-Watson, Breusch-Pagan, NSE, Willmott's *d* |
| Hyperparameter tuning | Fixed, hard-coded | RandomizedSearchCV (nested CV available); defaults retuned via repeated CV |
| Perforation skin model | Did not implement its cited Karakas-Tariq correlation | Rebuilt & validated against a published worked numerical example |
| PINN training | Loop never updated weights (fabricated loss curve) | Real Adam + analytic backprop, gradient-checked, early stopping |
| NSGA-II constraints | Documented (6) but 0 implemented | All 6 implemented via constrained-domination (Deb 2000) |
| Figures | Interactive Plotly only | + Matplotlib 600 DPI camera-ready export; 2D-9D views of *results* (ML predictions, Pareto front, sensitivity), not just raw data |
| GUI | 1,110-line monolithic `app.py` | Modular `views/` package, native multipage nav, light/dark theme toggle, tooltips throughout |
| Onboarding | None | In-app User Guide & Tutorial (walkthrough, glossary, FAQ, methodology) + a live getting-started checklist |
| Export | None | Excel (formatted, multi-sheet), PDF report, Word report, figure ZIP, browser print view |
| CI/CD | None | GitHub Actions (lint + test matrix + smoke test + Docker build), Dockerfile, 92 automated tests |

## Features

- **Data**: synthetic Niger Delta well generator (calibrated distributions) or upload real data — LAS logs, Excel/CSV, Eclipse GRDECL grids, DICOM/JPEG/PNG core images.
- **Simulation**: Karakas-Tariq perforation skin (validated against a published worked example), Darcy/Vogel IPR, Arps decline, Mohr-Coulomb sand-control, NPV/payback economics.
- **HF Design**: PKN/KGD/Radial fracture geometry with an implicit (Brent's-method) length-time solve, integrated proppant pump schedule, G-function DFIT closure analysis with a genuine two-regime closure signature.
- **ML & Statistical Lab**: OLS baseline (with full t-test/p-value coefficient table), Random-Forest+Gradient-Boosting ensemble, Gaussian-Process (Kriging) surrogate, and a from-scratch Physics-Informed Neural Network — all cross-validated, GOF-diagnosed, and statistically compared.
- **NSGA-II Optimization**: 8 objectives, 6 real constraints, Monte-Carlo hypervolume tracking, memetic (local-search) refinement of the Pareto front.
- **Statistical Validation Console**: a dedicated page for ad-hoc hypothesis tests, plus every GOF/significance report generated elsewhere in the app.
- **2D–9D Visualization**: scatter → surface → parallel-coordinates/SPLOM over the raw dataset, ML predictions, the optimisation Pareto front, or sensitivity results.
- **Export & Print Center**: one-click Excel/PDF/Word/ZIP bundles of every table and 600 DPI figure generated in the session, plus a browser-printable summary view.
- **User Guide & Tutorial**: an in-app getting-started walkthrough, page-by-page reference, searchable glossary, plain-language statistics primer, FAQ, and citations — plus tooltips on every meaningful control throughout the app.

## Architecture

```
perfosim_pro_v3/
├── app.py                     # Entry point: config, theme, sidebar, navigation
├── config.py                  # All constants, defaults, benchmarks, theme tokens
├── modules/
│   ├── simulator.py           # Skin, flow, geomechanics, coupled well simulation
│   ├── hf_design.py           # Fracture geometry, proppant, DFIT
│   ├── optimizer.py           # Constrained NSGA-II + Pareto analysis
│   ├── ml_engine.py           # PINN / Ensemble / Kriging / OLS + validation pipeline
│   ├── stats_engine.py        # GOF tests, hypothesis tests, CIs, model comparison
│   ├── analysis.py            # Performance metrics, benchmark t-tests, tabulation
│   ├── data_gen.py            # Synthetic Niger Delta dataset generator
│   ├── file_ingestion.py      # Multi-format real-data reader
│   ├── plots.py               # Interactive Plotly figures (2D-9D + diagnostics)
│   ├── pub_plots.py           # 600 DPI matplotlib publication figures
│   └── report_export.py       # Excel / PDF / Word / ZIP export
├── views/                     # One Streamlit page per module (render(ctx) function)
├── tests/                     # 92 pytest tests incl. Streamlit AppTest smoke tests
├── .github/workflows/         # CI (lint+test matrix+smoke+docker) & CD (image publish)
├── Dockerfile, Makefile, pyproject.toml, requirements*.txt
```

## Installation & Running

### Local (Python)

```bash
pip install -r requirements.txt
streamlit run app.py
# or:  make run
```

### Docker

```bash
docker build -t perfosim-pro .
docker run --rm -p 8501:8501 perfosim-pro
# or:  make docker-build && make docker-run
```

### Streamlit Community Cloud

Point a new app at this repository's `app.py` — Community Cloud auto-installs
from `requirements.txt` and auto-redeploys on every push to `main`.

## Testing

```bash
pip install -r requirements-dev.txt
pytest tests/ -v                       # or: make test
pytest tests/ --cov --cov-report=html  # or: make test-cov
ruff check .                           # or: make lint
```

The suite includes a numerical gradient check for the PINN's hand-written
backpropagation, regression tests for every v2→v3 bug fix, known-answer
statistical tests, and Streamlit `AppTest` smoke tests for every page in
both its empty and fully-populated states.

## Statistical Methodology

All hypothesis tests run at **α = 0.05** (configurable in `config.ALPHA`).
Cross-validated model comparisons across 3+ models use a Friedman omnibus
test followed by Holm-corrected pairwise paired t-tests (Demšar, 2006 —
the standard approach for comparing learners across resampled folds).
Regression GOF diagnostics follow Shapiro-Wilk/Jarque-Bera (normality),
Durbin-Watson (autocorrelation), and Breusch-Pagan (heteroscedasticity).
See the in-app **Statistical Validation Console** for full method notes and
citations.

## License

Academic / research use. No warranty of fitness for field operational
decisions — engineering results should be reviewed by a qualified
petroleum/completions engineer before use in well design.
