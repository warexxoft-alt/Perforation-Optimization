# Changelog

## v3.1.0 — Tutorial & Tooltips, Further Bug Fixes, Results-Aware Visualization

### 🐛 Additional bug fixes (found via continued empirical audit)

- **The Karakas-Tariq skin model *still* didn't match the published
  equations after the v3.0.0 rewrite was scoped** — a second, deeper pass
  found the coefficient table itself (`config.KT_COEFF`) did not match the
  published Karakas & Tariq (1988) values at all, a required
  phasing-dependent coefficient ("a", used for the effective wellbore
  radius) was missing entirely, and the formula structure differed from
  the source. Most visibly: the vertical pseudo-skin term — described in
  the source as "potentially the largest contributor" to total perforation
  skin — was **completely insensitive to shots-per-foot**, returning an
  identical value to 3 decimal places whether SPF was 1 or 20. Rebuilt
  `SkinCalculator.kt_skin()` and `config.KT_COEFF` from Karakas & Tariq
  (1988, SPE 18247) as reproduced in Bell, Snider & Barker, *Perforating*
  (SPE Monograph 16, Ch. 13), and validated against that chapter's fully
  worked numerical example (sH=-0.4, sV=4.3, swb=0.1, sp=4.0, reproduced
  to within 0.02) and its shots/ft sensitivity table. Since total skin
  feeds every PI/EUR/NPV/ML-label/optimisation-objective in the app, this
  is the single most consequential fix in the project.
- **Cohesion formula didn't match the standard Mohr-Coulomb relation.**
  `modules/data_gen.py` computed cohesion from UCS using
  `UCS*sin(90-phi)/2`, which is not the standard `UCS=2c·cos(φ)/(1-sin(φ))`
  relation (Jaeger, Cook & Zimmerman) and over-estimated cohesion by ~50%
  at a typical 30° friction angle. Corrected; `cohesion_MPa` is reported
  for completeness/QC (the app's `critical_drawdown` model is
  self-contained via UCS and friction angle directly, per Zoback's
  *Reservoir Geomechanics*, so this did not silently propagate into other
  calculations, but the reported value itself is now correct).
- **DFIT closure detection had no real signal to detect.** The synthetic
  post-shut-in pressure decline was a perfectly linear function of the
  G-function by construction (`P = ISIP - m·G`), which has zero curvature
  everywhere — so the inflection-point closure picker was really just
  reacting to floating-point noise. The decline now has a genuine
  two-regime shape (steeper pre-closure, shallower post-closure, per
  Nolte 1979 / Barree, Barree & Craig 2009), and the closure detector is
  verified to recover an injected closure point to within the sampling
  resolution.
- **`n_stages` was silently ignored.** `HFDesignEngine.run()` extracted
  the user-configured stage count but never passed it to
  `pump_schedule()`, which always used its hard-coded default of 6 ramp
  stages regardless. Wired through.
- Removed several more dead code paths found via `ruff`/manual audit
  (`Fcd_opt` computed and discarded, an unused `grid_map` dict in the
  tuning logic, a duplicate top-level `imageio` import) and wired an
  uploaded file's filename stem into `DataBundle.well_name` when the
  format itself doesn't provide one.

### 📖 Tutorial & documentation (new)

- New **User Guide & Tutorial** page: a step-by-step getting-started
  walkthrough (mirrors the onboarding checklist), a page-by-page feature
  reference, a searchable glossary, a plain-language explanation of every
  statistical method used in the app, an FAQ, and full citations.
- New onboarding **progress checklist** on the Dashboard, tracking six
  steps (load data → simulate → train models → optimise → design a
  fracture → export) with live status pulled from the session state.
- **`config.TOOLTIPS`**: a single source-of-truth dictionary of term
  definitions, reused both as native Streamlit `help=` tooltips on every
  meaningful input/section throughout the app *and* as the User Guide's
  glossary content, so a definition never drifts out of sync between the
  two.

### 🎨 UI/UX & theme

- Tooltips added to every sidebar parameter and the key configuration
  widgets/section headers on every page.
- Richer theme CSS: hover states on metric cards and buttons, a styled
  onboarding-step component, an inline glossary-term span helper, refined
  spacing and typography.

### 🌐 Visualization — now covers *results*, not just raw data

- The 2D–9D Visualization page gained a **data-source selector**: the
  same dimensional toolkit (3D scatter/surface up through 9D parallel
  coordinates/SPLOM) now works on the raw dataset, ML model predictions,
  the NSGA-II Pareto front, or Sobol sensitivity results.
- New generic, reusable N-dimensional plotting functions in
  `modules/plots.py` (`plot_nd_bubble`, `plot_nd_animated_bubble`,
  `plot_nd_faceted`, `plot_nd_parallel`, `plot_nd_splom`) that accept any
  dataframe and column list, plus two results-specific views:
  `plot_ml_predictions_multid` (every trained model's predicted-vs-actual
  in one chart: colour=model, size=residual) and
  `plot_pareto_parallel_full` (all 8 NSGA-II objectives at once).
- Fixed a crash where a Sobol interaction-effect column (ST−S1) can dip
  slightly negative from Monte-Carlo sampling noise around its
  theoretical floor of zero, which Plotly's marker-size encoding rejects
  outright; `plot_nd_bubble` now clips defensively for any caller.

### 🧠 Model result improvements

- `EnsembleModel`'s default hyperparameters were retuned
  (`min_samples_leaf_rf` 3→1, `n_estimators_gb` 200→300, `learning_rate_gb`
  0.05→0.07) based on repeated 5-fold cross-validation against the
  v3.0.0-corrected physics pipeline, checked across 3 independent random
  seeds for a consistent (not just lucky-draw) improvement. Full
  `RandomizedSearchCV` tuning (already available via the ML Lab's "Tune"
  option) typically finds a further improvement on top of these new
  defaults for any specific dataset.
- Verified the PINN's physics-loss weight (`lam_phys=0.10`) is already
  near-optimal by cross-validated comparison against 0.05 and 0.20.

## v3.0.0 — Statistical Rigor, Corrected Optimizer, Modular GUI, CI/CD

A comprehensive rebuild following a full code audit of v2.2. Every item
below was verified empirically (unit test, gradient check, or reproduced
crash) rather than assumed from a code read — see `tests/` for the
corresponding regression test of each fix.

### 🐛 Critical bug fixes

- **The core Karakas-Tariq perforation skin model did not implement
  Karakas-Tariq.** This is the single most consequential fix in the
  rebuild: `SkinCalculator.kt_skin()`'s coefficient table
  (`config.KT_COEFF`) did not match the published values at all, the
  phasing-dependent effective-wellbore-radius coefficient ("a" in the
  source table) was missing entirely, and the formula structure itself
  differed from the published equations. The most visible symptom: the
  vertical (converging-flow) pseudo-skin term — which the source
  describes as "potentially the largest contributor" to total perforation
  skin — was **completely insensitive to shots-per-foot**, returning an
  identical value (to 3 decimal places) whether SPF was 1 or 20, and was
  wrong in magnitude by up to 4 orders of magnitude against the published
  sensitivity table. Since this skin value feeds `PI`, `EUR`, `NPV`, every
  ML training label, and every optimisation objective in the app, this bug
  affected essentially every number the tool produced. **Fixed**: rewrote
  `kt_skin()` and `config.KT_COEFF` from the original Karakas & Tariq
  (1988, SPE 18247) coefficients and equations as reproduced in Bell,
  Snider & Barker, *Perforating* (SPE Monograph 16, Ch. 13), and validated
  the result against that chapter's fully worked numerical example
  (rw=0.328 ft, 2 shots/ft, 0.25 in perforation radius, 8 in penetration,
  180° phasing, kH/kV=10 → published sH=-0.4, sV=4.3, swb=0.1, sp=4.0 —
  reproduced to within 0.02 in `tests/test_simulator.py`) and against the
  same source's shots/ft sensitivity table.
- **PINN training was fake.** `PINNModel.train()`'s inner "epoch" loop
  re-evaluated the loss 300 times against the *same* fixed random weights
  without ever updating them — the only real optimisation happened in one
  untracked `L-BFGS-B` call at the end, so every loss curve in the app was
  fabricated. **Fixed**: hand-written analytic backpropagation (verified
  against a central-difference gradient check, max relative error ~2e-9)
  trained with Adam, early stopping, multiple restarts, and an optional
  analytic-gradient L-BFGS-B polish. `tests/test_ml_engine.py`.
- **NSGA-II EUR decoding was wrong by ~14 orders of magnitude.**
  `ParetoAnalyzer.OBJ_SIGN[1] = -1e-7` when the correct inverse of the
  optimiser's internal `f1 = -EUR/1e7` scaling — converted to the
  display unit of MMbbl — is `-10`. Every decoded EUR value in every
  Pareto table/plot was meaningless. `tests/test_optimizer.py`.
- **NSGA-II ran fully unconstrained.** The docstring promised six design
  constraints (SPF bounds, casing VM stress, critical drawdown, phasing
  feasibility, stage count, CDI limit); none were implemented anywhere.
  Implemented all six via Deb (2000) constrained-domination, with
  previously-undefined terms ("phasing feasibility", "CDI limit") given
  concrete, documented, configurable interpretations in `config.CONSTRAINTS`.
- **Sobol sensitivity analysis crashed.** The ML page sliced the feature
  list to `[:6]` regardless of how many features the model was actually
  trained on (8), so `StandardScaler` raised `ValueError: X has 6 features,
  but StandardScaler is expecting 8 features`. Reproduced and fixed;
  `SensitivityAnalyzer` now raises an explicit, actionable error immediately
  if `feat_names` and `n_features` disagree, instead of failing deep inside
  sklearn.
- **Two synthetic "reservoir properties" were silently constant.**
  `oil_viscosity_cp`'s formula exceeded its own clip ceiling (8 cp) for
  100% of reachable API/temperature combinations; `oil_fvf`'s formula used
  the wrong variables entirely (`api/pressure` instead of Standing's
  `GOR·sqrt(gas_sg/oil_sg)+1.25T` bracket) and always fell below its clip
  floor. Both were replaced with the standard, correctly-calibrated
  Beggs-Robinson (1975) and Standing (1947) correlations and now show
  realistic variance (verified in `tests/`).
- **A third variable, `damage_index`, was also frozen.** Its evolution
  rate constant (`A=1e-6`) was small enough that 90 simulated days of
  evolution changed it by ~1e-5 — imperceptible at any reporting precision.
  Recalibrated to `A=1e-2` (empirically scanned against this app's own
  parameter ranges) so it now meaningfully escalates for high-drawdown/
  low-strength wells while staying near its floor for well-designed ones.
- **`sand_risk` didn't depend on completion design at all.** It was a
  function of fixed reservoir/geomechanical properties only, so it was
  either always feasible or never feasible regardless of the candidate
  design being evaluated by the optimiser — providing zero optimisation
  signal. Rewired through a flow-convergence model (Tiab & Donaldson,
  2015) so larger/more-numerous/deeper perforations genuinely reduce risk.
- **The "no-regret" optimisation constraint compared against a best case,
  not a worst case.** It compared optimised PI against PI computed from
  damage skin *alone* (no perforation/crushed-zone/non-Darcy penalty) —
  an idealised floor that real, fully-modelled completions almost always
  fall below, so the constraint failed for ~100% of candidates. Replaced
  with PI from a generic reference completion evaluated through the
  identical skin model.
- **The PKN fracture-length time-march used a one-step-lagged
  approximation.** Leak-off area at time *t* was computed from the
  *previous* step's fracture length, and `scipy.optimize.brentq` was
  imported but never called. Replaced with a true implicit, self-consistent
  solve of the Carter material-balance equation via `brentq` at every time
  step.
- **Proppant mass per pump stage ignored the concentration ramp.**
  Each stage was treated as instantly jumping to and holding its target
  concentration (`constant × volume`); `scipy.integrate.quad` was imported
  but never used. Mass is now the true integral of the linear ramp within
  each stage.

### 📊 Statistical validation (new — `modules/stats_engine.py`)

- Goodness-of-fit: Shapiro-Wilk, Jarque-Bera, Kolmogorov-Smirnov
  (normality); Durbin-Watson (autocorrelation); Breusch-Pagan
  (heteroscedasticity); Nash-Sutcliffe Efficiency and Willmott's *d*
  (continuous-model fit quality).
- Hypothesis tests: one-sample/independent/paired t-tests, Wilcoxon
  signed-rank, one-way ANOVA, Friedman (for comparing 3+ models across
  identical CV folds — Demšar, 2006), Pearson/Spearman correlation with
  p-values.
- Confidence intervals: parametric (t-distribution) and percentile
  bootstrap, for arbitrary statistics and for any regression metric.
- `ModelComparison`: full cross-validated significance report — summary
  table with 95% CI, Friedman omnibus test, Holm-corrected pairwise
  tests, and the "statistically indistinguishable top tier" of models.
- `comprehensive_metrics()`: R², Adjusted R², RMSE, MAE, MAPE, MSE,
  explained variance, max error, NSE, Willmott's *d*, and
  tolerance-based "accuracy within ±10%/±20%" for regression tasks where
  classification accuracy isn't meaningful.
- Generic `kfold_cross_validate` / `nested_cross_validate` /
  `learning_curve_data` helpers that work with any `.fit`/`.predict`
  object — including this project's own model classes.

### 🧠 Machine learning (`modules/ml_engine.py`)

- All model classes (`PINNModel`, `EnsembleModel`, `KrigingModel`, and the
  new `LinearBaselineModel`) now subclass `sklearn.base.BaseEstimator` +
  `RegressorMixin`, so they work directly with `cross_val_score`,
  `learning_curve`, and `RandomizedSearchCV` instead of needing bespoke
  cross-validation code.
- New `LinearBaselineModel`: an OLS reference model exposing the full
  statsmodels inferential table (per-coefficient t-statistics and
  p-values, F-statistic p-value, R²/Adjusted R², AIC/BIC, Durbin-Watson,
  condition number).
- New `ModelValidator`: orchestrates hold-out evaluation, GOF diagnostics,
  k-fold/repeated-k-fold CV on identical folds across models, optional
  RandomizedSearchCV tuning, and the full cross-model statistical
  comparison, in one call.
- `EnsembleModel` gained `permutation_importance()` (Breiman, 2001) to
  cross-check the impurity-based RF/GB importances, which are known to be
  biased toward high-cardinality features.
- Feature/target Pearson correlation with p-values is now computed and
  surfaced (`scipy.stats.pearsonr` was imported in v2 and never called).

### 🎯 Optimization (`modules/optimizer.py`)

- Constrained NSGA-II via Deb (2000) constrained-domination.
- Monte-Carlo hypervolume indicator (normalised per-objective) replaces
  the v2 "product of per-axis means", which was not a valid hypervolume
  computation.
- Memetic local-search refinement (Moscato, 1989): representative Pareto
  solutions are polished via scalarised L-BFGS-B (`scipy.optimize.minimize`,
  imported in v2 and never used) and only accepted if still feasible and
  non-dominated.

### 🎨 Visualization

- New `modules/pub_plots.py`: matplotlib + seaborn, 600 DPI, serif
  (Liberation Serif) font, white background, no gridlines — predicted-vs-
  actual, 4-panel residual diagnostics, CV comparison boxplots with
  significance brackets, learning curves, feature importance, Sobol
  tornado, correlation heatmaps, Pareto fronts, PINN training history.
- New interactive Plotly companions in `modules/plots.py`: CV comparison,
  residual check, correlation heatmap with significance markers, learning
  curve.
- `modules/plots.py` now references `config.COLORS` as a single source of
  truth instead of ~46 repeated hard-coded hex literals.

### 🖥️ GUI / UX

- Replaced the 1,110-line monolithic `app.py` with a `views/` package (one
  module per page) and Streamlit's native multipage navigation
  (`st.navigation`/`st.Page`), grouped into sections with icons.
- Runtime light/dark theme toggle (CSS-variable based, since Streamlit's
  own `[theme]` config is fixed at process start).
- Two new pages: **Statistical Validation Console** (ad-hoc hypothesis
  test runner + aggregated significance reports) and **Export & Print
  Center**.
- Migrated every `use_container_width` call to the current `width=`
  parameter (the old parameter is deprecated).

### 📤 Export & print (new — `modules/report_export.py`)

- Formatted, multi-sheet Excel workbook (conditional formatting, tables,
  auto-sized columns).
- Print-ready PDF report embedding tables and 600 DPI figures.
- Word (.docx) summary report.
- One-click ZIP of every 600 DPI figure generated in the session.
- Browser-printable in-app summary view (sidebar/nav auto-hidden via
  `@media print`).

### 🔧 CI/CD & code quality

- `tests/`: 75 pytest tests — physics sanity checks, the PINN gradient
  check, known-answer statistical tests, sklearn-API compliance checks,
  and `streamlit.testing.v1.AppTest` smoke tests for every page in both
  empty and fully-populated states.
- `.github/workflows/ci.yml`: ruff lint, pytest matrix (Python
  3.10–3.12) with coverage, an app-load smoke test, and a Docker build
  check.
- `.github/workflows/docker-publish.yml`: builds and pushes a versioned
  image to GHCR on push to `main` / version tags.
- `Dockerfile`, `.dockerignore`, `Makefile`, `pyproject.toml`
  (pytest/coverage/ruff config), `requirements-dev.txt`.
- Removed dead imports/variables found via static analysis (`struct`,
  a duplicate `imageio` import, an unused `grid_map`, an unused
  `Fcd_opt`, unused `kv`/`ct` locals, ...) and wired up two that turned
  out to indicate real missing functionality (`n_stages` now actually
  controls the pump schedule's ramp-stage count; an uploaded file's
  filename now populates `DataBundle.well_name` when the format itself
  doesn't provide one).
- `nondarcy_coeff` now uses the well's actual porosity instead of a
  hardcoded 0.22 regardless of input.
- `CoupledSimulator.run_well`'s rate↔non-Darcy-skin coupling now iterates
  to a stated convergence tolerance (capped at 10 iterations) instead of
  a hard-coded two passes; `run_dataset` reports (rather than silently
  swallowing) any per-row simulation failures.

## v2.2.0 and earlier

See prior internal documentation. Not tracked in this changelog.
