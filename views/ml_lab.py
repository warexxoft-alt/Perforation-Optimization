"""
views/ml_lab.py — ML & Statistical Validation Lab

The direct answer to "improve performance evaluation, model training,
cross validation, level of significance, GOF, t-test, p-value, k-fold,
accuracy, and other performance metrics". Every model is trained, k-fold
(or repeated-k-fold) cross-validated on IDENTICAL folds, GOF-diagnosed, and
statistically compared against every other model with Holm-corrected
paired tests — none of which existed anywhere in v2.
"""
import numpy as np
import pandas as pd
import streamlit as st

from config import ALPHA, CV_FOLDS, CV_REPEATS, tip
from modules import plots, pub_plots
from modules.analysis import PerformanceMetrics
from modules.ml_engine import FEATURE_COLS, MODEL_LABELS, ModelValidator
from modules.stats_engine import GoodnessOfFit, learning_curve_data
from views.common import (
    badge,
    caption,
    info_card,
    kpi_row,
    register_figure,
    section_header,
    significance_pill,
)

TARGETS = ["PI_stb_day_psi", "S_total", "EUR_MMbbl", "NPV_MM"]


def render(ctx=None):
    ss = st.session_state
    st.title("🧠 ML & Statistical Validation Lab")
    caption(f"OLS baseline · Ensemble (RF+GB) · Kriging (GPR) · Physics-Informed Neural Network — "
            f"all cross-validated at k={CV_FOLDS} with significance level α={ALPHA}.")
    st.divider()

    if ss.get("reservoir_df") is None:
        info_card("No dataset yet. Head to <b>📂 Data Hub</b> to generate or upload one first.")
        return
    df = ss["reservoir_df"]

    # ── configuration ──────────────────────────────────────────────────────
    section_header("Configuration", "⚙️")
    c1, c2, c3 = st.columns(3)
    target = c1.selectbox("Prediction target", [t for t in TARGETS if t in df.columns],
                          help="Which outcome the models will learn to predict from the design "
                               "and reservoir parameters.")
    models_sel = c2.multiselect("Models to train", list(MODEL_LABELS.keys()),
                                default=list(MODEL_LABELS.keys()),
                                format_func=lambda k: MODEL_LABELS[k],
                                help="OLS is fastest and most interpretable; PINN is slowest but "
                                     "incorporates the underlying Darcy physics as a training signal.")
    cv_mode = c3.selectbox("Cross-validation", [f"{CV_FOLDS}-fold", f"Repeated {CV_FOLDS}-fold (x{CV_REPEATS})", "Skip CV (fast)"],
                           help=tip("cv_fold") + " 'Repeated' reruns the whole k-fold process "
                                f"{CV_REPEATS} times with different splits for a more stable estimate.")
    tune = st.checkbox("Tune Ensemble hyperparameters (RandomizedSearchCV, slower)", value=False,
                       help="Searches over tree count, depth, and learning rate using an inner "
                            "cross-validation loop, then reports honest performance on data that "
                            "search never saw (nested cross-validation).")

    if st.button("🚀 Train &amp; Validate All Models", type="primary"):
        repeats = CV_REPEATS if "Repeated" in cv_mode else 1
        run_cv = "Skip" not in cv_mode
        with st.spinner("Training models, running cross-validation, and computing statistical diagnostics..."):
            mv = ModelValidator(target=target)
            results = mv.run(df, models=tuple(models_sel), cv_folds=CV_FOLDS,
                             cv_repeats=repeats, tune=tune, run_cv=run_cv)
            ss["ml_results"] = results
            ss["ml_target"] = target
        st.success("Training complete.")

    if ss.get("ml_results") is None:
        return
    results = ss["ml_results"]
    target = ss.get("ml_target", target)
    pm = PerformanceMetrics()

    st.divider()
    section_header("Performance Metrics", "📏", help="R2/RMSE/MAE etc. on a held-out test set the model never trained on.")
    metrics_tbl = pm.ml_metrics_table(results, target)
    st.dataframe(metrics_tbl, width='stretch', hide_index=True)
    caption("Adj. R² penalises model complexity; NSE and Willmott's d are hydrology-standard "
            "goodness-of-fit indices; Accuracy@X% is the fraction of predictions within ±X% of actual.")

    if "comparison" in results:
        st.divider()
        section_header("Cross-Model Statistical Significance", "🔬", help="Are the differences between models' cross-validated scores real, or just noise? See the User Guide's Statistical Methods tab.")
        comp = results["comparison"]
        st.markdown("**Cross-validated score summary (mean ± 95% CI across folds)**")
        st.dataframe(comp["summary"], width='stretch', hide_index=True)

        omni = comp["omnibus"]
        st.markdown(f"**Omnibus test:** {omni.name} — statistic = {omni.statistic:.4f}, "
                    f"p = {omni.p_value:.4g}", unsafe_allow_html=True)
        st.markdown(significance_pill(omni.significant, omni.p_value), unsafe_allow_html=True)
        st.caption(omni.interpretation)

        with st.expander("Pairwise comparisons (Holm-corrected)", expanded=False):
            st.dataframe(comp["pairwise"], width='stretch', hide_index=True)
        st.info(f"**Statistically indistinguishable top tier:** "
                f"{', '.join(MODEL_LABELS.get(k, k) for k in comp['best_model_set'])}")

        fig_cv = plots.plot_cv_comparison({k: v for k, v in results["cv_scores_raw"].items()},
                                          pairwise_df=comp["pairwise"])
        st.plotly_chart(fig_cv, width='stretch')
        pub_fig = pub_plots.fig_cv_comparison(results["cv_scores_raw"], pairwise_df=comp["pairwise"], fig_no=1)
        png = pub_plots.save_fig_bytes(pub_fig)
        register_figure("CV_model_comparison", png)
        st.download_button("⬇️ Download 600 DPI figure (PNG)", png, "cv_comparison_600dpi.png", "image/png")

    st.divider()
    section_header("Predicted vs. Actual &amp; Residual Diagnostics", "📉", help="How closely predictions track reality, and whether the errors (residuals) show any worrying pattern.")
    model_pick = st.selectbox("Model", [k for k in results if k in MODEL_LABELS],
                              format_func=lambda k: MODEL_LABELS[k])
    entry = results[model_pick]
    yte, ypred = entry["yte"], entry["y_pred_te"]

    c1, c2 = st.columns(2)
    with c1:
        st.plotly_chart(plots.plot_predicted_vs_actual(yte, ypred, MODEL_LABELS[model_pick], target,
                        r2_ci=entry.get("r2_ci")), width='stretch')
    with c2:
        st.plotly_chart(plots.plot_residuals_interactive(yte, ypred, MODEL_LABELS[model_pick]),
                        width='stretch')

    section_header("Goodness-of-Fit Diagnostics", "✅", help=tip("gof"))
    st.dataframe(entry["gof"], width='stretch', hide_index=True)
    caption("Shapiro-Wilk / Jarque-Bera / KS test residual normality; Durbin-Watson checks "
            "autocorrelation; Breusch-Pagan checks heteroscedasticity — all at α = 0.05.")

    colA, colB = st.columns(2)
    with colA:
        if st.button("📸 Generate 600 DPI Predicted-vs-Actual figure"):
            fig = pub_plots.fig_predicted_vs_actual(yte, ypred, MODEL_LABELS[model_pick], target,
                    r2=entry["metrics"]["R2"], rmse=entry["metrics"]["RMSE"], fig_no=2)
            png = pub_plots.save_fig_bytes(fig)
            register_figure(f"{model_pick}_pred_vs_actual", png)
            st.image(png, caption="600 DPI export (downscaled for preview)")
            st.download_button("⬇️ Download PNG", png, f"{model_pick}_pred_vs_actual_600dpi.png", "image/png")
    with colB:
        if st.button("📸 Generate 600 DPI residual diagnostics figure"):
            fig = pub_plots.fig_residual_diagnostics(yte, ypred, MODEL_LABELS[model_pick], fig_no=3)
            png = pub_plots.save_fig_bytes(fig)
            register_figure(f"{model_pick}_residuals", png)
            st.image(png, caption="600 DPI export (downscaled for preview)")
            st.download_button("⬇️ Download PNG", png, f"{model_pick}_residuals_600dpi.png", "image/png")

    if model_pick == "pinn" and "history" in entry:
        st.divider()
        section_header("PINN Training Convergence", "📈", help="Loss and validation R2 over training epochs — should decrease/increase and then plateau.")
        st.caption("v3 fix: this is now REAL training history (Adam + analytic backprop) — "
                   "v2's training loop never updated its weights, so its loss curve was fabricated.")
        st.plotly_chart(plots.plot_training_history(entry["history"]), width='stretch')

    if model_pick == "linear":
        st.divider()
        section_header("OLS Coefficient Significance (t-tests)", "🧮", help="Each row tests H0: this feature's true coefficient is zero. p < 0.05 means the feature has a statistically detectable effect.")
        st.dataframe(entry["coefficients"], width='stretch', hide_index=True)
        s = entry["ols_summary"]
        kpi_row([("R²", f"{s['R2']:.4f}"), ("F-stat p-value", f"{s['F p-value']:.4g}"),
                ("Durbin-Watson", f"{s['Durbin-Watson']:.3f}"), ("AIC", f"{s['AIC']:.1f}")])

    if model_pick == "ensemble":
        st.divider()
        section_header("Feature Importance", "🌟", help="Which inputs the ensemble model relies on most to make its predictions.")
        c1, c2 = st.columns(2)
        with c1:
            st.plotly_chart(plots.plot_feature_importance(entry["feature_importance"]),
                            width='stretch')
        with c2:
            if st.button("Compute permutation importance (model-agnostic, slower)"):
                with st.spinner("Permuting features..."):
                    perm = entry["model"].permutation_importance(entry["Xte"], entry["yte"])
                st.plotly_chart(plots.plot_feature_importance(perm, "Permutation Importance"),
                                width='stretch')

    st.divider()
    section_header("Feature–Target Correlation Significance", "📐", help="Pearson correlation of each raw feature with the target, with a p-value — the previously-unused scipy.stats.pearsonr functionality.")
    st.dataframe(results["correlation_table"], width='stretch', hide_index=True)
    caption("Pearson correlation with a two-sided p-value per feature — the direct, previously "
            "unused `scipy.stats.pearsonr` functionality v2 imported but never called.")

    with st.expander("📚 Learning Curve (bias/variance diagnostic — slower)", expanded=False):
        if st.button("Compute learning curve"):
            from modules.ml_engine import MODEL_REGISTRY, build_feature_matrix
            X, y, _ = build_feature_matrix(df, target=target)
            with st.spinner("Computing learning curve..."):
                lc_df = learning_curve_data(MODEL_REGISTRY[model_pick], X, y, cv=5)
            st.plotly_chart(plots.plot_learning_curve(lc_df, MODEL_LABELS[model_pick]), width='stretch')
