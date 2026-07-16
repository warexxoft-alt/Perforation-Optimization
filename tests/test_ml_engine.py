"""tests/test_ml_engine.py — PINN backprop correctness (gradient check),
real-training regression test, sklearn-API compliance, and full
ModelValidator pipeline integration."""
import numpy as np
import pytest
from sklearn.base import clone

from modules.ml_engine import (
    EnsembleModel,
    KrigingModel,
    LinearBaselineModel,
    ModelValidator,
    PINNModel,
    SensitivityAnalyzer,
    build_feature_matrix,
)


class TestPINNGradientCheck:
    def test_analytic_gradient_matches_finite_difference(self, rng):
        """Critical correctness check for the hand-written backprop that
        replaced v2's fake (non-updating) training loop."""
        n, d = 40, 5
        X = rng.normal(0, 1, (n, d))
        y = rng.normal(0, 1, n)
        model = PINNModel(hidden=(6, 5), lam_phys=0.1, random_state=0)
        identity = type("I", (), {"transform": lambda self, x: x, "inverse_transform": lambda self, x: x})()
        model.scaler_X_, model.scaler_y_ = identity, identity
        n_feat = d
        params = model._init_params(n_feat, np.random.default_rng(1))
        Xph = X[:15]
        y_ph_target = rng.normal(0, 1, 15)

        _, _, _, grad = model._loss_grad(params, n_feat, X, y, Xph, y_ph_target)
        eps = 1e-5
        idxs = rng.choice(len(params), size=20, replace=False)
        for i in idxs:
            p_plus, p_minus = params.copy(), params.copy()
            p_plus[i] += eps; p_minus[i] -= eps
            l_plus, *_ = model._loss_grad(p_plus, n_feat, X, y, Xph, y_ph_target)
            l_minus, *_ = model._loss_grad(p_minus, n_feat, X, y, Xph, y_ph_target)
            numerical = (l_plus - l_minus) / (2 * eps)
            rel_err = abs(numerical - grad[i]) / (abs(numerical) + abs(grad[i]) + 1e-8)
            assert rel_err < 1e-3, f"Gradient mismatch at index {i}: analytic={grad[i]}, numeric={numerical}"


class TestPINNRealTraining:
    def test_loss_decreases_over_training(self, rng):
        """v3 BUG FIX regression test: v2's PINN 'epoch' loop never updated
        its weights, so the loss curve was fabricated (flat) until a single
        untracked L-BFGS-B call at the end. Validation RMSE must now
        genuinely decrease from the first to the last logged epoch."""
        n, d = 200, 6
        X = rng.uniform(0, 1, (n, d))
        y = X @ rng.normal(0, 1, d) + 0.05 * rng.normal(0, 1, n)
        model = PINNModel(max_epochs=200, n_restarts=1, random_state=0)
        model.fit(X, y)
        h = model.history_
        assert h["val_rmse"][-1] < h["val_rmse"][0] * 0.9, "PINN training does not appear to reduce error"

    def test_predict_returns_finite(self, rng):
        n, d = 100, 5
        X = rng.uniform(0, 1, (n, d))
        y = X[:, 0] + rng.normal(0, 0.1, n)
        model = PINNModel(max_epochs=100, n_restarts=1).fit(X, y)
        preds = model.predict(X)
        assert np.all(np.isfinite(preds))


class TestSklearnCompliance:
    @pytest.mark.parametrize("cls", [EnsembleModel, KrigingModel, LinearBaselineModel, PINNModel])
    def test_get_params_and_clone(self, cls):
        model = cls()
        params = model.get_params()
        assert isinstance(params, dict)
        cloned = clone(model)
        assert isinstance(cloned, cls)

    def test_cross_val_score_works_natively(self, rng):
        from sklearn.model_selection import cross_val_score
        X = rng.uniform(0, 1, (100, 5))
        y = X[:, 0] * 3 + rng.normal(0, 0.1, 100)
        scores = cross_val_score(EnsembleModel(n_estimators_rf=30, n_estimators_gb=30), X, y, cv=3, scoring="r2")
        assert len(scores) == 3
        assert np.all(np.isfinite(scores))


class TestSensitivityAnalyzer:
    def test_raises_clear_error_on_feature_mismatch(self):
        """v3 fix: v2 crashed with a raw sklearn ValueError on this page
        (`StandardScaler` expecting N features, got fewer) because the
        feature list passed to SensitivityAnalyzer was silently truncated
        elsewhere. The analyzer itself now raises an explicit, actionable
        error immediately at construction time."""
        with pytest.raises(ValueError):
            SensitivityAnalyzer(lambda X: X.sum(axis=1), n_features=8, feat_names=["a", "b"], N=10)

    def test_compute_returns_valid_indices(self, rng):
        def fn(X):
            return X[:, 0] * 2 + X[:, 1]
        sa = SensitivityAnalyzer(fn, n_features=2, feat_names=["a", "b"], N=500)
        df = sa.compute(bounds=[(0, 1), (0, 1)])
        assert set(df["feature"]) == {"a", "b"}
        assert (df["S1"] >= 0).all() and (df["S1"] <= 1).all()


class TestModelValidatorIntegration:
    def test_ensemble_default_hyperparams_beat_sklearn_naive_baseline(self, synthetic_df):
        """v3 improvement: EnsembleModel's defaults were retuned via
        repeated CV against the (v3-corrected) physics pipeline across
        multiple seeds — this locks in that the current defaults stay
        meaningfully better than a trivial mean-prediction baseline, so a
        future edit that silently degrades them gets caught."""
        from sklearn.dummy import DummyRegressor
        from sklearn.model_selection import cross_val_score
        X, y, _ = build_feature_matrix(synthetic_df, target="PI_stb_day_psi")
        ensemble_scores = cross_val_score(EnsembleModel(), X, y, cv=5, scoring="r2")
        dummy_scores = cross_val_score(DummyRegressor(strategy="mean"), X, y, cv=5, scoring="r2")
        assert ensemble_scores.mean() > dummy_scores.mean() + 0.3

    def test_full_pipeline_runs(self, synthetic_df):
        mv = ModelValidator(target="PI_stb_day_psi")
        results = mv.run(synthetic_df, models=("linear", "ensemble"), cv_folds=5, run_cv=True)
        assert "comparison" in results
        for k in ("linear", "ensemble"):
            assert np.isfinite(results[k]["metrics"]["R2"])

    def test_correlation_table_has_no_dead_pearsonr(self, synthetic_df):
        """v3 fix: scipy.stats.pearsonr was imported in v2 and never called."""
        mv = ModelValidator(target="PI_stb_day_psi")
        results = mv.run(synthetic_df, models=("linear",), cv_folds=3, run_cv=False)
        assert len(results["correlation_table"]) > 0
        assert "p-value" in results["correlation_table"].columns
