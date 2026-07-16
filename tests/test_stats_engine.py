"""tests/test_stats_engine.py — known-answer tests for every statistical
routine (t-tests, GOF, CIs, cross-validated model comparison)."""
import numpy as np
import pytest

from modules import stats_engine as se


class TestGoodnessOfFit:
    def test_shapiro_wilk_accepts_normal_data(self, rng):
        data = rng.normal(0, 1, 500)
        result = se.GoodnessOfFit.shapiro_wilk(data)
        assert result.p_value > 0.01  # should NOT reject normality for true normal data

    def test_shapiro_wilk_rejects_uniform_data(self, rng):
        data = rng.uniform(0, 1, 500)
        result = se.GoodnessOfFit.shapiro_wilk(data)
        assert result.p_value < 0.05  # should reject normality for uniform data

    def test_durbin_watson_near_2_for_iid_residuals(self, rng):
        resid = rng.normal(0, 1, 1000)
        result = se.GoodnessOfFit.durbin_watson(resid)
        assert 1.7 < result.statistic < 2.3

    def test_nash_sutcliffe_perfect_fit_is_one(self):
        y = np.array([1.0, 2.0, 3.0, 4.0])
        assert se.GoodnessOfFit.nash_sutcliffe_efficiency(y, y) == pytest.approx(1.0)


class TestComprehensiveMetrics:
    def test_perfect_prediction_gives_r2_one(self):
        y = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        m = se.comprehensive_metrics(y, y)
        assert m["R2"] == pytest.approx(1.0)
        assert m["RMSE"] == pytest.approx(0.0)
        assert m["Accuracy@10%"] == pytest.approx(1.0)

    def test_accuracy_within_tolerance(self):
        y_true = np.array([100.0, 100.0, 100.0, 100.0])
        y_pred = np.array([100.0, 105.0, 115.0, 130.0])   # 0%, 5%, 15%, 30% off
        acc10 = se.accuracy_within_tolerance(y_true, y_pred, 0.10)
        assert acc10 == pytest.approx(0.5)   # only the first two are within 10%


class TestHypothesisTests:
    def test_one_sample_ttest_known_difference(self, rng):
        sample = rng.normal(10, 1, 200)
        result = se.HypothesisTests.one_sample_ttest(sample, popmean=15)
        assert result.significant

    def test_one_sample_ttest_no_difference(self, rng):
        sample = rng.normal(10, 1, 500)
        result = se.HypothesisTests.one_sample_ttest(sample, popmean=10)
        assert not result.significant

    def test_paired_ttest_detects_shift(self, rng):
        a = rng.normal(10, 1, 100)
        b = a + 2.0   # constant shift
        result = se.HypothesisTests.paired_ttest(a, b)
        assert result.significant

    def test_correlation_significant_for_linear_relationship(self, rng):
        x = rng.normal(0, 1, 200)
        y = 3 * x + rng.normal(0, 0.1, 200)
        result = se.HypothesisTests.correlation_test(x, y)
        assert result.significant
        assert result.statistic > 0.9

    def test_correlation_handles_zero_variance_gracefully(self):
        x = np.ones(50)
        y = np.arange(50, dtype=float)
        result = se.HypothesisTests.correlation_test(x, y)
        assert result.statistic is None   # must not raise


class TestConfidenceIntervals:
    def test_t_interval_contains_true_mean_most_of_the_time(self, rng):
        contains = 0
        for _ in range(200):
            sample = rng.normal(5, 2, 30)
            lo, hi = se.ConfidenceIntervals.t_interval(sample, alpha=0.05)
            if lo <= 5 <= hi:
                contains += 1
        assert contains / 200 > 0.85   # nominal 95% coverage, allow MC slack


class TestModelComparison:
    def test_identifies_significantly_better_model(self, rng):
        scores = {
            "Good": rng.normal(0.90, 0.01, 10),
            "Bad": rng.normal(0.60, 0.02, 10),
        }
        mc = se.ModelComparison(scores)
        report = mc.full_report()
        assert report["best_model_set"] == ["Good"]
        assert report["omnibus"].significant

    def test_no_difference_when_scores_identical(self, rng):
        base = rng.normal(0.85, 0.02, 10)
        # small, but not numerically-degenerate, perturbation: a tiny (1e-6)
        # perturbation triggers catastrophic-cancellation instability in
        # ttest_rel's variance estimate, which is a numerical-precision
        # artifact rather than a real signal.
        scores = {"A": base, "B": base + rng.normal(0, 0.002, 10)}
        mc = se.ModelComparison(scores)
        pairwise = mc.pairwise_table()
        assert pairwise.iloc[0]["t-test p-value"] > 0.05   # should not be flagged as significant


class TestCrossValidation:
    def test_kfold_cross_validate_shape(self, rng):
        from sklearn.linear_model import LinearRegression
        X = rng.normal(0, 1, (100, 4))
        y = X @ np.array([1, 2, 3, 4]) + rng.normal(0, 0.1, 100)
        df = se.kfold_cross_validate(LinearRegression, X, y, k=5)
        assert len(df) == 5
        assert df["R2"].mean() > 0.9
