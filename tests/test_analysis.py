"""tests/test_analysis.py — benchmark t-test comparison + tabulation checks."""
import pandas as pd

from config import BENCHMARKS
from modules.analysis import BenchmarkComparator, PerformanceMetrics, ResultsTabulator


class TestPerformanceMetrics:
    def test_summary_table_includes_ci_and_normality(self, synthetic_df):
        pm = PerformanceMetrics()
        summ = pm.summary_table(synthetic_df)
        assert "Normality (Shapiro p)" in summ.columns
        assert any("CI" in c for c in summ.columns)
        assert len(summ) > 0


class TestBenchmarkComparator:
    def test_compare_distribution_flags_significant_difference(self, rng):
        bc = BenchmarkComparator()
        # Construct a series whose mean is far from every benchmark value
        key = "PI_stb_day_psi"
        bm_vals = [v for k, v in BENCHMARKS[key].items() if k != "source"]
        far_value = max(bm_vals) * 5
        series = pd.Series(rng.normal(far_value, far_value * 0.02, 100))
        result = bc.compare_distribution(series, key)
        assert (result["Significant?"] == "Yes").all()

    def test_compare_distribution_returns_empty_for_unknown_metric(self):
        bc = BenchmarkComparator()
        result = bc.compare_distribution(pd.Series([1, 2, 3]), "not_a_real_metric")
        assert len(result) == 0

    def test_percentile_rank_format(self):
        bc = BenchmarkComparator()
        rank = bc.percentile_rank(999999, BENCHMARKS["PI_stb_day_psi"])
        assert rank.startswith("P")


class TestResultsTabulator:
    def test_best_designs_table_sorted_descending(self, synthetic_df):
        rt = ResultsTabulator()
        top = rt.best_designs_table(synthetic_df, n_top=10)
        assert len(top) <= 10
        assert (top["Composite Score"].diff().dropna() <= 1e-9).all()

    def test_full_results_export_has_readable_column_names(self, synthetic_df):
        rt = ResultsTabulator()
        export = rt.full_results_export(synthetic_df)
        assert "PI (STB/day/psi)" in export.columns
