"""tests/test_plots.py — sanity checks for the generic N-dimensional
results-visualization functions added in the v3 UX enhancement."""
import pandas as pd

from modules import plots


class TestGenericNDPlots:
    def test_bubble_handles_negative_size_column(self, rng):
        """v3 fix: Plotly's marker-size encoding rejects negative values
        outright. Some legitimate quantities (e.g. a Sobol ST-S1 gap that
        can dip slightly negative from Monte-Carlo noise around a
        theoretical floor of zero) need automatic clipping rather than a
        crash."""
        df = pd.DataFrame({"x": rng.uniform(0, 1, 20), "y": rng.uniform(0, 1, 20),
                           "size_col": rng.uniform(-0.5, 0.5, 20)})
        fig = plots.plot_nd_bubble(df, "x", "y", size_col="size_col")
        assert fig is not None

    def test_bubble_basic(self, rng):
        df = pd.DataFrame({"x": rng.uniform(0, 1, 30), "y": rng.uniform(0, 1, 30),
                           "c": rng.uniform(0, 1, 30)})
        fig = plots.plot_nd_bubble(df, "x", "y", color_col="c")
        assert len(fig.data) >= 1

    def test_parallel_coordinates_arbitrary_columns(self, rng):
        df = pd.DataFrame({f"obj{i}": rng.uniform(0, 1, 25) for i in range(8)})
        fig = plots.plot_nd_parallel(df, list(df.columns), color_col="obj0")
        assert fig is not None

    def test_splom_arbitrary_columns(self, rng):
        df = pd.DataFrame({f"col{i}": rng.uniform(0, 1, 25) for i in range(5)})
        fig = plots.plot_nd_splom(df, list(df.columns)[:4])
        assert fig is not None

    def test_faceted_bins_high_cardinality_numeric_facets(self, rng):
        """facet_col/facet_row should auto-bin a high-cardinality numeric
        column rather than creating one facet per unique value."""
        df = pd.DataFrame({"x": rng.uniform(0, 1, 100), "y": rng.uniform(0, 1, 100),
                           "facet_a": rng.uniform(0, 1, 100)})
        fig = plots.plot_nd_faceted(df, "x", "y", facet_col="facet_a")
        assert fig is not None

    def test_animated_bubble_bins_high_cardinality_frame(self, rng):
        df = pd.DataFrame({"x": rng.uniform(0, 1, 100), "y": rng.uniform(0, 1, 100),
                           "s": rng.uniform(1, 10, 100), "c": rng.uniform(0, 1, 100),
                           "frame": rng.uniform(0, 1, 100)})
        fig = plots.plot_nd_animated_bubble(df, "x", "y", "s", "c", "frame")
        assert fig is not None


class TestResultsSpecificPlots:
    def test_ml_predictions_multid(self, synthetic_df):
        from modules.ml_engine import ModelValidator
        results = ModelValidator(target="PI_stb_day_psi").run(
            synthetic_df, models=("linear", "ensemble"), cv_folds=5, run_cv=False)
        fig = plots.plot_ml_predictions_multid(results)
        assert len(fig.data) >= 2   # at least the two models + 1:1 line

    def test_ml_predictions_multid_empty_is_graceful(self):
        fig = plots.plot_ml_predictions_multid({})
        assert fig is not None

    def test_pareto_parallel_full(self, rng):
        cols = ["PI (STB/d/psi)", "EUR (MMbbl)", "NPV (M$)", "Skin S_total"]
        df = pd.DataFrame({c: rng.uniform(0, 1, 20) for c in cols})
        fig = plots.plot_pareto_parallel_full(df, cols)
        assert fig is not None
