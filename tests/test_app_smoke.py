"""
tests/test_app_smoke.py — CI smoke tests using streamlit.testing.v1.AppTest.

Exercises every page's render() function end-to-end (including realistic
"already ran the pipeline" states) and asserts zero uncaught exceptions.
NOTE: AppTest.from_function extracts and execs ONLY the target function's
own source in an isolated namespace, so every test body must be fully
self-contained (imports and all) — it cannot call sibling helper functions
defined elsewhere in this module.
"""
from streamlit.testing.v1 import AppTest


def test_app_entrypoint_loads():
    at = AppTest.from_file("app.py", default_timeout=60)
    at.run()
    assert not at.exception


def test_dashboard_empty_state():
    def _run():
        from views import common
        common.init_session_state(); common.inject_theme_css()
        from views import dashboard
        dashboard.render()
    at = AppTest.from_function(_run)
    at.run(timeout=60)
    assert not at.exception


def test_dashboard_with_data():
    def _run():
        import streamlit as st

        from views import common
        common.init_session_state(); common.inject_theme_css()
        from modules.data_gen import NigerDeltaDataGenerator
        from modules.simulator import CoupledSimulator
        df = NigerDeltaDataGenerator(100, seed=1).generate()
        st.session_state["reservoir_df"] = CoupledSimulator().run_dataset(df)
        from views import dashboard
        dashboard.render()
    at = AppTest.from_function(_run)
    at.run(timeout=90)
    assert not at.exception


def test_data_hub():
    def _run():
        import streamlit as st

        from views import common
        common.init_session_state(); common.inject_theme_css()
        from modules.data_gen import NigerDeltaDataGenerator
        from modules.simulator import CoupledSimulator
        df = NigerDeltaDataGenerator(100, seed=1).generate()
        st.session_state["reservoir_df"] = CoupledSimulator().run_dataset(df)
        from views import data_hub
        data_hub.render()
    at = AppTest.from_function(_run)
    at.run(timeout=90)
    assert not at.exception


def test_data_explorer():
    def _run():
        import streamlit as st

        from views import common
        common.init_session_state(); common.inject_theme_css()
        from modules.data_gen import NigerDeltaDataGenerator
        from modules.simulator import CoupledSimulator
        df = NigerDeltaDataGenerator(100, seed=1).generate()
        st.session_state["reservoir_df"] = CoupledSimulator().run_dataset(df)
        from views import data_explorer
        data_explorer.render()
    at = AppTest.from_function(_run)
    at.run(timeout=90)
    assert not at.exception


def test_perforation_after_run():
    def _run():
        import streamlit as st

        from views import common
        common.init_session_state(); common.inject_theme_css()
        from modules.simulator import CoupledSimulator
        st.session_state["perf_result"] = CoupledSimulator().run_well(st.session_state["res_params"])
        from views import perforation
        perforation.render()
    at = AppTest.from_function(_run)
    at.run(timeout=60)
    assert not at.exception


def test_hf_design_after_run():
    def _run():
        import streamlit as st

        from views import common
        common.init_session_state(); common.inject_theme_css()
        from modules.hf_design import HFDesignEngine
        st.session_state["hf_results"] = HFDesignEngine().run(
            st.session_state["hf_params"], st.session_state["res_params"])
        from views import hf_design_view
        hf_design_view.render()
    at = AppTest.from_function(_run)
    at.run(timeout=60)
    assert not at.exception


def test_ml_lab_after_training():
    def _run():
        import streamlit as st

        from views import common
        common.init_session_state(); common.inject_theme_css()
        from modules.data_gen import NigerDeltaDataGenerator
        from modules.ml_engine import ModelValidator
        from modules.simulator import CoupledSimulator
        df = NigerDeltaDataGenerator(120, seed=1).generate()
        df = CoupledSimulator().run_dataset(df)
        st.session_state["reservoir_df"] = df
        st.session_state["ml_results"] = ModelValidator(target="PI_stb_day_psi").run(
            df, models=("linear", "ensemble", "kriging"), cv_folds=5, run_cv=True)
        st.session_state["ml_target"] = "PI_stb_day_psi"
        from views import ml_lab
        ml_lab.render()
    at = AppTest.from_function(_run)
    at.run(timeout=180)
    assert not at.exception


def test_optimization_after_run():
    def _run():
        import streamlit as st

        from views import common
        common.init_session_state(); common.inject_theme_css()
        from config import OPT_BOUNDS
        from modules.optimizer import NSGA2, ObjectiveFunctions, ParetoAnalyzer
        obj = ObjectiveFunctions(st.session_state["res_params"])
        opt = NSGA2(obj.evaluate, list(OPT_BOUNDS.values()), constraint_fn=obj.constraint_violation,
                   pop_size=30, n_gen=15, seed=1)
        pop, F, pareto_idx = opt.run()
        pa = ParetoAnalyzer()
        pareto_df = pa.decode(F[pareto_idx], pop[pareto_idx], constraint_fn=obj.constraint_violation)
        st.session_state["opt_results"] = {"pop": pop, "F": F, "pareto_idx": pareto_idx, "obj": obj,
                                           "history": opt.history, "pareto_df": pareto_df, "analyzer": pa}
        from views import optimization
        optimization.render()
    at = AppTest.from_function(_run)
    at.run(timeout=120)
    assert not at.exception


def test_visualization():
    def _run():
        import streamlit as st

        from views import common
        common.init_session_state(); common.inject_theme_css()
        from modules.data_gen import NigerDeltaDataGenerator
        from modules.simulator import CoupledSimulator
        df = NigerDeltaDataGenerator(100, seed=1).generate()
        st.session_state["reservoir_df"] = CoupledSimulator().run_dataset(df)
        from views import visualization
        visualization.render()
    at = AppTest.from_function(_run)
    at.run(timeout=90)
    assert not at.exception


def test_performance():
    def _run():
        import streamlit as st

        from views import common
        common.init_session_state(); common.inject_theme_css()
        from modules.data_gen import NigerDeltaDataGenerator
        from modules.simulator import CoupledSimulator
        df = NigerDeltaDataGenerator(100, seed=1).generate()
        st.session_state["reservoir_df"] = CoupledSimulator().run_dataset(df)
        from views import performance
        performance.render()
    at = AppTest.from_function(_run)
    at.run(timeout=90)
    assert not at.exception


def test_benchmarks():
    def _run():
        import streamlit as st

        from views import common
        common.init_session_state(); common.inject_theme_css()
        from modules.data_gen import NigerDeltaDataGenerator
        from modules.simulator import CoupledSimulator
        df = NigerDeltaDataGenerator(100, seed=1).generate()
        st.session_state["reservoir_df"] = CoupledSimulator().run_dataset(df)
        from views import benchmarks
        benchmarks.render()
    at = AppTest.from_function(_run)
    at.run(timeout=90)
    assert not at.exception


def test_results_analysis():
    def _run():
        import streamlit as st

        from views import common
        common.init_session_state(); common.inject_theme_css()
        from modules.data_gen import NigerDeltaDataGenerator
        from modules.ml_engine import ModelValidator
        from modules.simulator import CoupledSimulator
        df = NigerDeltaDataGenerator(100, seed=1).generate()
        df = CoupledSimulator().run_dataset(df)
        st.session_state["reservoir_df"] = df
        st.session_state["ml_results"] = ModelValidator(target="PI_stb_day_psi").run(
            df, models=("ensemble",), cv_folds=5, run_cv=False)
        from views import results_analysis
        results_analysis.render()
    at = AppTest.from_function(_run)
    at.run(timeout=120)
    assert not at.exception


def test_statistical_validation():
    def _run():
        import streamlit as st

        from views import common
        common.init_session_state(); common.inject_theme_css()
        from modules.data_gen import NigerDeltaDataGenerator
        from modules.simulator import CoupledSimulator
        df = NigerDeltaDataGenerator(100, seed=1).generate()
        st.session_state["reservoir_df"] = CoupledSimulator().run_dataset(df)
        from views import statistical_validation
        statistical_validation.render()
    at = AppTest.from_function(_run)
    at.run(timeout=90)
    assert not at.exception


def test_user_guide():
    def _run():
        from views import common
        common.init_session_state(); common.inject_theme_css()
        from views import user_guide
        user_guide.render()
    at = AppTest.from_function(_run)
    at.run(timeout=60)
    assert not at.exception


def test_visualization_all_data_sources():
    def _run():
        import streamlit as st

        from views import common
        common.init_session_state(); common.inject_theme_css()
        from config import OPT_BOUNDS
        from modules.data_gen import NigerDeltaDataGenerator
        from modules.ml_engine import ModelValidator, SensitivityAnalyzer
        from modules.optimizer import NSGA2, ObjectiveFunctions, ParetoAnalyzer
        from modules.simulator import CoupledSimulator
        df = NigerDeltaDataGenerator(100, seed=1).generate()
        df = CoupledSimulator().run_dataset(df)
        st.session_state["reservoir_df"] = df
        st.session_state["ml_results"] = ModelValidator(target="PI_stb_day_psi").run(
            df, models=("linear", "ensemble"), cv_folds=5, run_cv=False)
        obj = ObjectiveFunctions(st.session_state["res_params"])
        opt = NSGA2(obj.evaluate, list(OPT_BOUNDS.values()), constraint_fn=obj.constraint_violation,
                   pop_size=20, n_gen=8, seed=1)
        pop, F, pareto_idx = opt.run()
        pa = ParetoAnalyzer()
        pareto_df = pa.decode(F[pareto_idx], pop[pareto_idx], constraint_fn=obj.constraint_violation)
        st.session_state["opt_results"] = {"pop": pop, "F": F, "pareto_idx": pareto_idx, "obj": obj,
                                           "history": opt.history, "pareto_df": pareto_df, "analyzer": pa}
        feat_names = st.session_state["ml_results"]["feat_names"]
        model = st.session_state["ml_results"]["ensemble"]["model"]
        bounds = [(float(df[f].min()), float(df[f].max())) for f in feat_names]
        sa = SensitivityAnalyzer(lambda X: model.predict(X), len(feat_names), feat_names, N=80)
        st.session_state["sobol_df"] = sa.compute(bounds)
        from views import visualization
        visualization.render()
    at = AppTest.from_function(_run)
    at.run(timeout=180)
    assert not at.exception
    for label in ["🧠 ML Predictions", "🎯 Optimization Pareto Front", "🎛️ Sensitivity Results"]:
        at.radio[0].set_value(label).run(timeout=120)
        assert not at.exception, list(at.exception)


def test_export_center():
    def _run():
        import streamlit as st

        from views import common
        common.init_session_state(); common.inject_theme_css()
        from modules.data_gen import NigerDeltaDataGenerator
        from modules.ml_engine import ModelValidator
        from modules.simulator import CoupledSimulator
        df = NigerDeltaDataGenerator(100, seed=1).generate()
        df = CoupledSimulator().run_dataset(df)
        st.session_state["reservoir_df"] = df
        st.session_state["ml_results"] = ModelValidator(target="PI_stb_day_psi").run(
            df, models=("linear", "ensemble"), cv_folds=5, run_cv=True)
        st.session_state["ml_target"] = "PI_stb_day_psi"
        from views import export_center
        export_center.render()
    at = AppTest.from_function(_run)
    at.run(timeout=120)
    assert not at.exception
