"""tests/test_optimizer.py — NSGA-II correctness + regression tests for the
v2 -> v3 bug fixes (OBJ_SIGN, missing constraints, sand-risk degeneracy)."""
import numpy as np
import pytest

from config import DEFAULT_GEO, DEFAULT_RES, OPT_BOUNDS
from modules.optimizer import NSGA2, ObjectiveFunctions, ParetoAnalyzer


@pytest.fixture
def obj():
    return ObjectiveFunctions({**DEFAULT_RES, **DEFAULT_GEO})


@pytest.fixture
def bounds():
    return list(OPT_BOUNDS.values())


class TestObjectiveFunctions:
    def test_eur_decode_round_trips_to_mmbbl(self, obj):
        """v3 BUG FIX regression test: v2's OBJ_SIGN[1]=-1e-7 made decoded
        EUR wrong by ~1e14. The correct factor (-10) must decode f1 back to
        EUR in MMbbl, matching obj.EUR(x)/1e6 exactly."""
        x = np.array([12, 120, 18, 0.45, -500, 30, 5])
        f = obj.evaluate(x)
        pa = ParetoAnalyzer()
        df = pa.decode(f.reshape(1, -1), x.reshape(1, -1))
        expected_mmbbl = obj.EUR(x) / 1e6
        assert df["EUR (MMbbl)"].iloc[0] == pytest.approx(expected_mmbbl, rel=1e-6)

    def test_sand_risk_depends_on_decision_vector(self, obj):
        """v3 BUG FIX regression test: sand_risk must vary with completion
        design (SPF/pen/hole), not be a fixed reservoir-only constant —
        otherwise the sand-control constraint is either always satisfied or
        never satisfiable, and either way conveys no information to NSGA-II."""
        x_small = np.array([4, 120, 12, 0.32, -500, 30, 5])
        x_large = np.array([20, 120, 30, 0.75, -500, 30, 5])
        assert obj.sand_risk(x_small) != obj.sand_risk(x_large)

    def test_constraint_violation_shape(self, obj):
        x = np.array([12, 120, 18, 0.45, -500, 30, 5])
        g = obj.constraint_violation(x)
        assert g.shape == (6,)
        assert np.all(np.isfinite(g))

    def test_feasible_region_is_nonempty(self, obj, bounds):
        """At least some designs in the box bounds must be feasible, or the
        constrained optimiser has nothing to converge to."""
        rng = np.random.default_rng(0)
        lo = np.array([b[0] for b in bounds]); hi = np.array([b[1] for b in bounds])
        sample = lo + rng.random((500, len(bounds))) * (hi - lo)
        feasible = [np.all(obj.constraint_violation(x) <= 0) for x in sample]
        assert any(feasible), "No feasible designs found in 500 random samples — constraints may be miscalibrated"


class TestNSGA2ConstrainedDomination:
    def test_feasible_beats_infeasible(self):
        opt = NSGA2(lambda x: x, [(0, 1)])
        assert bool(opt._dominates_cv(np.array([5.0]), np.array([1.0]), cvp=0.0, cvq=2.0))
        assert not bool(opt._dominates_cv(np.array([1.0]), np.array([5.0]), cvp=2.0, cvq=0.0))

    def test_smaller_violation_wins_among_infeasible(self):
        opt = NSGA2(lambda x: x, [(0, 1)])
        assert bool(opt._dominates_cv(np.array([5.0]), np.array([1.0]), cvp=1.0, cvq=3.0))

    def test_ordinary_pareto_dominance_when_both_feasible(self):
        opt = NSGA2(lambda x: x, [(0, 1), (0, 1)])
        assert bool(opt._dominates_cv(np.array([1.0, 1.0]), np.array([2.0, 2.0]), cvp=0.0, cvq=0.0))
        assert not bool(opt._dominates_cv(np.array([1.0, 2.0]), np.array([2.0, 1.0]), cvp=0.0, cvq=0.0))


class TestNSGA2Run:
    def test_converges_to_feasible_population(self, obj, bounds):
        opt = NSGA2(obj.evaluate, bounds, constraint_fn=obj.constraint_violation,
                   pop_size=40, n_gen=30, seed=1)
        pop, F, pareto_idx = opt.run()
        assert opt.history["feasible_fraction"][-1] > 0.5

    def test_pareto_front_is_mutually_non_dominated(self, obj, bounds):
        opt = NSGA2(obj.evaluate, bounds, constraint_fn=obj.constraint_violation,
                   pop_size=40, n_gen=20, seed=2)
        pop, F, pareto_idx = opt.run()
        F_pf = F[pareto_idx]
        for i in range(len(F_pf)):
            for j in range(len(F_pf)):
                if i != j:
                    assert not opt._dominates(F_pf[j], F_pf[i]), "Pareto front contains a dominated point"

    def test_refine_never_reduces_feasibility(self, obj, bounds):
        opt = NSGA2(obj.evaluate, bounds, constraint_fn=obj.constraint_violation,
                   pop_size=30, n_gen=20, seed=3)
        pop, F, pareto_idx = opt.run()
        cv_before = np.mean([opt._cv(x) <= 1e-9 for x in pop[pareto_idx]])
        pop2, F2, pareto_idx2 = opt.refine(pop, F, pareto_idx, n_refine=5)
        cv_after = np.mean([opt._cv(x) <= 1e-9 for x in pop2[pareto_idx2]])
        assert cv_after >= cv_before - 1e-9
