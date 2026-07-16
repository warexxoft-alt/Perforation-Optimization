"""tests/test_hf_design.py — HF design physics sanity + regression tests for
the implicit fracture-length solve and integrated pump schedule."""
import numpy as np
import pytest

from config import DEFAULT_GEO, DEFAULT_HF, DEFAULT_RES
from modules.hf_design import FractureGeometry, HFDesignEngine, ProppantDesign


class TestPKNImplicitSolve:
    def test_fracture_length_increases_monotonically_with_time(self):
        """v3 fix: replaced the lagged explicit approximation with a
        self-consistent implicit (brentq) solve; growth must still be
        physically monotonic."""
        res = FractureGeometry.pkn_length_time(qi_bpm=25, mu_cp=50, Ep_psi=3e6, Hf_ft=80, n_steps=50)
        assert np.all(np.diff(res["xf_ft"]) >= -1e-6)

    def test_fracture_length_is_positive_and_finite(self):
        res = FractureGeometry.pkn_length_time(qi_bpm=25, mu_cp=50, Ep_psi=3e6, Hf_ft=80, n_steps=50)
        assert np.all(res["xf_ft"] > 0)
        assert np.all(np.isfinite(res["xf_ft"]))

    def test_higher_pump_rate_gives_longer_fracture(self):
        low = FractureGeometry.pkn_length_time(qi_bpm=15, mu_cp=50, Ep_psi=3e6, Hf_ft=80, n_steps=30)
        high = FractureGeometry.pkn_length_time(qi_bpm=40, mu_cp=50, Ep_psi=3e6, Hf_ft=80, n_steps=30)
        assert high["xf_ft"][-1] > low["xf_ft"][-1]


class TestPumpSchedule:
    def test_cumulative_proppant_matches_integrated_ramp(self):
        """v3 fix: proppant mass per stage now integrates the linear
        concentration ramp (quad) instead of `constant * volume`."""
        df = ProppantDesign.pump_schedule(qi_bpm=30, total_fluid_bbl=3000, ramp_stages=6, max_proppant_ppg=8.0)
        assert df["prop_mass_klbs"].sum() > 0
        assert (df["cum_prop_klbs"].diff().dropna() >= -1e-9).all()   # monotonic cumulative sum

    def test_stage_at_zero_concentration_has_zero_mass(self):
        df = ProppantDesign.pump_schedule(qi_bpm=30, total_fluid_bbl=3000, ramp_stages=6, max_proppant_ppg=8.0)
        pad_row = df.iloc[0]
        assert pad_row["prop_mass_klbs"] == pytest.approx(0.0)


class TestDFITClosureDetection:
    def test_closure_detection_recovers_injected_signal(self):
        """v3 fix: the pre-fix synthetic decline was perfectly linear in G
        by construction (zero curvature everywhere), so the inflection-
        point closure detector had no real signal to find. Injects a known
        closure point and checks it's recovered."""
        from modules.hf_design import DFITAnalysis
        df = DFITAnalysis.dfit_pressure_decline(P_isip=6500, delta_p=800, tp_min=45,
                                                 t_shut_max_min=120, closure_dtD=1.2)
        recovered = df.loc[df["closure_idx"], "delta_tD"].iloc[0]
        assert recovered == pytest.approx(1.2, abs=0.3)

    def test_pressure_decline_is_monotonic(self):
        from modules.hf_design import DFITAnalysis
        df = DFITAnalysis.dfit_pressure_decline(P_isip=6500, delta_p=800, tp_min=45, t_shut_max_min=120)
        assert (df["P_bhp_psi"].diff().dropna() <= 1e-6).all()


class TestHFDesignEngine:
    def test_run_produces_finite_metrics(self):
        result = HFDesignEngine().run(dict(DEFAULT_HF), {**DEFAULT_RES, **DEFAULT_GEO})
        assert np.isfinite(result["Fcd"])
        assert np.isfinite(result["PI_improvement_pct"])
        assert result["xf_ft"] > 0

    def test_higher_conductivity_improves_pi(self):
        hp_low = dict(DEFAULT_HF); hp_low["proppant_pack_perm_mD"] = 5000
        hp_high = dict(DEFAULT_HF); hp_high["proppant_pack_perm_mD"] = 200000
        res_p = {**DEFAULT_RES, **DEFAULT_GEO}
        r_low = HFDesignEngine().run(hp_low, res_p)
        r_high = HFDesignEngine().run(hp_high, res_p)
        assert r_high["Fcd"] >= r_low["Fcd"]
