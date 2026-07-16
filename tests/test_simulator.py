"""tests/test_simulator.py — physics sanity checks + regression tests for
the v2 -> v3 bug fixes in modules/simulator.py."""
import numpy as np
import pytest

from config import DEFAULT_ECON, DEFAULT_GEO, DEFAULT_PERF, DEFAULT_RES
from modules.simulator import CoupledSimulator, GeoMechModel, SkinCalculator


@pytest.fixture
def base_params():
    return {**DEFAULT_RES, **DEFAULT_PERF, **DEFAULT_GEO, **DEFAULT_ECON}


class TestSkinCalculator:
    def test_matches_published_worked_example(self):
        """v3 BUG FIX regression test — the single most consequential fix in
        the v2->v3 rebuild. v2's kt_skin() used a fabricated coefficient
        table and a non-standard formula (sV was insensitive to
        shots-per-foot, wrong by up to 4 orders of magnitude). This
        reproduces the fully worked example from Bell, Snider & Barker,
        "Perforating" (SPE Monograph 16, Sec. 13-7.2.1): rw=0.328ft,
        2 shots/ft, 0.25in perforation radius, 8in penetration, 180 deg
        phasing, kH/kV=10 -> published answer sH=-0.4, sV=4.3, swb=0.1,
        sp=4.0."""
        sc = SkinCalculator()
        r = sc.kt_skin(spf=2, pen_in=8, hole_in=0.5, rw_ft=0.328, kH=100, kV=10, phasing=180)
        assert r["S_H"] == pytest.approx(-0.4, abs=0.05)
        assert r["S_V"] == pytest.approx(4.3, abs=0.05)
        assert r["S_wb"] == pytest.approx(0.1, abs=0.05)
        assert r["S_KT"] == pytest.approx(4.0, abs=0.05)

    def test_vertical_skin_decreases_with_shots_per_foot(self):
        """Regression test for the specific v2 defect: sV must vary
        meaningfully (and monotonically decrease) with SPF — v2 gave an
        identical sV (to 3 decimals) for SPF=1 through SPF=20."""
        sc = SkinCalculator()
        vals = [sc.kt_skin(spf=s, pen_in=8, hole_in=0.5, rw_ft=0.328, kH=100, kV=10,
                           phasing=180)["S_V"] for s in [0.5, 1, 2, 3, 4]]
        assert all(vals[i] > vals[i+1] for i in range(len(vals) - 1))
        assert vals[0] / vals[-1] > 5   # meaningfully different, not a rounding artifact

    def test_vertical_skin_increases_with_anisotropy(self):
        sc = SkinCalculator()
        low = sc.kt_skin(spf=2, pen_in=8, hole_in=0.5, rw_ft=0.328, kH=10, kV=10, phasing=180)
        high = sc.kt_skin(spf=2, pen_in=8, hole_in=0.5, rw_ft=0.328, kH=100, kV=10, phasing=180)
        assert high["S_V"] > low["S_V"]

    def test_higher_spf_reduces_kt_skin(self, base_params):
        """More shots per foot should generally reduce (or not worsen)
        perforation geometry skin, all else equal — a basic physical
        sanity check on the Karakas-Tariq implementation."""
        sc = SkinCalculator()
        rw = base_params["wellbore_radius_ft"]
        kH, kV = base_params["permeability_mD"], base_params["k_vertical_mD"]
        low = sc.kt_skin(spf=4, pen_in=18, hole_in=0.45, rw_ft=rw, kH=kH, kV=kV, phasing=120)
        high = sc.kt_skin(spf=16, pen_in=18, hole_in=0.45, rw_ft=rw, kH=kH, kV=kV, phasing=120)
        assert high["S_KT"] <= low["S_KT"] + 1e-6

    def test_total_skin_is_finite(self, base_params):
        sc = SkinCalculator()
        result = sc.total(base_params, q_est=500)
        assert np.isfinite(result["S_total"])

    def test_nondarcy_uses_actual_porosity(self):
        """v3 fix: nondarcy_coeff previously ignored the well's actual
        porosity and always used a hardcoded 0.22. Uses parameters far
        enough from the 0.003 physical cap that the porosity dependence
        isn't masked by clipping."""
        sc = SkinCalculator()
        d_low_phi = sc.nondarcy_coeff(k=1e7, h=200, rw=2.0, mu=2000, phi=0.10)
        d_high_phi = sc.nondarcy_coeff(k=1e7, h=200, rw=2.0, mu=2000, phi=0.30)
        assert d_low_phi != d_high_phi
        assert d_low_phi < 0.003 and d_high_phi < 0.003   # sanity: below the cap, so the diff is real


class TestGeoMechModel:
    def test_damage_evolution_is_not_frozen(self):
        """v3 fix: v2's damage-evolution rate constant (A=1e-6) was so small
        that damage_index was mathematically frozen at its initial value for
        every well in the app's realistic parameter space. At least some
        wells drawn from that space must now show real evolution."""
        geo = GeoMechModel()
        rng = np.random.default_rng(0)
        changed = 0
        for _ in range(200):
            PR = rng.uniform(2500, 7000)
            Pwf_op = max(200.0, PR * 0.30)
            DP_act = PR - Pwf_op
            UCS = rng.uniform(10, 40)
            re, rw = rng.uniform(800, 2000), 0.354
            D_d = 0.05
            gp, gp_ref = DP_act / max(re - rw, 1), 5.0
            tau_c = UCS * 145.038 * 0.30
            for _ in range(30):
                D_d = geo.damage_evolution(D_d, 0.15 * DP_act, tau_c, gp, gp_ref)
            if D_d > 0.0501:
                changed += 1
        assert changed > 0, "damage_index never evolves — the v2 regression has resurfaced"

    def test_critical_drawdown_positive(self):
        geo = GeoMechModel()
        dp = geo.critical_drawdown(UCS_MPa=25, sig_h=3500, sig_H=4000, rw=0.354, phi_deg=30)
        assert dp > 0

    def test_casing_vm_stress_positive(self):
        geo = GeoMechModel()
        result = geo.casing_vm_stress(Pwb_psi=1500, sig_h_psi=4200, sig_H_psi=4830, ri=3.09, ro=3.5)
        assert result["VM_psi"] > 0


class TestCoupledSimulator:
    def test_run_well_returns_finite_positive_pi(self, base_params):
        result = CoupledSimulator().run_well(base_params)
        assert np.isfinite(result["PI_stb_day_psi"])
        assert result["PI_stb_day_psi"] > 0

    def test_run_well_converges(self, base_params):
        result = CoupledSimulator().run_well(base_params)
        assert result["rate_skin_iterations"] <= 10

    def test_run_dataset_no_silent_nan_for_valid_input(self):
        from modules.data_gen import NigerDeltaDataGenerator
        df = NigerDeltaDataGenerator(50, seed=1).generate()
        out = CoupledSimulator().run_dataset(df)
        assert out["PI_stb_day_psi"].isna().sum() == 0

    def test_eur_and_npv_are_finite(self, base_params):
        result = CoupledSimulator().run_well(base_params)
        assert np.isfinite(result["EUR_MMbbl"])
        assert np.isfinite(result["NPV_MM"])
