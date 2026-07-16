"""
modules/data_gen.py
Synthetic Niger Delta reservoir & perforation dataset generator.
Distributions calibrated to: SPE-140437 (SPDC), SPE-162990 (ExxonMobil NG),
SPE-196009 (TotalEnergies Ofon), World Stress Map (West Africa entries).
"""

import numpy as np
import pandas as pd
from scipy.stats import truncnorm


class NigerDeltaDataGenerator:
    def __init__(self, n_samples: int = 500, seed: int = 42):
        self.n = n_samples
        self.rng = np.random.default_rng(seed)
        self.seed = seed

    # ── helpers ──────────────────────────────────────────────────────────────
    def _tnorm(self, mean, std, lo, hi):
        a, b = (lo - mean) / std, (hi - mean) / std
        return truncnorm.rvs(a, b, loc=mean, scale=std, size=self.n,
                             random_state=self.seed)

    def _lognorm(self, mu_log, sig_log, lo, hi):
        vals = np.exp(self.rng.normal(mu_log, sig_log, self.n))
        return np.clip(vals, lo, hi)

    # ── perforation parameters ───────────────────────────────────────────────
    def _perforation(self) -> pd.DataFrame:
        spf_w  = [0.10, 0.20, 0.35, 0.25, 0.10]
        spf    = self.rng.choice([4, 8, 12, 16, 20], self.n, p=spf_w).astype(float)

        ph_w   = [0.15, 0.30, 0.30, 0.15, 0.10]
        phasing = self.rng.choice([60, 90, 120, 135, 180], self.n, p=ph_w).astype(float)

        pen    = np.clip(np.exp(self.rng.normal(np.log(18), 0.35, self.n)), 12, 30)
        hdiam  = self._tnorm(0.45, 0.08, 0.30, 0.75)
        ubpsi  = self._tnorm(-600, 350, -1500, 300)
        csp    = self._tnorm(30, 10, 15, 50)
        ncl    = self.rng.integers(2, 9, self.n).astype(float)
        nst    = self.rng.integers(8, 35, self.n).astype(float)
        orient = self._tnorm(5, 8, 0, 30)

        return pd.DataFrame({"spf": spf, "phasing": phasing,
            "pen_depth_in": pen, "hole_dia_in": hdiam,
            "underbalance_psi": ubpsi, "cluster_spacing_ft": csp,
            "n_clusters": ncl, "n_stages": nst, "orient_offset_deg": orient})

    # ── reservoir properties ─────────────────────────────────────────────────
    def _reservoir(self) -> pd.DataFrame:
        # bi-modal permeability (moderate / high-perm Agbada sands)
        mix = self.rng.uniform(0, 1, self.n)
        lk1 = self.rng.normal(np.log(80),  0.8, self.n)
        lk2 = self.rng.normal(np.log(500), 0.6, self.n)
        k   = np.clip(np.exp(np.where(mix < 0.6, lk1, lk2)), 1, 5000)

        kv_r  = np.clip(np.exp(self.rng.normal(np.log(0.12), 0.4, self.n)), 0.02, 0.5)
        kv    = k * kv_r

        phi_b = 0.18 + 0.08 * (np.log(np.clip(k, 1, 5000)) - np.log(10)) / \
                              (np.log(5000) - np.log(10))
        phi   = np.clip(phi_b + self.rng.normal(0, 0.02, self.n), 0.10, 0.35)

        h     = np.clip(np.exp(self.rng.normal(np.log(25), 0.6, self.n)), 5, 100)

        depth = np.clip(self.rng.normal(9000, 2000, self.n), 5000, 15000)
        P_grad= np.clip(self.rng.normal(0.42, 0.03, self.n), 0.36, 0.50)
        Pi    = np.clip(depth * P_grad, 2500, 8000)
        T_F   = np.clip(80 + depth * 1.5 / 100, 100, 250)

        api   = np.clip(self.rng.normal(35, 6, self.n), 22, 48)
        # Dead-oil viscosity — Beggs & Robinson (1975), Trans. AIME 255.
        # NOTE (v3 fix): the v2 formula `exp(-0.021*api + 2500/(T_F+460) - 0.30)`
        # exceeded its own clip ceiling (8.0 cp) for EVERY combination of API
        # (22-48) and T_F (100-250) reachable here, so `oil_viscosity_cp` was
        # a silent constant (=8.0) for 100% of generated wells — a dead
        # feature that contributed zero information to every downstream
        # model. Verified in tests/test_data_gen.py (variance-nonzero check).
        Z     = 3.0324 - 0.02023 * api
        Y     = 10 ** Z
        X_br  = Y * T_F ** (-1.163)
        mu_o  = np.clip(10 ** X_br - 1, 0.3, 15.0)

        GOR   = np.clip(np.exp(self.rng.normal(np.log(400), 0.4, self.n)), 50, 2000)
        # Oil FVF — Standing (1947) correlation. NOTE (v3 fix): the v2
        # formula used `(api/Pi)**0.5` in place of Standing's actual
        # `[GOR*sqrt(gas_sg/oil_sg) + 1.25*T]^1.2` bracket term, which
        # produced Bo < 1.05 for every reachable input, so `oil_fvf` was
        # likewise a silent constant (=1.05, its clip floor) for 100% of
        # wells. A representative Niger Delta associated-gas gravity of
        # 0.75 is assumed (SPE-162990) since solution-gas gravity is not
        # otherwise tracked by this generator.
        gas_sg = 0.75
        oil_sg = 141.5 / (131.5 + api)
        F_std  = GOR * np.sqrt(gas_sg / oil_sg) + 1.25 * T_F
        Bo     = np.clip(0.9759 + 0.00012 * F_std ** 1.2, 1.02, 2.20)

        S_d   = self._tnorm(3, 2, 0, 15)
        re    = np.clip(self.rng.normal(1320, 200, self.n), 800, 2000)
        ct    = np.clip(np.exp(self.rng.normal(np.log(1.5e-5), 0.3, self.n)), 5e-6, 5e-5)
        pp    = np.clip(depth * np.clip(self.rng.normal(0.45, 0.03, self.n), 0.39, 0.52),
                        2000, 7000)

        return pd.DataFrame({"permeability_mD": k, "k_vertical_mD": kv, "kv_kh": kv_r,
            "porosity": phi, "net_pay_ft": h, "depth_ft": depth,
            "initial_pressure_psi": Pi, "temperature_F": T_F,
            "oil_viscosity_cp": mu_o, "oil_fvf": Bo, "oil_API": api, "GOR_scf_stb": GOR,
            "damage_skin": S_d, "drainage_radius_ft": re,
            "total_compressibility": ct, "pore_pressure_psi": pp})

    # ── geomechanical properties ─────────────────────────────────────────────
    def _geomechanical(self) -> pd.DataFrame:
        depth = np.clip(self.rng.normal(9000, 2000, self.n), 5000, 15000)
        UCS   = np.clip(np.exp(self.rng.normal(np.log(20), 0.6, self.n)), 3, 80)
        E     = np.clip(UCS * self.rng.normal(0.70, 0.15, self.n), 2, 50)
        nu    = self._tnorm(0.28, 0.05, 0.15, 0.40)

        sig_v  = np.clip(self.rng.normal(1.02, 0.04, self.n), 0.95, 1.10) * depth
        sig_h  = np.clip(self.rng.normal(0.75, 0.04, self.n), 0.68, 0.85) * depth
        sig_H  = sig_h * np.clip(self.rng.normal(1.15, 0.10, self.n), 1.0, 1.5)
        pp     = np.clip(self.rng.normal(0.45, 0.03, self.n), 0.39, 0.52) * depth
        biot   = self._tnorm(0.82, 0.08, 0.50, 1.00)
        phi_f  = self._tnorm(28, 5, 15, 42)
        # Cohesion from UCS via the standard Mohr-Coulomb relation
        # UCS = 2c*cos(phi)/(1-sin(phi))  =>  c = UCS*(1-sin(phi))/(2*cos(phi))
        # (Jaeger, Cook & Zimmerman, "Fundamentals of Rock Mechanics", 4th ed., Ch.4).
        # NOTE (v3 fix): v2 used `UCS*sin(90-phi)/2` = `UCS*cos(phi)/2`, which
        # is not this relation and over-estimated cohesion by ~50% at a
        # typical friction angle of 30 deg. `cohesion_MPa` is reported for
        # completeness/QC but is not otherwise consumed by this app's
        # `critical_drawdown` model, which is expressed directly in terms of
        # UCS and friction angle (Zoback, "Reservoir Geomechanics", Ch.6).
        phi_f_rad = np.radians(phi_f)
        coh    = np.clip(UCS * (1 - np.sin(phi_f_rad)) / (2 * np.cos(phi_f_rad)), 0.3, 25)
        BI     = np.clip((E - 2) / 50 * 0.6 + (UCS - 5) / 75 * 0.4, 0, 1)
        az     = (self.rng.normal(315, 30, self.n)) % 360  # NW σH typical Niger Delta

        return pd.DataFrame({"UCS_MPa": UCS, "youngs_modulus_GPa": E,
            "poisson_ratio": nu, "overburden_psi": sig_v,
            "min_h_stress_psi": sig_h, "max_h_stress_psi": sig_H,
            "pore_pressure_geo_psi": pp, "biot_coefficient": biot,
            "friction_angle_deg": phi_f, "cohesion_MPa": coh,
            "brittleness_index": BI, "sigma_H_azimuth": az})

    # ── public API ───────────────────────────────────────────────────────────
    def generate(self) -> pd.DataFrame:
        return pd.concat(
            [self._perforation(), self._reservoir(), self._geomechanical()],
            axis=1
        )
