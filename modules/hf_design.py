"""
modules/hf_design.py
Hydraulic Fracturing design and analysis engine.

Models:
  · PKN  — Perkins-Kern-Nordgren (1961/1972) height-constrained
  · KGD  — Khristianovic-Geertsma-de Klerk (1955/1969) blade-like
  · Radial — penny-shaped radial fracture
  · Proppant schedule & conductivity optimization
  · G-function DFIT analysis (Nolte 1979)
  · Post-frac productivity prediction (Cinco-Ley 1981, Prats 1961)
"""

import numpy as np
import pandas as pd
from scipy.integrate import quad
from scipy.optimize import brentq


# ══════════════════════════════════════════════════════════════════════════════
# 1.  FRACTURE GEOMETRY MODELS
# ══════════════════════════════════════════════════════════════════════════════
class FractureGeometry:
    """
    PKN / KGD / Radial fracture width and length models.
    All equations in field units unless noted.
    Reference: Economides & Nolte "Reservoir Stimulation" 3rd ed (2000),
               Gidley et al. "Recent Advances in HF" (1989).
    """

    # ── PKN ───────────────────────────────────────────────────────────────────
    @staticmethod
    def pkn_width(qi_bpm: float, mu_cp: float, xf_ft: float,
                  Ep_psi: float, Hf_ft: float) -> dict:
        """
        PKN maximum wellbore width and average width.
        w_w [in] = 0.1524 · (qi·μ·xf / (E'·Hf))^(1/4)
        Average width = π/4 · w_w  (elliptical cross-section)
        """
        if Ep_psi <= 0 or Hf_ft <= 0:
            return {"w_max_in": 0.1, "w_avg_in": 0.08, "model": "PKN"}
        arg   = (qi_bpm * mu_cp * xf_ft) / (Ep_psi * Hf_ft)
        w_max = 0.1524 * max(arg, 1e-12) ** 0.25
        w_avg = (np.pi / 4) * w_max
        p_net = Ep_psi * w_max / (4 * Hf_ft * 12)  # psi
        return {"w_max_in": float(w_max), "w_avg_in": float(w_avg),
                "net_pressure_psi": float(p_net), "model": "PKN"}

    @staticmethod
    def pkn_length_time(qi_bpm, mu_cp, Ep_psi, Hf_ft,
                        CL=0.002, t_min=60, n_steps=100):
        """
        PKN fracture length vs time including fluid loss.
        Uses Carter (1957) material balance:
            V_inj(t) = V_frac(xf) + V_leakoff(xf, t)
        solved IMPLICITLY and self-consistently for xf(t) at every time step
        via Brent's method (v3 fix — see note below).
        Returns arrays: t[min], xf[ft], ww[in], p_net[psi].

        NOTE (v3 fix): v2 computed the leak-off area at time t using xf from
        the *previous* time step (`A_f = xf_now * Hf_ft` before xf_now was
        updated), i.e. an explicit, one-step-lagged approximation, then
        solved the (now leak-off-inconsistent) volume balance in closed
        form. `scipy.optimize.brentq` was imported for exactly this kind of
        root-find but was never actually called anywhere in the module. v3
        solves the true implicit equation
            g(xf) = V_inj(t) - V_leakoff(xf, t) - V_frac(xf) = 0
        for a single self-consistent xf at each t.
        """
        t_arr  = np.linspace(0.1, t_min, n_steps)  # minutes
        xf_arr = np.zeros(n_steps)
        ww_arr = np.zeros(n_steps)
        pn_arr = np.zeros(n_steps)
        C = (np.pi**2 / 16) * 0.1524 * (qi_bpm * mu_cp / (Ep_psi * Hf_ft))**0.25 * Hf_ft / 12

        def volume_balance(xf, V_inj_gal, t):
            V_lo_gal   = 2 * CL * (xf * Hf_ft) * np.sqrt(t) * 7.4805
            V_frac_gal = C * (xf ** 1.25) * 42
            return V_inj_gal - V_lo_gal - V_frac_gal

        xf_prev = 1.0
        for i, t in enumerate(t_arr):
            V_inj = qi_bpm * t * 42  # gallons (one wing)
            lo_bracket = 1e-3
            hi_bracket = max(xf_prev * 4, 50.0)
            g_hi = volume_balance(hi_bracket, V_inj, t)
            expand_iter = 0
            while g_hi > 0 and expand_iter < 20:
                hi_bracket *= 2
                g_hi = volume_balance(hi_bracket, V_inj, t)
                expand_iter += 1
            g_lo = volume_balance(lo_bracket, V_inj, t)
            if g_lo <= 0:
                xf_now = lo_bracket
            elif g_hi > 0:
                xf_now = hi_bracket   # pathological (very high leak-off); use cap
            else:
                xf_now = brentq(volume_balance, lo_bracket, hi_bracket, args=(V_inj, t), xtol=1e-6)
            xf_now  = max(xf_now, 1.0)
            xf_prev = xf_now
            geo = FractureGeometry.pkn_width(qi_bpm, mu_cp, xf_now, Ep_psi, Hf_ft)
            xf_arr[i] = xf_now
            ww_arr[i] = geo["w_max_in"]
            pn_arr[i] = geo["net_pressure_psi"]
        return {"t_min": t_arr, "xf_ft": xf_arr,
                "w_max_in": ww_arr, "p_net_psi": pn_arr}

    # ── KGD ───────────────────────────────────────────────────────────────────
    @staticmethod
    def kgd_width(qi_bpm: float, mu_cp: float, L_ft: float,
                  Ep_psi: float, Hf_ft: float) -> dict:
        """
        KGD maximum wellbore width.
        w_w [in] = 0.1524 · (qi·μ·L³ / (E'·Hf²))^(1/4)
        """
        if Ep_psi <= 0 or Hf_ft <= 0:
            return {"w_max_in": 0.1, "w_avg_in": 0.08, "model": "KGD"}
        arg   = (qi_bpm * mu_cp * L_ft**3) / (Ep_psi * Hf_ft**2)
        w_max = 0.1524 * max(arg, 1e-12) ** 0.25
        w_avg = w_max  # KGD approximately constant width along length
        p_net = Ep_psi * w_max / (2 * np.pi * L_ft / 12)
        return {"w_max_in": float(w_max), "w_avg_in": float(w_avg),
                "net_pressure_psi": float(p_net), "model": "KGD"}

    # ── Radial (Penny) ────────────────────────────────────────────────────────
    @staticmethod
    def radial_width(qi_bpm: float, mu_cp: float, R_ft: float,
                     Ep_psi: float) -> dict:
        """
        Radial fracture width at wellbore.
        w_w [in] = 2.24 · (qi·μ·R / E')^(1/4)  (approximate)
        """
        if Ep_psi <= 0:
            return {"w_max_in": 0.1, "w_avg_in": 0.08, "model": "Radial"}
        arg   = (qi_bpm * mu_cp * R_ft) / Ep_psi
        w_max = 2.24 * max(arg, 1e-12) ** 0.25
        w_avg = (2/3) * w_max  # radial profile
        p_net = Ep_psi * w_max / (4 * R_ft * 12)
        return {"w_max_in": float(w_max), "w_avg_in": float(w_avg),
                "net_pressure_psi": float(p_net), "model": "Radial"}

    # ── Plane Strain Modulus ──────────────────────────────────────────────────
    @staticmethod
    def plane_strain_modulus(E_psi: float, nu: float) -> float:
        """E' = E / (1 - ν²)"""
        return E_psi / max(1 - nu**2, 0.01)

    # ── Fracture Volume ───────────────────────────────────────────────────────
    @staticmethod
    def fracture_volume_bbl(w_avg_in: float, xf_ft: float, Hf_ft: float) -> float:
        """One-wing fracture volume in bbls."""
        V_ft3 = (w_avg_in / 12) * xf_ft * Hf_ft
        return float(V_ft3 / 5.615)

    # ── Fluid Efficiency ─────────────────────────────────────────────────────
    @staticmethod
    def fluid_efficiency(V_frac_bbl: float, V_inj_bbl: float) -> float:
        return float(np.clip(V_frac_bbl / max(V_inj_bbl, 0.01), 0.01, 0.99))


# ══════════════════════════════════════════════════════════════════════════════
# 2.  PROPPANT DESIGN
# ══════════════════════════════════════════════════════════════════════════════
class ProppantDesign:
    """Proppant scheduling, propped width, and fracture conductivity."""

    # Empirical conductivity vs closure stress for common proppant types (mD-ft)
    PROPPANT_DB = {
        # kf0 [mD] = fracture permeability at zero closure stress
        # beta = stress-decay exponent (Barree & Conway 2004)
        # At σ_eff=4000 psi: kf = kf0 * exp(-beta * σ_eff/1000)
        # Reference: API RP 19D (2008); Palisch et al. SPE-166265 (2013)
        "20/40 Ottawa Sand":     {"kf0": 500000,  "beta": 0.55},
        "20/40 Resin Sand":      {"kf0": 800000,  "beta": 0.45},
        "20/40 Intermediate ICP":{"kf0": 1500000, "beta": 0.40},
        "16/20 Lightweight":     {"kf0": 2000000, "beta": 0.35},
        "20/40 High-Strength":   {"kf0": 2500000, "beta": 0.32},
    }

    @staticmethod
    def propped_width(mass_klbs: float, xf_ft: float, Hf_ft: float,
                      rho_prop_gcc: float = 2.65, phi_pack: float = 0.35) -> float:
        """
        Propped fracture width (one wing, two faces):
        wp [in] = mass [lb] / (ρ_bulk [lb/ft³] × 2 × xf [ft] × Hf [ft]) × 12
        ρ_bulk = ρ_prop × (1 - φ_pack)
        """
        rho_lb_ft3 = rho_prop_gcc * 62.4 * (1 - phi_pack)
        area_ft2   = 2 * xf_ft * Hf_ft
        wp_ft      = (mass_klbs * 1000) / (rho_lb_ft3 * area_ft2)
        return float(wp_ft * 12)  # inches

    @staticmethod
    def fracture_conductivity(wp_in: float, sigma_eff_psi: float,
                              proppant_type: str = "20/40 Ottawa Sand") -> dict:
        """
        Fracture conductivity kf·wf [mD-ft].
        kf(σ) = kf0 · exp(–β · σ_eff/1000)
        Reference: API RP 19D (2008); Palisch et al. SPE-166265 (2013).
        At 4000 psi effective stress:
          Ottawa sand   ≈ 1000–2000 mD-ft (wp=0.5 in)
          Ceramic ICP   ≈ 4000–8000 mD-ft
        """
        db  = ProppantDesign.PROPPANT_DB.get(proppant_type,
              ProppantDesign.PROPPANT_DB["20/40 Ottawa Sand"])
        kf0 = db["kf0"]
        b   = db["beta"]
        kf  = kf0 * np.exp(-b * max(sigma_eff_psi, 0) / 1000)
        kf_wf = kf * (wp_in / 12)  # convert wp inches→ft, multiply by kf [mD]
        return {"kf_mD": float(kf), "kf_wf_mDft": float(kf_wf),
                "propped_width_in": float(wp_in)}

    @staticmethod
    def dimensionless_conductivity(kf_wf_mDft: float,
                                    k_res_mD: float, xf_ft: float) -> float:
        """
        Fcd = kf · wf / (k_res · xf)
        Optimal Fcd ≈ 1.6 (Cinco-Ley & Samaniego 1981 / Prats 1961).
        """
        return float(kf_wf_mDft / max(k_res_mD * xf_ft, 1e-6))

    @staticmethod
    def pump_schedule(qi_bpm: float, total_fluid_bbl: float,
                      pad_fraction: float = 0.20,
                      ramp_stages: int = 6,
                      max_proppant_ppg: float = 8.0) -> pd.DataFrame:
        """
        Generate HF pump schedule with ramp from 0 to max proppant concentration.
        Returns DataFrame: stage, volume_bbl, cum_vol_bbl, rate_bpm, prop_conc_ppg, prop_mass_klbs

        NOTE (v3 fix): `scipy.integrate.quad` was imported but never used
        anywhere in v2. Proppant mass per stage is now the true integral of
        a linearly-ramping concentration over the stage's pumped volume,
        rather than `constant_concentration x stage_volume` (which
        implicitly assumed each stage jumps instantly to its target
        concentration and holds it — a coarser approximation that
        understates mass during the ramp-up portion of every stage).
        """
        pad_vol  = total_fluid_bbl * pad_fraction
        frac_vol = total_fluid_bbl - pad_vol
        stage_vol= frac_vol / ramp_stages
        conc_steps = np.linspace(0, max_proppant_ppg, ramp_stages)

        rows = [{"stage": 0, "name": "Pad (0 ppg)",
                 "volume_bbl": pad_vol, "rate_bpm": qi_bpm,
                 "prop_conc_ppg": 0.0, "prop_mass_klbs": 0.0}]
        cum = pad_vol
        conc_prev = 0.0
        for i, conc in enumerate(conc_steps):
            conc_now = float(conc)
            # integrate the linear ramp c(v) = conc_prev + (conc_now-conc_prev)*v/stage_vol
            # over v in [0, stage_vol]; result in lb (ppg * gal), then -> klbs
            mass_lb, _ = quad(lambda v, cp=conc_prev, cn=conc_now, sv=stage_vol:
                               (cp + (cn - cp) * v / max(sv, 1e-9)) * 42, 0, stage_vol)
            pm = mass_lb / 1000
            cum += stage_vol
            rows.append({
                "stage": i + 1,
                "name":  f"Stage {i+1} ({conc_now:.1f} ppg)",
                "volume_bbl": stage_vol, "rate_bpm": qi_bpm,
                "prop_conc_ppg": conc_now, "prop_mass_klbs": float(pm),
            })
            conc_prev = conc_now
        rows.append({"stage": ramp_stages+1, "name": "Flush",
                     "volume_bbl": qi_bpm * 5, "rate_bpm": qi_bpm,
                     "prop_conc_ppg": 0.0, "prop_mass_klbs": 0.0})
        df = pd.DataFrame(rows)
        df["cum_vol_bbl"]      = df["volume_bbl"].cumsum()
        df["cum_prop_klbs"]    = df["prop_mass_klbs"].cumsum()
        df["pump_time_min"]    = df["volume_bbl"] / qi_bpm
        df["cum_pump_time_min"]= df["pump_time_min"].cumsum()
        return df


# ══════════════════════════════════════════════════════════════════════════════
# 3.  G-FUNCTION  DFIT / MINI-FRAC ANALYSIS  (Nolte 1979)
# ══════════════════════════════════════════════════════════════════════════════
class DFITAnalysis:
    """
    Diagnostic Fracture Injection Test (DFIT) analysis using G-function.
    Reference: Nolte (1979) SPE-8297; Craig & Blasingame (2006) SPE-96040.
    """

    @staticmethod
    def g_function(delta_tD: np.ndarray, alpha: float = 0.5) -> np.ndarray:
        """
        G(ΔtD) = (4/3)·[(1+ΔtD)^(1+α) - ΔtD^(1+α) - 1] / (1+α)
        Standard form with α = 0.5 (high-leak-off) or 1.0 (low-leak-off).
        """
        return (4/3) * ((1 + delta_tD)**(1+alpha) - delta_tD**(1+alpha) - 1) / (1+alpha)

    @staticmethod
    def dfit_pressure_decline(P_isip: float, delta_p: float,
                               tp_min: float, t_shut_max_min: float = 120,
                               CL: float = 0.002, n_pts: int = 200,
                               closure_dtD: float = 1.2) -> pd.DataFrame:
        """
        Simulate post-shut-in pressure decline on a G-function plot.
        Returns DataFrame: t_shut_min, delta_tD, G, P_bhp_psi, dPdG, ...

        NOTE (v3 fix): the previous synthetic decline was a perfectly
        linear function of G (`P = ISIP - m_G*G`) everywhere, which by
        construction has zero curvature (d2P/dG2 = 0 at every point). The
        closure-picking logic below it looks for where the curvature
        changes most sharply — but a perfectly linear curve carries no such
        signal, so it was really just picking up floating-point noise.
        Real DFIT closure is identified exactly as a *departure* from the
        pre-closure straight-line trend (Nolte, 1979; see also Barree,
        Barree & Craig, 2009, SPE 122208): pressure declines linearly with
        G while the fracture is open and storing fluid, then the slope
        shallows once the fracture closes and flow transitions toward the
        formation. The synthetic curve now has that two-regime shape by
        construction, giving the inflection-point detector a genuine
        feature to find — validated in tests/test_hf_design.py by checking
        the recovered closure index lands near the injected `closure_dtD`.
        """
        t_arr = np.linspace(0.01, t_shut_max_min, n_pts)
        dtD   = t_arr / tp_min
        G     = DFITAnalysis.g_function(dtD)
        G_c   = float(DFITAnalysis.g_function(np.array([closure_dtD]))[0])

        # Pre-closure slope m1 is steeper; post-closure m2 is shallower
        # (formation-dominated flow), calibrated so the TOTAL decline over
        # the full window still equals delta_p.
        slope_ratio = 0.35
        denom = G_c + slope_ratio * max(G[-1] - G_c, 1e-9)
        m1 = delta_p / max(denom, 1e-9)
        m2 = slope_ratio * m1
        P = np.where(G <= G_c, P_isip - m1 * G,
                     P_isip - m1 * G_c - m2 * (G - G_c))

        dPdG   = np.gradient(P, G)
        d2PdG2 = np.gradient(dPdG, G)
        clos_idx = int(np.argmin(np.abs(d2PdG2[5:-5])) + 5) if n_pts > 10 else 0
        # the sharpest curvature change is a local property; report the
        # single sharpest kink even if numerical noise creates ties nearby
        clos_idx = int(np.argmax(np.abs(np.gradient(d2PdG2)))) if n_pts > 10 else clos_idx
        P_closure = float(P[clos_idx])
        return pd.DataFrame({
            "t_shut_min":  t_arr,
            "delta_tD":    dtD,
            "G":           G,
            "P_bhp_psi":   P,
            "dPdG":        dPdG,
            "closure_idx": [clos_idx == i for i in range(len(t_arr))],
            "P_closure_psi": P_closure,
        })

    @staticmethod
    def closure_stress(P_closure_psi: float, pore_pressure_psi: float) -> dict:
        """Compute closure stress and net pressure."""
        sigma_c  = P_closure_psi
        p_net    = max(0, sigma_c - pore_pressure_psi)
        return {
            "closure_stress_psi": float(sigma_c),
            "effective_stress_psi": float(sigma_c - pore_pressure_psi),
            "net_pressure_psi": float(p_net),
        }


# ══════════════════════════════════════════════════════════════════════════════
# 4.  POST-FRAC PRODUCTIVITY
# ══════════════════════════════════════════════════════════════════════════════
class PostFracProductivity:
    """
    Fractured well productivity index and folds-of-increase.
    References: Cinco-Ley & Samaniego (1981) JPT;
                Prats (1961) Trans. AIME; McGuire & Sikora (1960).
    """

    @staticmethod
    def pi_ratio(Fcd: float, xf_ft: float, re_ft: float,
                 rw_ft: float = 0.354) -> float:
        """
        PI_frac / PI_unfrac using Cinco-Ley (1981) bilinear flow model.
        For Fcd >> 10 (high conductivity): approaches infinite conductivity limit.
        """
        ln_re_rw  = np.log(re_ft / rw_ft)
        ln_re_xf  = np.log(re_ft / xf_ft)

        if Fcd >= 50:
            # Infinite-conductivity fracture (Prats 1961)
            J_ratio = ln_re_rw / max(ln_re_xf - 0.75, 0.1)
        elif Fcd >= 1.6:
            # Finite conductivity (Cinco-Ley approximation)
            f_Fcd   = np.pi / (4 * Fcd)  + 0.5 * np.log(xf_ft / rw_ft) / Fcd
            J_ratio = ln_re_rw / max(ln_re_xf - 0.75 + f_Fcd, 0.1)
        else:
            # Low conductivity (bilinear flow dominated)
            f_Fcd   = min(np.pi / (4 * max(Fcd, 0.01)) + 2.0, 50.0)
            J_ratio = ln_re_rw / max(ln_re_xf + f_Fcd, 0.5)

        return float(max(J_ratio, 1.0))

    @staticmethod
    def optimal_xf(k_mD: float, kf_wf_mDft: float,
                   re_ft: float = 1320, rw_ft: float = 0.354) -> dict:
        """
        Unified Fracture Design (Economides et al. 2002):
        Optimal xf where Fcd = 1.6 (maximises PI for given proppant volume).
        xf_opt = kf·wf / (1.6 · k_res)  [ft]
        """
        xf_opt = kf_wf_mDft / max(1.6 * k_mD, 1e-6)
        xf_opt = float(np.clip(xf_opt, 50, re_ft * 0.8))
        Fcd_opt = 1.6  # optimal per Cinco-Ley (1981)
        return {"xf_optimal_ft": xf_opt, "target_Fcd": Fcd_opt}

    @staticmethod
    def productivity_improvement_table(
            k_mD: float, h_ft: float, mu_cp: float, Bo: float,
            re_ft: float, rw_ft: float,
            xf_values_ft: list, Fcd_values: list) -> pd.DataFrame:
        """Generate PI comparison table for different xf and Fcd combinations."""
        from modules.simulator import FlowSimulator
        J0 = FlowSimulator(rw_ft).PI(k_mD, h_ft, mu_cp, Bo, re_ft, S=0)
        rows = []
        for xf in xf_values_ft:
            for Fcd in Fcd_values:
                ratio = PostFracProductivity.pi_ratio(Fcd, xf, re_ft, rw_ft)
                rows.append({
                    "xf_ft": xf, "Fcd": Fcd,
                    "J_frac_stbdpsi": round(J0 * ratio, 3),
                    "PI_ratio": round(ratio, 2),
                    "PI_improvement_pct": round((ratio - 1) * 100, 1),
                })
        return pd.DataFrame(rows)


# ══════════════════════════════════════════════════════════════════════════════
# 5.  COMPLETE HF DESIGN RUNNER
# ══════════════════════════════════════════════════════════════════════════════
class HFDesignEngine:
    """
    Orchestrates all HF modules: geometry → proppant → conductivity →
    pump schedule → DFIT analysis → post-frac PI.
    """

    def __init__(self):
        self.geo   = FractureGeometry()
        self.prop  = ProppantDesign()
        self.dfit  = DFITAnalysis()
        self.pf    = PostFracProductivity()

    def run(self, hf_params: dict, res_params: dict) -> dict:
        # ── unpack ──────────────────────────────────────────────────────────
        E_psi      = hf_params.get("youngs_modulus_psi",
                        res_params.get("youngs_modulus_GPa", 15) * 145038)
        nu         = res_params.get("poisson_ratio", 0.28)
        Ep         = FractureGeometry.plane_strain_modulus(E_psi, nu)
        qi         = hf_params.get("pump_rate_bpm", 30.0)
        mu_f       = hf_params.get("fluid_viscosity_cp", 200.0)
        xf_t       = hf_params.get("target_xf_ft", 500.0)
        Hf         = hf_params.get("target_hf_ft", 80.0)
        total_fl   = hf_params.get("total_fluid_bbl", 3000.0)
        prop_mass  = hf_params.get("proppant_mass_klbs", 400.0)
        prop_type  = hf_params.get("proppant_type", "20/40 Ottawa Sand")
        rho_prop   = hf_params.get("proppant_density_gcc", 2.65)
        phi_pack   = hf_params.get("proppant_pack_phi", 0.35)
        sigma_cl   = hf_params.get("closure_stress_psi", 4200.0)
        CL         = hf_params.get("leakoff_coeff_ft_sqrtmin", 0.002)
        model      = hf_params.get("model", "PKN").upper()
        k_res      = res_params.get("permeability_mD", 150)
        re_ft      = res_params.get("drainage_radius_ft", 1320)
        rw_ft      = res_params.get("wellbore_radius_ft", 0.354)
        Pp         = res_params.get("pore_pressure_psi", 3325)
        h_res      = res_params.get("net_pay_ft", 25)
        mu_o       = res_params.get("oil_viscosity_cp", 2.5)
        Bo         = res_params.get("oil_fvf", 1.25)
        n_stages   = hf_params.get("n_stages", 5)

        # ── Step 1: Fracture geometry ────────────────────────────────────────
        if model == "KGD":
            geo = self.geo.kgd_width(qi, mu_f, xf_t, Ep, Hf)
        elif model == "RADIAL":
            geo = self.geo.radial_width(qi, mu_f, xf_t, Ep)
        else:
            geo = self.geo.pkn_width(qi, mu_f, xf_t, Ep, Hf)

        w_avg  = geo["w_avg_in"]
        p_net  = geo["net_pressure_psi"]

        # ── Step 2: Fracture volume and efficiency ───────────────────────────
        V_frac = self.geo.fracture_volume_bbl(w_avg, xf_t, Hf)
        V_inj  = total_fl / 2  # one wing
        eff    = self.geo.fluid_efficiency(V_frac, V_inj)

        # ── Step 3: Propped width and conductivity ───────────────────────────
        wp    = self.prop.propped_width(prop_mass, xf_t, Hf, rho_prop, phi_pack)
        sigma_eff = sigma_cl - Pp
        cond  = self.prop.fracture_conductivity(wp, sigma_eff, prop_type)
        Fcd   = self.prop.dimensionless_conductivity(cond["kf_wf_mDft"], k_res, xf_t)

        # ── Step 4: Pump schedule ────────────────────────────────────────────
        schedule_df = self.prop.pump_schedule(qi, total_fl, ramp_stages=max(2, int(n_stages)))

        # ── Step 5: PKN time evolution ───────────────────────────────────────
        pump_time = total_fl / qi  # minutes
        time_evo = self.geo.pkn_length_time(qi, mu_f, Ep, Hf, CL, pump_time)

        # ── Step 6: DFIT simulation ──────────────────────────────────────────
        P_isip = sigma_cl + p_net * 0.85
        dfit_df = self.dfit.dfit_pressure_decline(
            P_isip, p_net * 1.2, pump_time, pump_time * 2, CL)
        clos_info = self.dfit.closure_stress(sigma_cl, Pp)

        # ── Step 7: Post-frac productivity ──────────────────────────────────
        pi_ratio = self.pf.pi_ratio(Fcd, xf_t, re_ft, rw_ft)

        # Un-fractured baseline
        from modules.simulator import FlowSimulator
        J0   = FlowSimulator(rw_ft).PI(k_res, h_res, mu_o, Bo, re_ft, S=0)
        J_f  = J0 * pi_ratio

        xf_table = self.pf.productivity_improvement_table(
            k_res, h_res, mu_o, Bo, re_ft, rw_ft,
            [100, 250, 500, 750, 1000], [0.5, 1.6, 5.0, 10.0, 50.0])

        # ── Compile results ──────────────────────────────────────────────────
        return {
            # Geometry
            "model":          model,
            "xf_ft":          xf_t,
            "Hf_ft":          Hf,
            "w_max_in":       geo["w_max_in"],
            "w_avg_in":       w_avg,
            "net_pressure_psi": p_net,
            "V_frac_bbl":     V_frac,
            "fluid_efficiency": eff,
            "pump_time_min":  pump_time,
            # Proppant
            "propped_width_in": wp,
            "kf_mD":          cond["kf_mD"],
            "kf_wf_mDft":     cond["kf_wf_mDft"],
            "Fcd":            Fcd,
            # Reservoir productivity
            "PI_unfrac":      J0,
            "PI_frac":        J_f,
            "PI_ratio":       pi_ratio,
            "PI_improvement_pct": (pi_ratio - 1) * 100,
            # Stress
            "closure_stress_psi": clos_info["closure_stress_psi"],
            "ISIP_psi":       P_isip,
            # Arrays
            "schedule_df":    schedule_df,
            "time_evolution": time_evo,
            "dfit_df":        dfit_df,
            "xf_table":       xf_table,
        }
