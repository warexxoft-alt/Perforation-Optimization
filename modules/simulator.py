"""
modules/simulator.py
Coupled flow-geomechanical simulation engine.
Implements: Karakas-Tariq skin, Darcy IPR, Arps decline, Biot poroelasticity,
Mohr-Coulomb failure, damage evolution (Eq 3.17 analog from proposal).
"""

import numpy as np
import pandas as pd

from config import KT_COEFF


# ══════════════════════════════════════════════════════════════════════════════
# 1.  PERFORATION SKIN CALCULATOR  (Karakas & Tariq, 1992 — SPE-18247)
# ══════════════════════════════════════════════════════════════════════════════
class SkinCalculator:
    """Karakas-Tariq semianalytical perforation skin model."""

    def _coeff(self, phasing: float) -> dict:
        available = sorted(KT_COEFF.keys())
        best = min(available, key=lambda x: abs(x - phasing))
        return KT_COEFF[best]

    def kt_skin(self, spf, pen_in, hole_in, rw_ft, kH, kV, phasing) -> dict:
        """
        Karakas & Tariq (1988) perforation skin — SPE 18247.
        Parameters (all floats):
          spf     – shots / ft
          pen_in  – penetration depth (in)
          hole_in – hole (perforation) diameter (in)
          rw_ft   – wellbore radius (ft)
          kH, kV  – horizontal / vertical permeability (mD)
          phasing – degrees
        Returns dict with S_H, S_V, S_wb, S_KT.

        NOTE (v3 fix): this method previously used a fabricated coefficient
        table and a non-standard formula structure (see config.KT_COEFF for
        the full history); most visibly, the vertical-skin term was
        completely insensitive to shots-per-foot, which the source
        describes as "potentially the largest contributor to sp". Every
        equation below is numbered against Karakas & Tariq as reproduced in
        Bell, Snider & Barker, "Perforating", SPE Monograph 16, Ch.13, and
        the whole method is validated against that chapter's fully worked
        example in tests/test_simulator.py.
        """
        Lp = max(pen_in / 12.0, 1e-6)                 # ft, eq. inputs
        rp = hole_in / 24.0                            # ft (radius = dia/2)
        hp = 1.0 / max(spf, 0.05)                       # ft, perforation spacing = 1/SPF
        aniso = np.sqrt(max(kH, 1e-6) / max(kV, 1e-6))  # sqrt(kH/kV)
        c = self._coeff(phasing)

        # (13-5)/(13-4): effective wellbore radius & horizontal (plane-flow) skin
        rw_eff = Lp if abs(phasing) < 1e-6 else c["a"] * (rw_ft + Lp)
        SH = float(np.log(rw_ft / max(rw_eff, 1e-9)))

        # (13-6)/(13-7): dimensionless vertical spacing & perforation radius
        hD = (hp / Lp) * aniso
        rD = (rp / (2.0 * hp)) * (1.0 + 1.0 / aniso)

        # (13-9)/(13-10)/(13-8): vertical (converging-flow) pseudo-skin
        a_exp = c["a1"] * np.log10(max(rD, 1e-12)) + c["a2"]
        b_exp = c["b1"] * rD + c["b2"]
        SV = float((10.0 ** a_exp) * (hD ** (b_exp - 1.0)) * (rD ** b_exp))

        # (13-11)/(13-12): wellbore-geometry skin
        rwD = rw_ft / (Lp + rw_ft)
        Swb = float(c["c1"] * np.exp(c["c2"] * rwD))

        SKT = SH + SV + Swb
        return {"S_H": SH, "S_V": SV, "S_wb": Swb, "S_KT": float(SKT)}

    def cleanup_efficiency(self, ub_psi, k):
        """Yang & Wang (2015) SPE-172892 cleanup model. Returns fraction 0–1."""
        if ub_psi >= 0:
            return float(np.clip(0.20 + 0.04 * k / 1000, 0.10, 0.40))
        abs_ub = abs(ub_psi)
        ce = 1.0 - np.exp(-abs_ub / 500.0 * np.log1p(k / 50.0))
        return float(np.clip(ce, 0.10, 0.98))

    def damage_skin(self, k, k_cz_ratio, rw, rp_cz, pen_in):
        """Crushed-zone skin contribution (McLeod 1983, SPE-10649)."""
        Lp = pen_in / 12.0
        kcz = k * k_cz_ratio
        if kcz <= 0 or kcz >= k:
            return 0.0
        s = (k / kcz - 1.0) * np.log((rw + rp_cz + Lp) / max(rw + rp_cz, 1e-6))
        return float(max(0.0, s))

    def nondarcy_coeff(self, k, h, rw, mu, phi=0.22):
        """
        Non-Darcy (Forchheimer) coefficient D for oil wells [1/(STB/day)].
        For oil at typical rates, turbulence is minor.
        Correlation: D ∝ β·k·ρ / (α·μ·h·rw)  capped at 0.005 for oil.
        Ref: Jones (1987); Furui et al. SPE-84401.
        v3: `phi` is now the well's actual porosity (was hardcoded to 0.22
        for every well regardless of its true reservoir quality).
        """
        beta = 1.88e10 * (max(k, 1e-3) ** -1.47) * (max(phi, 1e-3) ** -0.53)  # 1/ft
        rho  = 50.0   # lb/ft³
        D    = beta * rho * k / (887040.0 * mu * h * rw)
        # Cap: oil turbulence contribution S_nd = D×q typically 0.1–3 for oil
        D    = min(D, 0.003)
        return float(max(0.0, D))

    def total(self, p: dict, q_est=800.0) -> dict:
        k   = p.get("permeability_mD", 100.0)
        kv  = p.get("k_vertical_mD", k * 0.10)
        h   = p.get("net_pay_ft", 25.0)
        rw  = p.get("wellbore_radius_ft", 0.354)
        mu  = p.get("oil_viscosity_cp", 2.5)
        Sd  = p.get("damage_skin", 3.0)
        spf = p.get("spf", 12.0)
        ph  = p.get("phasing", 90.0)
        pen = p.get("pen_depth_in", 18.0)
        hd  = p.get("hole_dia_in", 0.45)
        ub  = p.get("underbalance_psi", -500.0)

        kt = self.kt_skin(spf, pen, hd, rw, k, kv, ph)
        ce = self.cleanup_efficiency(ub, k)
        Scrush = self.damage_skin(k, 0.10, rw, 0.05, pen)
        Sdeff  = Sd * (1 - ce) + Scrush * (1 - ce * 0.8)
        D      = self.nondarcy_coeff(k, h, rw, mu, phi=p.get("porosity", 0.22))
        Snd    = D * q_est

        St = kt["S_KT"] + Sdeff + Snd
        return {**kt, "S_damage": float(Sdeff), "S_nondarcy": float(Snd),
                "S_total": float(St), "CE": float(ce), "D_coeff": float(D)}


# ══════════════════════════════════════════════════════════════════════════════
# 2.  FLOW SIMULATOR  (Darcy radial flow + Vogel IPR + Arps decline)
# ══════════════════════════════════════════════════════════════════════════════
class FlowSimulator:
    def __init__(self, rw=0.354):
        self.rw = rw

    def PI(self, k, h, mu, Bo, re, S, regime="PSS"):
        """Productivity Index (STB/day/psi) — field units (Darcy law)."""
        F  = 0.5 if regime == "PSS" else 0.75
        ln = np.log(re / self.rw) - F + S
        ln = max(ln, 0.01)
        return float(0.00708 * k * h / (mu * Bo * ln))

    def ipr(self, PR, J, Pb=None, n=80):
        """Vogel + Darcy composite IPR.  Returns (Pwf_array, q_array)."""
        Pb = Pb or 0.70 * PR
        Pwf = np.linspace(0, PR, n)
        q   = np.zeros(n)
        qmax_v = J * PR / 1.8
        for i, pw in enumerate(Pwf):
            if pw >= Pb:
                q[i] = J * (PR - pw)
            else:
                r = pw / PR
                q[i] = qmax_v * (1 - 0.2 * r - 0.8 * r * r)
        return Pwf, np.maximum(q, 0)

    def forecast(self, qi, Di=0.35, b=1.2, tmax=20.0, qecon=50, dt=0.1):
        """Hyperbolic Arps decline (1945).  Returns (t, q, EUR, t_econ)."""
        t = np.arange(0, tmax + dt, dt)
        if b < 1e-4:
            q = qi * np.exp(-Di * t)
        elif abs(b - 1.0) < 1e-4:
            q = qi / (1.0 + Di * t)
        else:
            q = qi / (1.0 + b * Di * t) ** (1.0 / b)
        q = np.maximum(q, 0)
        idx = np.searchsorted(-q, -qecon)
        idx = min(idx, len(t) - 1)
        t_econ = t[idx]
        q[idx:] = 0.0
        EUR = float(np.trapezoid(q, t) * 365)   # convert yr→days
        return t, q, EUR, float(t_econ)

    def npv(self, t, q, Poil=70, Cop=8, r=0.10, Ccap=7e6):
        """Discounted cash-flow NPV."""
        dt_arr = np.diff(t, prepend=0)
        net_rev = (Poil - Cop) * q * 365 * dt_arr
        disc    = 1.0 / (1.0 + r) ** t
        DCF     = net_rev * disc
        NPV     = float(np.sum(DCF)) - Ccap
        ROI     = NPV / max(Ccap, 1)
        pay_idx = int(np.searchsorted(np.cumsum(DCF), Ccap))
        pay     = float(t[min(pay_idx, len(t) - 1)])
        return {"NPV": NPV, "ROI": float(ROI), "payback": pay,
                "total_DCF": float(np.sum(DCF)), "Ccap": float(Ccap)}


# ══════════════════════════════════════════════════════════════════════════════
# 3.  GEOMECHANICAL MODEL  (Biot poroelasticity + Mohr-Coulomb)
# ══════════════════════════════════════════════════════════════════════════════
class GeoMechModel:

    def eff_stress(self, sig_total, Pp, alpha=0.85):
        return float(sig_total - alpha * Pp)

    def mohr_coulomb(self, sig1, sig3, UCS_psi, phi_deg):
        phi  = np.radians(phi_deg)
        tp   = np.tan(np.pi / 4 + phi / 2)
        sig1_fail = UCS_psi + sig3 * tp ** 2
        SF   = sig1_fail / max(sig1, 0.001)
        return {"failed": bool(sig1 >= sig1_fail),
                "SF": float(SF), "sig1_critical": float(sig1_fail)}

    def critical_drawdown(self, UCS_MPa, sig_h, sig_H, rw, phi_deg, alpha=0.85):
        """Analytical wellbore stability (Kirsch + M-C). Returns psi."""
        UCS_psi = UCS_MPa * 145.038
        sig_theta_max = 3 * sig_H - sig_h           # Kirsch tangential stress
        phi = np.radians(phi_deg)
        DP = max(0, sig_theta_max - UCS_psi / (1 - np.sin(phi)))
        return float(DP)

    def damage_evolution(self, D, tau, tau_c, grad_p, grad_p_ref,
                         A=1e-2, B=0.5, m=1.5, dt=3.0):
        """
        dD/dt = A·[(τ-τc)/τc]^1.5 · (∇p/∇p_ref)^m — Eq 3.17 analog.

        NOTE (v3 fix): the v2 default A=1e-6 was small enough that, combined
        with the ~30-step/90-day integration window used in
        `CoupledSimulator.run_well`, `damage_index` was mathematically
        frozen at its initial value (0.05) for every well across the entire
        realistic Niger Delta parameter space used by this app — the
        mechanism looked like a real, evolving physical quantity but was
        actually an inert constant (verified: 0/500 synthetic wells showed
        ANY change with A=1e-6; see tests/test_simulator.py). A=1e-2 was
        selected by scanning A against this app's actual parameter ranges
        so that damage meaningfully escalates for high-drawdown/low-
        strength wells while staying low and stable for well-designed ones,
        without routinely saturating at the 0.99 cap.
        """
        if tau <= tau_c:
            return float(D)
        mech  = ((tau - tau_c) / max(tau_c, 1e-9)) ** 1.5
        hydro = max(grad_p / max(grad_p_ref, 1e-9), 0.01) ** m
        dDdt  = A * mech * hydro * B
        return float(min(D + dDdt * dt, 0.99))

    def casing_vm_stress(self, Pwb_psi, sig_h_psi, sig_H_psi, ri, ro):
        """Von Mises stress in casing wall (Lamé + far-field tectonic loading)."""
        t   = ro - ri
        if t <= 0:
            return {"VM_psi": 0.0}
        sig_h_c = Pwb_psi * ri / t
        sig_ext = (sig_h_psi + sig_H_psi) / 2 * (ro**2 + ri**2) / (ro**2 - ri**2)
        sh      = sig_h_c - sig_ext
        sa      = 0.3 * sh
        VM      = float(abs(np.sqrt(sh**2 - sh * sa + sa**2)))
        return {"VM_psi": VM, "hoop_psi": float(sh), "axial_psi": float(sa)}


# ══════════════════════════════════════════════════════════════════════════════
# 4.  COUPLED SIMULATOR  (orchestrates all three modules)
# ══════════════════════════════════════════════════════════════════════════════
class CoupledSimulator:
    def __init__(self):
        self.skin = SkinCalculator()
        self.flow = FlowSimulator()
        self.geo  = GeoMechModel()

    # ── single-well run ───────────────────────────────────────────────────────
    def run_well(self, p: dict) -> dict:
        k    = float(p.get("permeability_mD", 100))
        h    = float(p.get("net_pay_ft", 25))
        phi  = float(p.get("porosity", 0.22))
        PR   = float(p.get("initial_pressure_psi", 3500))
        mu   = float(p.get("oil_viscosity_cp", 2.5))
        Bo   = float(p.get("oil_fvf", 1.25))
        re   = float(p.get("drainage_radius_ft", 1320))
        rw   = float(p.get("wellbore_radius_ft", 0.354))
        Pp   = float(p.get("pore_pressure_psi", PR * 0.95))
        sig_h = float(p.get("min_h_stress_psi", 4200))
        sig_H = float(p.get("max_h_stress_psi", 5000))
        biot  = float(p.get("biot_coefficient", 0.85))
        phi_f = float(p.get("friction_angle_deg", 30))
        UCS   = float(p.get("UCS_MPa", 25))
        nst   = float(p.get("n_stages", 20))
        ncl   = float(p.get("n_clusters", 4))
        spf   = float(p.get("spf", 12))
        pen   = float(p.get("pen_depth_in", 18))

        # Step 1 – skin (estimate q_est for non-Darcy)
        sr = self.skin.total(p, q_est=500)
        St = sr["S_total"]

        # Step 2 – productivity index
        J  = self.flow.PI(k, h, mu, Bo, re, St)
        Pwf_op = max(200.0, PR * 0.30)
        qi = max(10.0, J * (PR - Pwf_op))

        # Step 3 – iterate rate <-> non-Darcy-skin coupling to convergence.
        # v3: v2 took exactly two fixed passes through this loop regardless
        # of whether the rate estimate had actually settled down. Since the
        # non-Darcy skin term depends on q and q depends on skin, this is a
        # genuine fixed-point problem; iterating to a stated tolerance (or a
        # safety cap) is more numerically defensible than a hard-coded pass
        # count, and the iteration count/residual are now reported for
        # transparency.
        n_iter, converged = 0, False
        for n_iter in range(1, 11):
            sr2 = self.skin.total(p, q_est=qi)
            St_new = sr2["S_total"]
            J_new  = self.flow.PI(k, h, mu, Bo, re, St_new)
            qi_new = max(10.0, J_new * (PR - Pwf_op))
            resid  = abs(qi_new - qi) / max(qi, 1e-6)
            St, J, qi = St_new, J_new, qi_new
            if resid < 1e-4:
                converged = True
                break

        # Step 4 – decline forecast
        Di = float(np.clip(1.5 - 0.2 * np.log(max(k, 1)), 0.05, 0.85))
        b  = 1.2 if k > 100 else 1.5
        t_arr, q_arr, EUR, t_econ = self.flow.forecast(qi, Di, b, 20, 50)

        # Step 5 – NPV
        Ccap = 5e6 + nst * 150_000
        econ = self.flow.npv(t_arr, q_arr, 70, 8, 0.10, Ccap)

        # Step 6 – geomechanics
        eff_sh = self.geo.eff_stress(sig_h, Pp, biot)
        eff_sH = self.geo.eff_stress(sig_H, Pp, biot)
        DP_crit = self.geo.critical_drawdown(UCS, eff_sh, eff_sH, rw, phi_f, biot)
        DP_act  = PR - Pwf_op
        sand_rf = float(np.clip(DP_act / max(DP_crit, 1), 0, 5))

        # Step 7 – damage evolution (30 steps × 3 day)
        D_d = 0.05
        gp  = DP_act / max(re - rw, 1)
        gp_ref = 5.0
        tau_c  = UCS * 145.038 * 0.30
        for _ in range(30):
            tau = 0.15 * DP_act
            D_d = self.geo.damage_evolution(D_d, tau, tau_c, gp, gp_ref)

        # Step 8 – IPR for plotting
        Pwf_ipr, q_ipr = self.flow.ipr(PR, J, 0.70 * PR, 100)

        # Step 9 – FDI (simplified cluster heterogeneity index)
        VK  = float(p.get("Dykstra_Parsons", 0.40))
        FDI = float(np.clip(VK * 0.5 * (1 + 1 / (spf * pen / 18)), 0, 1))

        # Step 10 – SRV (simplified scaling)
        cs  = float(p.get("cluster_spacing_ft", 30))
        SRV = ncl * cs * h * pen / 12 * 0.8   # ft³ approximate
        SRV_MM = SRV / 1e6

        # Recovery factor
        pore_vol = phi * 5.615 * h * np.pi * re**2
        RF = float(np.clip(EUR / max(pore_vol * 0.65, 1), 0, 0.70))

        return {
            # skin
            "S_KT": sr2["S_KT"], "S_H": sr2["S_H"], "S_V": sr2["S_V"],
            "S_wb": sr2["S_wb"], "S_damage": sr2["S_damage"],
            "S_nondarcy": sr2["S_nondarcy"], "S_total": St,
            "CE": sr2["CE"], "D_coeff": sr2["D_coeff"],
            # flow
            "PI_stb_day_psi": J, "q_initial_stbd": qi,
            "EUR_MMbbl": EUR / 1e6, "recovery_factor": RF,
            "Di_yr": Di, "b_arps": b,
            # economics
            "NPV_MM": econ["NPV"] / 1e6, "ROI": econ["ROI"],
            "payback_yr": econ["payback"], "capex_MM": econ["Ccap"] / 1e6,
            "completion_cost_MM": nst * 150_000 / 1e6,
            # geomechanics
            "DP_crit_psi": DP_crit, "DP_act_psi": DP_act,
            "sand_risk": sand_rf, "damage_index": D_d,
            "eff_sigma_h_psi": eff_sh,
            # system
            "FDI": FDI, "SRV_MMft3": SRV_MM,
            "rate_skin_iterations": n_iter, "rate_skin_converged": converged,
            # arrays for plotting
            "t_arr": t_arr, "q_arr": q_arr,
            "Pwf_ipr": Pwf_ipr, "q_ipr": q_ipr,
        }

    # ── vectorised run over full DataFrame ───────────────────────────────────
    def run_dataset(self, df: pd.DataFrame) -> pd.DataFrame:
        COLS = ["PI_stb_day_psi","S_total","S_KT","S_damage","S_nondarcy",
                "q_initial_stbd","EUR_MMbbl","NPV_MM","ROI","payback_yr",
                "capex_MM","completion_cost_MM","sand_risk","damage_index",
                "FDI","SRV_MMft3","recovery_factor","DP_crit_psi","CE","Di_yr"]
        rows, failures = [], []
        for i, row in df.iterrows():
            try:
                r = self.run_well(row.to_dict())
                rows.append({c: r[c] for c in COLS})
            except Exception as exc:
                rows.append({c: np.nan for c in COLS})
                failures.append((i, str(exc)))
        if failures:
            import warnings
            warnings.warn(f"CoupledSimulator.run_dataset: {len(failures)}/{len(df)} rows failed "
                           f"and were filled with NaN. First failure at row {failures[0][0]}: "
                           f"{failures[0][1]}", RuntimeWarning)
        return pd.concat([df.reset_index(drop=True),
                          pd.DataFrame(rows)], axis=1)
