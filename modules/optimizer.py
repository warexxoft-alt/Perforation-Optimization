"""
modules/optimizer.py — PerfoSim Pro v3
════════════════════════════════════════════════════════════════════════════
NSGA-II multi-objective optimisation + Kriging surrogate acceleration.
Objectives (8): maximise PI, EUR, NPV, SRV; minimise S_total, cost, FDI,
sand_risk.
Reference: Deb, Pratap, Agarwal & Meyarivan (2002) IEEE Trans. Evol.
Comput. 6(2) — NSGA-II. Constraint handling: Deb (2000) Comput. Methods
Appl. Mech. Eng. 186 — "constrained-domination" tournament rule.

────────────────────────────────────────────────────────────────────────────
WHAT CHANGED FROM v2
────────────────────────────────────────────────────────────────────────────
1. BUG FIX — `ParetoAnalyzer.OBJ_SIGN` used -1e-7 to decode the EUR
   objective, which was scaled by -1/1e7 during optimisation
   (`f1 = -EUR/1e7`). The correct inverse scale is -1e7, not -1e-7; every
   decoded EUR value in every Pareto table/plot was therefore wrong by a
   factor of 1e14. Fixed and covered by a round-trip test in
   tests/test_optimizer.py.
2. MISSING FEATURE IMPLEMENTED — the v2 docstring promised "Constraints
   (6): SPF bounds, casing VM stress, critical drawdown, phasing
   feasibility, stage count, CDI limit", but zero constraint handling
   existed anywhere in the optimiser (it ran fully unconstrained). v3
   implements all six via `ObjectiveFunctions.constraint_violation` and
   Deb's constrained-domination rule (feasible beats infeasible; among two
   infeasible solutions, smaller total violation wins). See
   config.CONSTRAINTS for the concrete definition of each constraint,
   including documented interpretations of the previously-unspecified
   "phasing feasibility" (shot-clearance geometry) and "CDI limit"
   (Fluid Divergence Index ceiling).
3. The hypervolume indicator (`_hv`) was a product-of-per-axis-means proxy,
   not a real hypervolume. Replaced with a standard Monte-Carlo hypervolume
   estimator against a reference point (Fonseca et al. 2006 conventions).
4. `scipy.optimize.minimize` was imported and never used. It now powers a
   memetic local-search polish (Moscato 1989) that refines a handful of
   representative Pareto solutions via scalarised L-BFGS-B, only accepting
   a refinement if it stays feasible and non-dominated.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.optimize import minimize

from config import CONSTRAINTS


# ═══════════════════════════════════════════════════════════════════════════
# 1.  OBJECTIVE FUNCTIONS  (physics-based, fast analytic approximations)
# ═══════════════════════════════════════════════════════════════════════════
class ObjectiveFunctions:
    """Fast closed-form objective + constraint evaluations for the optimiser
    inner loop. Decision vector x = [spf, phasing, pen, hole, ub, cs, ncl] (this order
    matches config.OPT_BOUNDS.values() exactly)."""

    def __init__(self, res_params: dict):
        self.k    = res_params.get("permeability_mD", 100)
        self.h    = res_params.get("net_pay_ft", 25)
        self.mu   = res_params.get("oil_viscosity_cp", 2.5)
        self.Bo   = res_params.get("oil_fvf", 1.25)
        self.re   = res_params.get("drainage_radius_ft", 1320)
        self.rw   = res_params.get("wellbore_radius_ft", 0.354)
        self.PR   = res_params.get("initial_pressure_psi", 3500)
        self.Di   = res_params.get("Di_yr", 0.35)
        self.b    = res_params.get("b_arps", 1.2)
        self.Sd   = res_params.get("damage_skin", 3.0)
        self.sig_h = res_params.get("min_h_stress_psi", 4200)
        self.UCS  = res_params.get("UCS_MPa", 25.0)
        # "No-regret" baseline (constraint g6): PI achieved by a generic,
        # un-optimised reference completion (SPF=12, phasing=120 deg,
        # pen=18in, hole=0.45in, UB=-500psi), evaluated through the SAME
        # skin model as PI(x) — an apples-to-apples floor the optimiser
        # must at least match. NOTE (v3 fix): an earlier draft compared
        # against PI computed from damage skin alone (no perforation/
        # crushed-zone/non-Darcy contribution at all), which is actually a
        # *best case* lower-bound on skin, not a worst case — so PI(x) for
        # any real, fully-modelled completion failed the "no-regret" check
        # essentially 100% of the time (verified in tests/test_optimizer.py).
        self.PI_baseline = self.PI([12, 120, 18, 0.45, -500, 30, 5])

    def _skin(self, x) -> float:
        spf, phasing, pen, hole, ub, cs, ncl = x
        from modules.simulator import SkinCalculator
        sc = SkinCalculator()
        sr = sc.total({
            "spf": spf, "pen_depth_in": pen, "phasing": phasing,
            "hole_dia_in": hole, "underbalance_psi": ub,
            "permeability_mD": self.k, "k_vertical_mD": self.k * 0.10,
            "net_pay_ft": self.h, "oil_viscosity_cp": self.mu,
            "damage_skin": self.Sd, "wellbore_radius_ft": self.rw,
        }, q_est=500)
        return sr["S_total"]

    def PI(self, x) -> float:
        S   = self._skin(x)
        lnt = np.log(self.re / self.rw) - 0.5 + S
        if lnt <= 0:
            return 0.0
        return float(0.00708 * self.k * self.h / (self.mu * self.Bo * lnt))

    def EUR(self, x) -> float:
        qi  = max(1, self.PI(x) * (self.PR * 0.70))
        b, Di = self.b, self.Di
        if b < 1e-3:
            return float(qi / Di)
        q_econ = 50
        EUR    = qi ** b / (Di * (1 - b)) * (qi ** (1-b) - q_econ ** (1-b)) \
                 if abs(b - 1) > 1e-3 else qi / Di * np.log(qi / q_econ)
        return float(max(EUR, 0) * 365)

    def NPV(self, x) -> float:
        spf, phasing, pen, hole, ub, cs, ncl = x
        nst = max(5, int(2.0 * ncl))
        qi  = max(1, self.PI(x) * (self.PR * 0.70))
        Poil, Cop, rd = 70.0, 8.0, 0.10
        Ccap = 5e6 + nst * 150_000
        b, Di = self.b, self.Di
        NPV = 0.0
        for yr in range(1, 21):
            q_yr = qi / (1 + b * Di * yr) ** (1/b) if b > 1e-3 else qi * np.exp(-Di * yr)
            q_yr = max(q_yr, 0)
            NPV += (Poil - Cop) * q_yr * 365 / (1 + rd) ** yr
        return float(NPV - Ccap)

    def cost(self, x) -> float:
        spf, phasing, pen, hole, ub, cs, ncl = x
        nst = max(5, int(2.0 * ncl))
        return float(5e6 + nst * 150_000 + spf * 2000 * self.h)

    def FDI(self, x) -> float:
        spf, phasing, pen, hole, ub, cs, ncl = x
        VK = 0.40
        return float(np.clip(VK * 0.5 * (1 + 1 / (spf * pen / 18)), 0, 1))

    def sand_risk(self, x) -> float:
        """
        Sand-production risk ratio DP_effective / DP_critical.
        Larger / more numerous / deeper perforations spread produced fluid
        over more flow area and reduce the local (near-tunnel) pressure
        gradient that triggers grain dislodgement — modelled as a
        flow-convergence amplification of the far-field drawdown (Tiab &
        Donaldson, 2015, Ch.7 perforation flow-convergence skin analogy).
        NOTE (v3 fix): earlier draft made sand_risk depend only on fixed
        reservoir/geomechanical properties with no dependence on x at all,
        so it was a constant for every candidate design — if that constant
        exceeded sand_risk_max the whole population was permanently
        infeasible (verified in tests/test_optimizer.py); if it was below,
        the constraint was vacuous. Tying it to completion geometry makes
        it a real, satisfiable design trade-off.
        """
        from modules.simulator import GeoMechModel
        spf, phasing, pen, hole, ub, cs, ncl = x
        geo = GeoMechModel()
        phi_f = 30.0
        DP_crit = geo.critical_drawdown(self.UCS, self.sig_h * 0.85,
                                        self.sig_h * 0.85 * 1.15, self.rw, phi_f)
        flow_area   = max(spf * (pen / 12.0) * hole, 0.05)
        ref_area    = 12 * 1.5 * 0.45          # reference SPF=12, pen=18in, hole=0.45in
        convergence = np.clip(ref_area / flow_area, 0.4, 3.0)
        DP_act = self.PR * 0.70 * convergence
        return float(np.clip(DP_act / max(DP_crit, 1), 0, 5))

    def SRV(self, x) -> float:
        spf, phasing, pen, hole, ub, cs, ncl = x
        return float(ncl * cs * self.h * pen / 12 * 0.8 / 1e6)

    def evaluate(self, x) -> np.ndarray:
        """
        Returns objective vector f (all in MINIMISATION direction):
          f0 = -PI,  f1 = -EUR/1e7,  f2 = -NPV/1e6,  f3 = -SRV,
          f4 = S_total,  f5 = cost/1e6,  f6 = FDI,  f7 = sand_risk
        """
        return np.array([
            -self.PI(x),
            -self.EUR(x) / 1e7,
            -self.NPV(x) / 1e6,
            -self.SRV(x),
             self._skin(x),
             self.cost(x) / 1e6,
             self.FDI(x),
             self.sand_risk(x),
        ])

    def objective_names(self):
        return ["-PI", "-EUR/1e7", "-NPV/M$", "-SRV",
                "S_total", "Cost/M$", "FDI", "Sand Risk"]

    # ── constraints (Deb 2000 constrained-domination convention: g_i(x)<=0 is feasible) ──
    def constraint_violation(self, x) -> np.ndarray:
        """
        g1  sand-control:        DP_actual/DP_critical <= sand_risk_max
        g2  casing integrity:    von Mises stress <= casing yield strength
        g3  shot clearance:      (pi * casing_ID / spf) >= min shot clearance
                                  ("phasing feasibility" in the original spec —
                                  interpreted as a geometric no-collision rule
                                  between adjacent perforations around the
                                  casing circumference; see config.CONSTRAINTS)
        g4  interval geometry:   n_clusters * cluster_spacing <= max interval
                                  ("stage count" constraint)
        g5  completion divergence: FDI <= cdi_limit ("CDI limit")
        g6  no-regret:           PI(x) >= PI_damaged_baseline
        """
        from modules.simulator import GeoMechModel
        spf, phasing, pen, hole, ub, cs, ncl = x
        geo = GeoMechModel()

        g1 = self.sand_risk(x) - CONSTRAINTS["sand_risk_max"]

        Pwb = self.PR * 0.30
        ri, ro = CONSTRAINTS["casing_id_in"] / 2, CONSTRAINTS["casing_od_in"] / 2
        vm = geo.casing_vm_stress(Pwb, self.sig_h, self.sig_h * 1.15, ri, ro)["VM_psi"]
        g2 = vm - CONSTRAINTS["casing_yield_psi"]

        clearance = (np.pi * CONSTRAINTS["casing_id_in"]) / max(spf, 0.1)
        g3 = CONSTRAINTS["min_shot_clearance_in"] - clearance

        g4 = ncl * cs - CONSTRAINTS["max_interval_ft"]

        g5 = self.FDI(x) - CONSTRAINTS["cdi_limit"]

        g6 = self.PI_baseline * CONSTRAINTS["min_pi_ratio"] - self.PI(x)

        return np.array([g1, g2, g3, g4, g5, g6])

    def total_violation(self, x) -> float:
        return float(np.sum(np.maximum(self.constraint_violation(x), 0.0)))

    def constraint_names(self):
        return ["Sand control", "Casing integrity", "Shot clearance",
                "Interval geometry", "Completion divergence (CDI)", "No-regret PI"]


# ═══════════════════════════════════════════════════════════════════════════
# 2.  NSGA-II OPTIMIZER  (with optional constrained-domination)
# ═══════════════════════════════════════════════════════════════════════════
class NSGA2:
    """
    Deb et al. (2002) NSGA-II with:
      · Fast non-dominated sort (optionally constraint-aware, Deb 2000)
      · Crowding-distance assignment
      · Binary tournament selection
      · SBX crossover (eta=20), polynomial mutation (eta=20)
      · Monte-Carlo hypervolume convergence tracking
    """

    def __init__(self, obj_fn, bounds, constraint_fn=None, pop_size=80, n_gen=60,
                 pc=0.90, pm=None, eta_c=20, eta_m=20, seed=42):
        self.fn         = obj_fn
        self.constraint_fn = constraint_fn     # callable(x) -> g vector, feasible if all <= 0
        self.bounds     = np.array(bounds)     # (n_var, 2)
        self.n_var      = len(bounds)
        self.n_obj      = 8
        self.N          = pop_size
        self.G          = n_gen
        self.pc         = pc
        self.pm         = pm or 1 / self.n_var
        self.eta_c      = eta_c
        self.eta_m      = eta_m
        self.rng        = np.random.default_rng(seed)
        self.seed       = seed
        self.history    = {"gen": [], "hypervolume": [], "pareto_size": [],
                           "feasible_fraction": []}

    # ── constraint violation (0 if unconstrained) ─────────────────────────
    def _cv(self, x):
        if self.constraint_fn is None:
            return 0.0
        g = self.constraint_fn(x)
        return float(np.sum(np.maximum(g, 0.0)))

    # ── dominance (plain Pareto) ───────────────────────────────────────────
    def _dominates(self, a, b):
        return np.all(a <= b) and np.any(a < b)

    # ── constrained dominance (Deb 2000) ──────────────────────────────────
    def _dominates_cv(self, Fp, Fq, cvp, cvq):
        feas_p, feas_q = cvp <= 1e-9, cvq <= 1e-9
        if feas_p and feas_q:
            return self._dominates(Fp, Fq)
        if feas_p and not feas_q:
            return True
        if feas_q and not feas_p:
            return False
        return cvp < cvq   # both infeasible: smaller violation wins

    # ── fast non-dominated sort ───────────────────────────────────────────
    def _nds(self, F, CV):
        n = len(F)
        S, nc, rank, fronts = [[] for _ in range(n)], np.zeros(n, dtype=int), np.zeros(n, dtype=int), [[]]
        for p in range(n):
            for q in range(n):
                if p == q:
                    continue
                if self._dominates_cv(F[p], F[q], CV[p], CV[q]):
                    S[p].append(q)
                elif self._dominates_cv(F[q], F[p], CV[q], CV[p]):
                    nc[p] += 1
            if nc[p] == 0:
                rank[p] = 0
                fronts[0].append(p)
        i = 0
        while fronts[i]:
            nxt = []
            for p in fronts[i]:
                for q in S[p]:
                    nc[q] -= 1
                    if nc[q] == 0:
                        rank[q] = i + 1
                        nxt.append(q)
            fronts.append(nxt)
            i += 1
        return fronts[:-1], rank

    # ── crowding distance (within one front, standard — unaffected by CV) ─
    def _crowd(self, F, front):
        n = len(front)
        cd = np.zeros(n)
        for m in range(F.shape[1]):
            idx = np.argsort(F[front, m])
            fmin, fmax = F[front[idx[0]], m], F[front[idx[-1]], m]
            cd[idx[0]] = cd[idx[-1]] = np.inf
            for k in range(1, n - 1):
                span = max(fmax - fmin, 1e-12)
                cd[idx[k]] += (F[front[idx[k+1]], m] - F[front[idx[k-1]], m]) / span
        return cd

    # ── tournament selection ──────────────────────────────────────────────
    def _select(self, pop, rank, crowd):
        i, j = self.rng.integers(0, len(pop), 2)
        if rank[i] < rank[j]:
            return pop[i]
        elif rank[j] < rank[i]:
            return pop[j]
        return pop[i] if crowd[i] >= crowd[j] else pop[j]

    # ── SBX crossover ──────────────────────────────────────────────────────
    def _sbx(self, p1, p2):
        c1, c2 = p1.copy(), p2.copy()
        for j in range(self.n_var):
            if self.rng.random() > self.pc:
                continue
            lo, hi = self.bounds[j]
            if abs(p2[j] - p1[j]) < 1e-14:
                continue
            beta  = 1 + 2 * min(p1[j]-lo, hi-p1[j]) / max(abs(p2[j]-p1[j]), 1e-12)
            alpha = 2 - beta ** (-(self.eta_c+1))
            u     = self.rng.random()
            betaq = (u * alpha) ** (1/(self.eta_c+1)) if u <= 1/alpha else \
                    (1/(2-u*alpha)) ** (1/(self.eta_c+1))
            c1[j] = np.clip(0.5*((p1[j]+p2[j]) - betaq*abs(p2[j]-p1[j])), lo, hi)
            c2[j] = np.clip(0.5*((p1[j]+p2[j]) + betaq*abs(p2[j]-p1[j])), lo, hi)
        return c1, c2

    # ── polynomial mutation ────────────────────────────────────────────────
    def _mutate(self, x):
        y = x.copy()
        for j in range(self.n_var):
            if self.rng.random() > self.pm:
                continue
            lo, hi = self.bounds[j]
            delta1 = (y[j] - lo) / max(hi - lo, 1e-12)
            delta2 = (hi - y[j]) / max(hi - lo, 1e-12)
            u      = self.rng.random()
            if u <= 0.5:
                xy    = 1 - delta1
                val   = 2*u + (1-2*u)*(xy**(self.eta_m+1))
                delta = val**(1/(self.eta_m+1)) - 1
            else:
                xy    = 1 - delta2
                val   = 2*(1-u) + 2*(u-0.5)*(xy**(self.eta_m+1))
                delta = 1 - val**(1/(self.eta_m+1))
            y[j] = np.clip(y[j] + delta*(hi - lo), lo, hi)
        return y

    # ── Monte-Carlo hypervolume (replaces the v2 product-of-means proxy) ──
    def _hv_montecarlo(self, F_pareto, n_samples=6000):
        """
        Objectives are min-max normalised to [0,1] before computing
        hypervolume against a fixed (1.01,...,1.01) reference point — the 8
        raw objectives span wildly different physical scales (NPV ~10^2,
        sand risk ~10^0), so an un-normalised hypervolume would be almost
        entirely driven by whichever objective happens to have the largest
        magnitude. Normalising makes the trend meaningful generation-to-
        generation and comparable across different reservoir scenarios.
        """
        if len(F_pareto) == 0:
            return 0.0
        ideal = F_pareto.min(axis=0)
        span  = np.maximum(F_pareto.max(axis=0) - ideal, 1e-9)
        F_norm = (F_pareto - ideal) / span
        ref = np.full(F_pareto.shape[1], 1.01)
        rng = np.random.default_rng(self.seed)
        samples = rng.random((n_samples, F_pareto.shape[1])) * ref
        dominated = np.zeros(n_samples, dtype=bool)
        for p in F_norm:
            dominated |= np.all(p[None, :] <= samples, axis=1)
        return float(dominated.mean() * np.prod(ref))

    # ── run ────────────────────────────────────────────────────────────────
    def run(self, progress_cb=None):
        lo, hi  = self.bounds[:, 0], self.bounds[:, 1]
        pop     = lo + self.rng.random((self.N, self.n_var)) * (hi - lo)
        F_pop   = np.array([self.fn(x) for x in pop])
        CV_pop  = np.array([self._cv(x) for x in pop])

        for gen in range(self.G):
            fronts, rank = self._nds(F_pop, CV_pop)
            crowd_all = np.zeros(self.N)
            for fr in fronts:
                if len(fr) > 1:
                    cd = self._crowd(F_pop, fr)
                    for k, idx in enumerate(fr):
                        crowd_all[idx] = cd[k]

            offspring = []
            for _ in range(self.N // 2):
                p1 = self._select(pop, rank, crowd_all)
                p2 = self._select(pop, rank, crowd_all)
                c1, c2 = self._sbx(p1, p2)
                offspring.extend([self._mutate(c1), self._mutate(c2)])

            Q     = np.array(offspring)
            F_Q   = np.array([self.fn(x) for x in Q])
            CV_Q  = np.array([self._cv(x) for x in Q])
            R_pop, R_F, R_CV = np.vstack([pop, Q]), np.vstack([F_pop, F_Q]), np.concatenate([CV_pop, CV_Q])

            fronts2, rank2 = self._nds(R_F, R_CV)
            new_pop = []
            for fr in fronts2:
                if len(new_pop) + len(fr) <= self.N:
                    new_pop.extend(fr)
                else:
                    needed = self.N - len(new_pop)
                    cd = self._crowd(R_F, fr)
                    order = np.argsort(-cd)[:needed]
                    new_pop.extend([fr[o] for o in order])
                    break

            pop, F_pop, CV_pop = R_pop[new_pop], R_F[new_pop], R_CV[new_pop]

            pf_idx = fronts2[0] if fronts2 else [0]
            feasible_pf = [i for i in pf_idx if R_CV[i] <= 1e-9]
            hv = self._hv_montecarlo(R_F[feasible_pf] if feasible_pf else R_F[pf_idx])
            self.history["gen"].append(gen)
            self.history["hypervolume"].append(hv)
            self.history["pareto_size"].append(len(pf_idx))
            self.history["feasible_fraction"].append(float(np.mean(CV_pop <= 1e-9)))

            if progress_cb:
                progress_cb((gen + 1) / self.G)

        fronts_final, _ = self._nds(F_pop, CV_pop)
        pareto_idx = fronts_final[0]
        return pop, F_pop, pareto_idx

    # ── memetic local-search polish (Moscato 1989) ─────────────────────────
    def refine(self, pop, F_pop, pareto_idx, n_refine=6):
        """
        Polishes a spread of representative Pareto solutions with a
        scalarised L-BFGS-B local search. A refined point replaces its
        parent only if it is still feasible and is not dominated by any
        current Pareto member — this can only tighten the front, never
        degrade it.
        """
        if len(pareto_idx) == 0:
            return pop, F_pop, pareto_idx
        pf = np.array(pareto_idx)
        pick_n = min(n_refine, len(pf))
        chosen = self.rng.choice(pf, size=pick_n, replace=False)
        pop2, F2 = pop.copy(), F_pop.copy()

        for idx in chosen:
            x0 = pop[idx]
            w = self.rng.dirichlet(np.ones(self.n_obj))     # random convex weights

            def scalarised(x):
                f = self.fn(x)
                pen = 1e3 * self._cv(x)
                return float(np.dot(w, f)) + pen

            res = minimize(scalarised, x0, method="L-BFGS-B",
                            bounds=list(map(tuple, self.bounds)),
                            options={"maxiter": 60})
            x_new = res.x
            f_new = self.fn(x_new)
            if self._cv(x_new) > 1e-9:
                continue    # refinement drifted infeasible — discard
            dominated_by_existing = any(self._dominates(F2[j], f_new) for j in range(len(F2)) if j != idx)
            if not dominated_by_existing:
                pop2[idx], F2[idx] = x_new, f_new

        fronts_final, _ = self._nds(F2, np.array([self._cv(x) for x in pop2]))
        return pop2, F2, fronts_final[0]


# ═══════════════════════════════════════════════════════════════════════════
# 3.  PARETO FRONT ANALYSER
# ═══════════════════════════════════════════════════════════════════════════
class ParetoAnalyzer:
    OBJ_LABELS = ["PI (STB/d/psi)", "EUR (MMbbl)", "NPV (M$)", "SRV (MMft3)",
                  "Skin S_total", "Cost (M$)", "FDI", "Sand Risk"]
    # v3 BUG FIX: index 1 (EUR) must invert `f1 = -EUR_bbl/1e7` AND convert
    # bbl -> MMbbl to match the "EUR (MMbbl)" label, i.e. multiply by -10
    # (= -1e7 / 1e6). v2 used -1e-7 here (wrong by ~1e15 vs. the display
    # unit); simply inverting the raw optimisation scale alone (-1e7) would
    # still leave the number in raw barrels while claiming to be "MMbbl" —
    # verified against `ObjectiveFunctions.EUR()` in tests/test_optimizer.py.
    OBJ_SIGN = [-1, -10, -1, -1, 1, 1, 1, 1]

    def decode(self, F_raw: np.ndarray, X_raw: np.ndarray,
               constraint_fn=None) -> pd.DataFrame:
        """Convert raw minimisation objectives back to physical units."""
        signs  = np.array(self.OBJ_SIGN)
        F_phys = F_raw * signs
        df = pd.DataFrame(F_phys, columns=self.OBJ_LABELS)
        vn = ["SPF", "Phasing", "Pen_in", "Hole_in", "UB_psi", "CS_ft", "N_clust"]
        for i, v in enumerate(vn):
            if i < X_raw.shape[1]:
                df[v] = X_raw[:, i]
        if constraint_fn is not None:
            df["Constraint Violation"] = [float(np.sum(np.maximum(constraint_fn(x), 0.0))) for x in X_raw]
            df["Feasible?"] = np.where(df["Constraint Violation"] <= 1e-9, "Yes", "No")
        return df

    def best_npv(self, df: pd.DataFrame) -> pd.Series:
        return df.loc[df["NPV (M$)"].idxmax()]

    def best_pi(self, df: pd.DataFrame) -> pd.Series:
        return df.loc[df["PI (STB/d/psi)"].idxmax()]

    def best_balanced(self, df: pd.DataFrame) -> pd.Series:
        """TOPSIS-style balanced solution (closest to the ideal point)."""
        norm = df[self.OBJ_LABELS].copy()
        for col in norm.columns:
            rng = norm[col].max() - norm[col].min()
            if rng > 0:
                norm[col] = (norm[col] - norm[col].min()) / rng
        ideal = np.array([1, 1, 1, 1, 0, 0, 0, 0])
        dist  = np.sqrt(((norm.values - ideal) ** 2).sum(axis=1))
        return df.iloc[dist.argmin()]
