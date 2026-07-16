"""
modules/plots.py — PerfoSim Pro v3
All visualization functions: 2D → 9D + HF-specific plots.
Dimensional encoding:
  3-D → x,y,z        4-D → +colour       5-D → +size
  6-D → +symbol      7-D → +animation    8-D → +facet row/col
  9-D → parallel coordinates (9 axes) or SPLOM
"""
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from scipy.interpolate import griddata

from config import COLORS, CSCALES, PLOTLY_THEME

SYMB = ["circle","square","diamond","cross","x","triangle-up","triangle-down","star"]

# ─── helpers ─────────────────────────────────────────────────────────────────
def _size_norm(s: pd.Series, lo=4, hi=20):
    r = s.max() - s.min()
    if r < 1e-9: return np.full(len(s), (lo+hi)/2)
    return lo + (hi-lo)*(s - s.min())/r


# ══════════════════════════════════════════════════════════════════════════════
#  2-D
# ══════════════════════════════════════════════════════════════════════════════
def plot_2d_ipr(sim: dict, title="IPR Curve — Vogel / Darcy"):
    Pwf = np.array(sim["Pwf_ipr"]); q = np.array(sim["q_ipr"])
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=q, y=Pwf, mode="lines",
        name="Composite IPR", line=dict(color=COLORS["primary"], width=3)))
    fig.add_hline(y=float(max(Pwf))*0.30, line_dash="dot",
        line_color=COLORS["warning"], annotation_text="Operating BHP")
    fig.add_hline(y=float(max(Pwf)), line_dash="dash",
        line_color=COLORS["good"], annotation_text=f"Pᵣ = {max(Pwf):.0f} psi")
    fig.update_layout(template=PLOTLY_THEME, title=title, height=400,
        xaxis_title="Rate (STB/day)", yaxis_title="BHFP (psi)", hovermode="x unified")
    return fig

def plot_2d_decline(sim: dict, title="Hyperbolic Arps Decline"):
    t = np.array(sim["t_arr"]); q = np.array(sim["q_arr"])
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=t, y=q, mode="lines", fill="tozeroy",
        fillcolor="rgba(46,117,182,0.15)", line=dict(color=COLORS["primary"], width=2.5),
        name="q(t)"))
    fig.add_hline(y=50, line_dash="dot", line_color=COLORS["danger"],
        annotation_text="Economic Limit (50 STB/d)")
    fig.update_layout(template=PLOTLY_THEME, title=title, height=400,
        xaxis_title="Time (years)", yaxis_title="Oil Rate (STB/day)")
    return fig

def plot_2d_cumulative(sim: dict, title="Cumulative Production"):
    t = np.array(sim["t_arr"]); q = np.array(sim["q_arr"])
    cum = np.cumsum(q * 365 * np.diff(t, prepend=0)) / 1e3
    fig = go.Figure(go.Scatter(x=t, y=cum, mode="lines", fill="tozeroy",
        fillcolor="rgba(40,167,69,0.15)", line=dict(color=COLORS["good"], width=2.5)))
    fig.add_annotation(x=t[-1]*0.7, y=cum[-1]*0.85,
        text=f"EUR ≈ {cum[-1]:.0f} Mbbl", font=dict(color="white",size=13), showarrow=False)
    fig.update_layout(template=PLOTLY_THEME, title=title, height=400,
        xaxis_title="Time (years)", yaxis_title="Cumulative Oil (Mbbl)")
    return fig

def plot_2d_skin_vs_spf(df: pd.DataFrame, title="Skin Factor vs Shot Density"):
    fig = go.Figure()
    for ph in sorted(df["phasing"].unique()):
        sub = df[df["phasing"]==ph].groupby("spf")["S_total"].mean().reset_index()
        fig.add_trace(go.Scatter(x=sub["spf"], y=sub["S_total"],
            mode="lines+markers", name=f"φ={int(ph)}°", line=dict(width=2), marker=dict(size=8)))
    fig.add_hline(y=0, line_dash="dash", line_color=COLORS["good"], annotation_text="Zero Skin")
    fig.update_layout(template=PLOTLY_THEME, title=title, height=400,
        xaxis_title="SPF", yaxis_title="Skin S_total", hovermode="x unified")
    return fig

def plot_2d_heatmap(df: pd.DataFrame, title="Pearson Correlation Heatmap"):
    cols = ["spf","phasing","pen_depth_in","permeability_mD","net_pay_ft",
            "initial_pressure_psi","PI_stb_day_psi","S_total","EUR_MMbbl","NPV_MM","FDI","sand_risk"]
    av = [c for c in cols if c in df.columns]
    corr = df[av].corr()
    fig = go.Figure(go.Heatmap(z=corr.values, x=corr.columns, y=corr.columns,
        colorscale="RdBu", zmid=0,
        text=corr.round(2).values, texttemplate="%{text}",
        colorbar=dict(title="r")))
    fig.update_layout(template=PLOTLY_THEME, title=title, height=520)
    return fig

def plot_2d_pi_vs_pen(df: pd.DataFrame, title="PI vs Penetration Depth"):
    fig = px.scatter(df, x="pen_depth_in", y="PI_stb_day_psi",
        color="permeability_mD", color_continuous_scale=CSCALES["PI"], opacity=0.65,
        labels={"pen_depth_in":"Pen. Depth (in)","PI_stb_day_psi":"PI (STB/d/psi)"},
        title=title, template=PLOTLY_THEME)
    fig.update_layout(height=400)
    return fig

def plot_skin_waterfall(skin: dict, title="Skin Component Waterfall"):
    comps = [("Horizontal K&T","S_H"),("Vertical K&T","S_V"),
             ("Wellbore K&T","S_wb"),("Formation Damage","S_damage"),("Non-Darcy","S_nondarcy")]
    labels = [c[0] for c in comps]; vals = [skin.get(c[1],0) for c in comps]
    running = 0; bases, heights, cols_wf = [], [], []
    for v in vals:
        bases.append(running); heights.append(v)
        cols_wf.append(COLORS["primary"] if v >= 0 else COLORS["good"]); running += v
    fig = go.Figure(go.Bar(x=labels, y=heights, base=bases,
        marker_color=cols_wf, text=[f"{v:+.3f}" for v in vals], textposition="outside"))
    fig.add_hline(y=running, line_dash="dash", line_color=COLORS["danger"],
        annotation_text=f"S_total = {running:.2f}")
    fig.update_layout(template=PLOTLY_THEME, title=title, height=420,
        yaxis_title="Skin Contribution")
    return fig

def plot_damage_evolution(title="Damage Evolution (30 days)"):
    from modules.simulator import GeoMechModel
    geo = GeoMechModel(); steps = 50
    D_ub = np.zeros(steps+1); D_ob = np.zeros(steps+1)
    t = np.linspace(0, 30, steps+1); tau_c = 1500
    for i in range(steps):
        D_ub[i+1] = geo.damage_evolution(D_ub[i], 1200, tau_c, 3, 5)
        D_ob[i+1] = geo.damage_evolution(D_ob[i], 2200, tau_c, 8, 5)
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=t, y=D_ub, name="Underbalanced", line=dict(color=COLORS["good"],width=2.5)))
    fig.add_trace(go.Scatter(x=t, y=D_ob, name="Overbalanced",  line=dict(color=COLORS["danger"],width=2.5)))
    fig.add_hline(y=0.75, line_dash="dot", line_color=COLORS["warning"], annotation_text="D_crit = 0.75")
    fig.update_layout(template=PLOTLY_THEME, title=title, height=380,
        xaxis_title="Days", yaxis_title="Damage Index D_d")
    return fig


# ══════════════════════════════════════════════════════════════════════════════
#  3-D
# ══════════════════════════════════════════════════════════════════════════════
def plot_3d_surface(df: pd.DataFrame, title="3-D PI Response Surface (SPF × Pen × PI)"):
    try:
        xi = np.linspace(df["spf"].min(), df["spf"].max(), 40)
        yi = np.linspace(df["pen_depth_in"].min(), df["pen_depth_in"].max(), 40)
        XI, YI = np.meshgrid(xi, yi)
        ZI = griddata((df["spf"], df["pen_depth_in"]), df["PI_stb_day_psi"],
                       (XI, YI), method="linear")
        ZI = np.nan_to_num(ZI, nan=float(df["PI_stb_day_psi"].mean()))
    except Exception:
        xi = yi = ZI = np.zeros((10,10))
    fig = go.Figure(go.Surface(x=xi, y=yi, z=ZI,
        colorscale=CSCALES["PI"], opacity=0.88,
        colorbar=dict(title="PI (STB/d/psi)")))
    fig.update_layout(template=PLOTLY_THEME, title=title, height=520,
        scene=dict(xaxis_title="SPF", yaxis_title="Pen. Depth (in)",
                   zaxis_title="PI (STB/d/psi)"))
    return fig

def plot_3d_scatter(df: pd.DataFrame, title="3-D Reservoir Space"):
    fig = go.Figure(go.Scatter3d(
        x=df["permeability_mD"], y=df["net_pay_ft"], z=df["initial_pressure_psi"],
        mode="markers",
        marker=dict(size=4, color=df["PI_stb_day_psi"],
                    colorscale=CSCALES["PI"], opacity=0.75,
                    colorbar=dict(title="PI")),
        hovertemplate="k=%{x:.0f} mD<br>h=%{y:.1f} ft<br>Pᵢ=%{z:.0f} psi<extra></extra>"))
    fig.update_layout(template=PLOTLY_THEME, title=title, height=520,
        scene=dict(xaxis_title="k (mD)", yaxis_title="h (ft)", zaxis_title="Pᵢ (psi)"))
    return fig


# ══════════════════════════════════════════════════════════════════════════════
#  4-D  (x, y, z + colour)
# ══════════════════════════════════════════════════════════════════════════════
def plot_4d_scatter(df: pd.DataFrame, title="4-D: SPF × Pen × Phasing — colour=PI"):
    fig = go.Figure(go.Scatter3d(
        x=df["spf"], y=df["pen_depth_in"], z=df["phasing"], mode="markers",
        marker=dict(size=5, color=df["PI_stb_day_psi"],
                    colorscale=CSCALES["PI"], opacity=0.80,
                    colorbar=dict(title="PI (STB/d/psi)", x=1.02),
                    line=dict(width=0)),
        hovertemplate="SPF=%{x}<br>Pen=%{y:.1f} in<br>φ=%{z}°<br>PI=%{marker.color:.3f}<extra></extra>"))
    fig.update_layout(template=PLOTLY_THEME, title=title, height=560,
        scene=dict(xaxis_title="Dim1: SPF", yaxis_title="Dim2: Pen (in)",
                   zaxis_title="Dim3: Phasing (°)"),
        margin=dict(r=120))
    return fig


# ══════════════════════════════════════════════════════════════════════════════
#  5-D  (+size)
# ══════════════════════════════════════════════════════════════════════════════
def plot_5d_scatter(df: pd.DataFrame, title="5-D: k × h × Pᵢ — colour=PI, size=EUR"):
    sz = _size_norm(df["EUR_MMbbl"].clip(lower=0.1))
    fig = go.Figure(go.Scatter3d(
        x=df["permeability_mD"], y=df["net_pay_ft"], z=df["initial_pressure_psi"],
        mode="markers",
        marker=dict(size=sz, color=df["PI_stb_day_psi"],
                    colorscale=CSCALES["PI"], opacity=0.75,
                    colorbar=dict(title="PI"), line=dict(width=0)),
        customdata=np.stack([df["EUR_MMbbl"], df["PI_stb_day_psi"]], axis=-1),
        hovertemplate="k=%{x:.0f}|h=%{y:.1f}|Pᵢ=%{z:.0f}<br>PI=%{customdata[1]:.3f}|EUR=%{customdata[0]:.2f}<extra></extra>"))
    fig.add_annotation(x=0.98, y=0.02, xref="paper", yref="paper",
        text="Bubble size ∝ EUR (MMbbl)", showarrow=False,
        font=dict(color="lightgrey", size=11), align="right")
    fig.update_layout(template=PLOTLY_THEME, title=title, height=560,
        scene=dict(xaxis_title="Dim1: k (mD)", yaxis_title="Dim2: h (ft)",
                   zaxis_title="Dim3: Pᵢ (psi)"))
    return fig


# ══════════════════════════════════════════════════════════════════════════════
#  6-D  (+symbol)
# ══════════════════════════════════════════════════════════════════════════════
def plot_6d_scatter(df: pd.DataFrame, title="6-D: SPF × Pen × k — colour=PI, size=EUR, shape=Phasing"):
    phase_cats = sorted(df["phasing"].unique())
    sz = _size_norm(df["EUR_MMbbl"].clip(lower=0.1))
    fig = go.Figure()
    for i, ph in enumerate(phase_cats):
        m = df["phasing"] == ph; sub = df[m]
        fig.add_trace(go.Scatter3d(
            x=sub["spf"], y=sub["pen_depth_in"], z=sub["permeability_mD"],
            mode="markers", name=f"φ={int(ph)}°",
            marker=dict(size=sz[m], symbol=SYMB[i % len(SYMB)],
                        color=sub["PI_stb_day_psi"], colorscale=CSCALES["PI"],
                        opacity=0.80, line=dict(width=0),
                        colorbar=dict(title="PI") if i == 0 else {}),
            customdata=sub[["EUR_MMbbl","PI_stb_day_psi","phasing"]].values,
            hovertemplate="SPF=%{x}|Pen=%{y:.1f}|k=%{z:.0f}<br>"
                          "PI=%{customdata[1]:.3f}|EUR=%{customdata[0]:.2f}|φ=%{customdata[2]}°<extra></extra>"))
    fig.update_layout(template=PLOTLY_THEME, title=title, height=600,
        scene=dict(xaxis_title="Dim1: SPF", yaxis_title="Dim2: Pen (in)",
                   zaxis_title="Dim3: k (mD)"),
        legend=dict(title="Dim6: Phasing"))
    return fig


# ══════════════════════════════════════════════════════════════════════════════
#  7-D  (+animation)
# ══════════════════════════════════════════════════════════════════════════════
def plot_7d_animated(df: pd.DataFrame, title="7-D: + Animation by Underbalance Band"):
    df2 = df.copy()
    df2["ub_band"] = pd.cut(df2["underbalance_psi"],
        bins=[-1600,-1000,-500,0,400], labels=["<-1000","−1000→-500","−500→0",">0"])
    phase_cats = sorted(df2["phasing"].unique())
    sz = _size_norm(df2["EUR_MMbbl"].clip(lower=0.1))
    df2["sz"] = sz
    frames, bands = [], df2["ub_band"].cat.categories.tolist()
    for band in bands:
        sub = df2[df2["ub_band"]==band]; traces = []
        for i, ph in enumerate(phase_cats):
            m = sub["phasing"]==ph; s = sub[m]
            if len(s)==0: continue
            traces.append(go.Scatter3d(
                x=s["spf"], y=s["pen_depth_in"], z=s["permeability_mD"],
                mode="markers", name=f"φ={int(ph)}°",
                marker=dict(size=s["sz"], symbol=SYMB[i%len(SYMB)],
                            color=s["PI_stb_day_psi"], colorscale=CSCALES["PI"],
                            opacity=0.80, line=dict(width=0))))
        frames.append(go.Frame(data=traces, name=str(band)))
    fig = go.Figure(data=frames[0].data if frames else [], frames=frames)
    fig.update_layout(template=PLOTLY_THEME, title=title, height=620,
        scene=dict(xaxis_title="SPF", yaxis_title="Pen (in)", zaxis_title="k (mD)"),
        updatemenus=[dict(type="buttons", showactive=False, buttons=[
            dict(label="▶ Play",  method="animate",
                 args=[None, {"frame":{"duration":1200},"fromcurrent":True}]),
            dict(label="⏸ Pause", method="animate",
                 args=[[None],{"mode":"immediate"}])])],
        sliders=[dict(steps=[dict(args=[[f.name],{"frame":{"duration":800}}],
            label=f.name, method="animate") for f in frames],
            currentvalue=dict(prefix="Underbalance band: "), pad=dict(t=50))])
    return fig


# ══════════════════════════════════════════════════════════════════════════════
#  8-D  (+facet rows & cols)
# ══════════════════════════════════════════════════════════════════════════════
def plot_8d_faceted(df: pd.DataFrame, title="8-D: Faceted by k-band × Phasing-band"):
    df2 = df.copy()
    df2["k_band"]  = pd.qcut(df2["permeability_mD"], 2, labels=["Low-k","High-k"])
    df2["ph_band"] = df2["phasing"].apply(lambda v: "Low φ (60-90°)" if v<=90 else "High φ (120-180°)")
    df2["ub_cat"]  = (df2["underbalance_psi"]<-500).map({True:"Underbal.",False:"Overbal."})
    sz = _size_norm(df2["EUR_MMbbl"].clip(lower=0.1)); df2["sz"] = sz
    kb = ["Low-k","High-k"]; pb = ["Low φ (60-90°)","High φ (120-180°)"]
    fig = make_subplots(rows=2, cols=2,
        subplot_titles=[f"{r} | {c}" for r in kb for c in pb],
        specs=[[{"type":"scatter3d"}]*2]*2,
        horizontal_spacing=0.01, vertical_spacing=0.06)
    ub_syms = {"Underbal.":"circle","Overbal.":"square"}
    first = True
    for ri,kb_ in enumerate(kb):
        for ci,pb_ in enumerate(pb):
            sub = df2[(df2["k_band"]==kb_)&(df2["ph_band"]==pb_)]
            for ub,sym in ub_syms.items():
                s = sub[sub["ub_cat"]==ub]
                if len(s)==0: continue
                fig.add_trace(go.Scatter3d(
                    x=s["spf"], y=s["pen_depth_in"], z=s["net_pay_ft"],
                    mode="markers", name=f"{ub}", showlegend=first,
                    marker=dict(size=s["sz"], symbol=sym, opacity=0.80,
                                color=s["PI_stb_day_psi"],
                                colorscale=CSCALES["PI"], line=dict(width=0))),
                    row=ri+1, col=ci+1)
                first = False
    fig.update_layout(template=PLOTLY_THEME, title=title, height=720,
        legend=dict(title="Dim8: UB"), showlegend=True)
    return fig


# ══════════════════════════════════════════════════════════════════════════════
#  9-D  (parallel coordinates — 9 axes)
# ══════════════════════════════════════════════════════════════════════════════
def plot_9d_parallel(df: pd.DataFrame, title="9-D Parallel Coordinates — colour=NPV"):
    cols   = ["spf","pen_depth_in","phasing","permeability_mD","net_pay_ft",
              "initial_pressure_psi","UCS_MPa","S_total","PI_stb_day_psi"]
    labels = ["Dim1 SPF","Dim2 Pen(in)","Dim3 φ(°)","Dim4 k(mD)","Dim5 h(ft)",
              "Dim6 Pᵢ(psi)","Dim7 UCS(MPa)","Dim8 Skin","Dim9 PI"]
    av  = [c for c in cols if c in df.columns]
    lav = [labels[cols.index(c)] for c in av]
    cv  = df["NPV_MM"] if "NPV_MM" in df.columns else df[av[0]]
    dims = [dict(label=l, values=df[c], range=[df[c].min(), df[c].max()])
            for c,l in zip(av,lav)]
    fig = go.Figure(go.Parcoords(
        line=dict(color=cv, colorscale=CSCALES["NPV"], showscale=True,
                  colorbar=dict(title="NPV (M$)")),
        dimensions=dims))
    fig.update_layout(template=PLOTLY_THEME, title=title, height=520)
    return fig

def plot_9d_splom(df: pd.DataFrame, title="9-D Scatter Plot Matrix (SPLOM)"):
    cols   = ["spf","pen_depth_in","phasing","permeability_mD","net_pay_ft",
              "initial_pressure_psi","UCS_MPa","S_total","PI_stb_day_psi"]
    labels = ["SPF","Pen","φ","k","h","Pᵢ","UCS","Skin","PI"]
    av = [c for c in cols if c in df.columns]
    lav= [labels[cols.index(c)] for c in av]
    fig = px.scatter_matrix(df[av].rename(columns=dict(zip(av,lav))),
        dimensions=lav,
        color=df["PI_stb_day_psi"] if "PI_stb_day_psi" in df.columns else None,
        color_continuous_scale=CSCALES["PI"], template=PLOTLY_THEME, title=title)
    fig.update_traces(diagonal_visible=False, showupperhalf=False,
                      marker=dict(size=2.5, opacity=0.55))
    fig.update_layout(height=720, coloraxis_colorbar=dict(title="PI"))
    return fig


# ══════════════════════════════════════════════════════════════════════════════
#  HYDRAULIC FRACTURING PLOTS
# ══════════════════════════════════════════════════════════════════════════════
def plot_hf_geometry(hf_res: dict, title="Fracture Geometry"):
    xf = hf_res["xf_ft"]; Hf = hf_res["Hf_ft"]
    w  = hf_res["w_max_in"]
    # Elliptical cross-section
    theta = np.linspace(0, 2*np.pi, 200)
    x_ell = (w/2) * np.cos(theta)
    y_ell = (Hf/2) * np.sin(theta)
    fig = make_subplots(rows=1, cols=2,
        subplot_titles=["Fracture Cross-Section (Wellbore)", "Plan View (One Wing)"])
    fig.add_trace(go.Scatter(x=x_ell, y=y_ell, fill="toself",
        fillcolor="rgba(46,117,182,0.30)", line=dict(color=COLORS["primary"],width=2),
        name="Width profile"), row=1, col=1)
    fig.add_trace(go.Scatter(x=[0,0], y=[-Hf/2,Hf/2],
        line=dict(color=COLORS["danger"],dash="dash",width=2), name="Wellbore"), row=1,col=1)
    # Plan view
    xf_arr = np.linspace(0, xf, 100)
    w_prof = (w/2) * (1 - xf_arr/xf)**0.5
    fig.add_trace(go.Scatter(x=xf_arr, y=w_prof, fill="tozeroy",
        fillcolor="rgba(46,117,182,0.25)", line=dict(color=COLORS["primary"],width=2),
        name="Width (in)"), row=1, col=2)
    fig.add_trace(go.Scatter(x=xf_arr, y=-w_prof, fill="tozeroy",
        fillcolor="rgba(46,117,182,0.25)", line=dict(color=COLORS["primary"],width=2),
        showlegend=False), row=1, col=2)
    fig.update_xaxes(title_text="Width (in)", row=1, col=1)
    fig.update_yaxes(title_text="Height (ft)", row=1, col=1)
    fig.update_xaxes(title_text="Fracture Half-Length (ft)", row=1, col=2)
    fig.update_yaxes(title_text="Half-Width (in)", row=1, col=2)
    fig.update_layout(template=PLOTLY_THEME, title=title, height=440)
    return fig

def plot_hf_3d(hf_res: dict, title="3-D Fracture Visualisation"):
    xf = hf_res["xf_ft"]; Hf = hf_res["Hf_ft"]; w = hf_res["w_max_in"]/12
    # Create fracture ellipsoid surface
    u = np.linspace(0, np.pi, 30); v = np.linspace(0, 2*np.pi, 30)
    U, V = np.meshgrid(u, v)
    Xs = xf * np.sin(U) * np.cos(V)
    Ys = w  * np.sin(U) * np.sin(V)
    Zs = (Hf/2) * np.cos(U)
    fig = go.Figure()
    fig.add_trace(go.Surface(x=Xs, y=Ys, z=Zs,
        colorscale=CSCALES["hf"], opacity=0.70,
        colorbar=dict(title="Width proxy"),
        showscale=True))
    # Wellbore line
    fig.add_trace(go.Scatter3d(
        x=[0,0], y=[0,0], z=[-Hf/2, Hf/2],
        mode="lines", line=dict(color=COLORS["danger"],width=8), name="Wellbore"))
    fig.update_layout(template=PLOTLY_THEME, title=title, height=560,
        scene=dict(xaxis_title="Length (ft)", yaxis_title="Width (ft)",
                   zaxis_title="Height (ft)"))
    return fig

def plot_hf_time_evolution(evo: dict, title="Fracture Growth vs Time (PKN)"):
    fig = make_subplots(rows=1, cols=2,
        subplot_titles=["Half-Length vs Time","Width & Net Pressure vs Time"])
    t = evo["t_min"]; xf = evo["xf_ft"]
    fig.add_trace(go.Scatter(x=t, y=xf, mode="lines",
        line=dict(color=COLORS["primary"],width=2.5), name="xf (ft)"), row=1, col=1)
    fig.add_trace(go.Scatter(x=t, y=evo["w_max_in"], mode="lines",
        line=dict(color=COLORS["accent"],width=2), name="w_max (in)"), row=1, col=2)
    fig.add_trace(go.Scatter(x=t, y=evo["p_net_psi"], mode="lines",
        line=dict(color=COLORS["danger"],width=2,dash="dash"), name="p_net (psi)"), row=1, col=2)
    fig.update_xaxes(title_text="Pump Time (min)", row=1, col=1)
    fig.update_xaxes(title_text="Pump Time (min)", row=1, col=2)
    fig.update_yaxes(title_text="xf (ft)", row=1, col=1)
    fig.update_layout(template=PLOTLY_THEME, title=title, height=420)
    return fig

def plot_pump_schedule(sched_df: pd.DataFrame, title="HF Pump Schedule"):
    fig = make_subplots(rows=2, cols=1, shared_xaxes=True,
        subplot_titles=["Proppant Concentration (ppg)", "Slurry Volume (bbl)"])
    fig.add_trace(go.Bar(x=sched_df["name"], y=sched_df["prop_conc_ppg"],
        marker_color=sched_df["prop_conc_ppg"], marker_colorscale="YlOrRd",
        name="Conc (ppg)"), row=1, col=1)
    fig.add_trace(go.Bar(x=sched_df["name"], y=sched_df["volume_bbl"],
        marker_color=COLORS["primary"], name="Volume (bbl)"), row=2, col=1)
    fig.update_layout(template=PLOTLY_THEME, title=title, height=480,
        xaxis_tickangle=-30, showlegend=True)
    return fig

def plot_dfit_gfunction(dfit_df: pd.DataFrame, title="G-Function DFIT Analysis"):
    clos_psi = dfit_df["P_closure_psi"].iloc[0]
    fig = make_subplots(rows=1, cols=2,
        subplot_titles=["P vs G-Function", "Pressure Decline vs Shut-In Time"])
    fig.add_trace(go.Scatter(x=dfit_df["G"], y=dfit_df["P_bhp_psi"],
        mode="lines", line=dict(color=COLORS["primary"],width=2.5), name="P(G)"), row=1, col=1)
    fig.add_hline(y=clos_psi, line_dash="dash", line_color=COLORS["danger"],
        annotation_text=f"P_closure={clos_psi:.0f} psi", row=1, col=1)
    fig.add_trace(go.Scatter(x=dfit_df["t_shut_min"], y=dfit_df["P_bhp_psi"],
        mode="lines", line=dict(color=COLORS["accent"],width=2), name="P(t)"), row=1, col=2)
    fig.update_xaxes(title_text="G-Function", row=1, col=1)
    fig.update_xaxes(title_text="Shut-In Time (min)", row=1, col=2)
    fig.update_yaxes(title_text="BHP (psi)", row=1, col=1)
    fig.update_layout(template=PLOTLY_THEME, title=title, height=430)
    return fig

def plot_frac_conductivity_vs_stress(proppant_types: list,
    stress_range=(1000,8000), title="Fracture Conductivity vs Closure Stress"):
    from modules.hf_design import ProppantDesign
    stresses = np.linspace(*stress_range, 100)
    fig = go.Figure()
    for ptype in proppant_types:
        kfw = [ProppantDesign.fracture_conductivity(0.25, s, ptype)["kf_wf_mDft"]
               for s in stresses]
        fig.add_trace(go.Scatter(x=stresses, y=kfw, mode="lines",
            name=ptype, line=dict(width=2)))
    fig.update_layout(template=PLOTLY_THEME, title=title, height=420,
        xaxis_title="Closure Stress (psi)", yaxis_title="kf·wf (mD-ft)",
        yaxis_type="log")
    return fig

def plot_fcd_sensitivity(xf_table: pd.DataFrame, title="PI Ratio vs xf and Fcd"):
    fig = go.Figure()
    for Fcd in xf_table["Fcd"].unique():
        sub = xf_table[xf_table["Fcd"]==Fcd]
        fig.add_trace(go.Scatter(x=sub["xf_ft"], y=sub["PI_ratio"],
            mode="lines+markers", name=f"Fcd={Fcd}",
            line=dict(width=2), marker=dict(size=7)))
    fig.add_hline(y=1.6, line_dash="dot", line_color=COLORS["warning"],
        annotation_text="Fcd optimal=1.6")
    fig.update_layout(template=PLOTLY_THEME, title=title, height=420,
        xaxis_title="Fracture Half-Length xf (ft)",
        yaxis_title="PI Ratio (J_frac / J_unfrac)")
    return fig

def plot_log_view(df: pd.DataFrame, title="Well Log Cross-Plot"):
    cols = [c for c in ["GR_api","Rt_ohmm","NPHI_pu","RHOB_gcc","DT_usft","porosity"]
            if c in df.columns]
    if "depth_ft" not in df.columns or len(cols) == 0:
        return go.Figure().update_layout(template=PLOTLY_THEME,
            title="No log curves found", height=300)
    n_tracks = len(cols)
    fig = make_subplots(rows=1, cols=n_tracks,
        subplot_titles=cols, shared_yaxes=True)
    colors = [COLORS["primary"],COLORS["danger"],COLORS["good"],COLORS["accent"],"#9B59B6",COLORS["secondary"]]
    for i, col in enumerate(cols):
        fig.add_trace(go.Scatter(x=df[col], y=df["depth_ft"],
            mode="lines", line=dict(color=colors[i%len(colors)],width=1.5),
            name=col, fill="tozerox"), row=1, col=i+1)
    fig.update_yaxes(autorange="reversed", title_text="Depth (ft)", row=1, col=1)
    fig.update_layout(template=PLOTLY_THEME, title=title, height=650,
        showlegend=False)
    return fig

def plot_grid_slice(grid: dict, prop="poro", title="Reservoir Grid Slice"):
    data = grid.get("data", {})
    dims = grid.get("dims", (10,10,10))
    NI, NJ, NK = dims
    arr_key = {"poro":"PORO","permx":"PERMX","swat":"SWAT"}.get(prop, "PORO")
    if arr_key not in data:
        arr_key = list(data.keys())[0] if data else None
    if arr_key is None:
        return go.Figure().update_layout(template=PLOTLY_THEME, title="No grid data", height=300)
    arr = np.array(data[arr_key])
    try:
        vol = arr[:NI*NJ*NK].reshape(NJ, NI, NK)
        mid_k = NK // 2
        slc   = vol[:,:,mid_k]
    except Exception:
        side = int(np.sqrt(len(arr))); slc = arr[:side**2].reshape(side, side)
    fig = go.Figure(go.Heatmap(z=slc, colorscale="Viridis",
        colorbar=dict(title=arr_key)))
    fig.update_layout(template=PLOTLY_THEME, title=f"{title} — {arr_key} (k=mid)",
        height=450, xaxis_title="I direction", yaxis_title="J direction")
    return fig

def plot_image_analysis(img_arr: np.ndarray, title="Core / CT Scan Image"):
    fig = make_subplots(rows=1, cols=2,
        subplot_titles=["Image", "Intensity Histogram"])
    arr = np.array(img_arr)
    if arr.ndim == 1:
        side = int(np.sqrt(len(arr))); arr = arr[:side**2].reshape(side,side)
    fig.add_trace(go.Heatmap(z=arr, colorscale="Gray", showscale=False), row=1, col=1)
    fig.add_trace(go.Histogram(x=arr.ravel(), nbinsx=60,
        marker_color=COLORS["primary"], name="Intensity"), row=1, col=2)
    fig.update_layout(template=PLOTLY_THEME, title=title, height=430)
    return fig


# ══════════════════════════════════════════════════════════════════════════════
#  OPTIMIZATION / PARETO
# ══════════════════════════════════════════════════════════════════════════════
def plot_pareto_2d(pf: pd.DataFrame, x_col, y_col, title="Pareto Front (2-D)"):
    fig = go.Figure(go.Scatter(x=pf[x_col], y=pf[y_col], mode="markers",
        marker=dict(color=COLORS["accent"],size=9,opacity=0.85,
                    line=dict(color="white",width=0.5)), name="Pareto"))
    xn = (pf[x_col]-pf[x_col].min())/(pf[x_col].max()-pf[x_col].min()+1e-9)
    yn = (pf[y_col]-pf[y_col].min())/(pf[y_col].max()-pf[y_col].min()+1e-9)
    best = ((1-xn)**2+(1-yn)**2).idxmin()
    fig.add_trace(go.Scatter(x=[pf[x_col][best]], y=[pf[y_col][best]],
        mode="markers", name="Best balanced",
        marker=dict(color=COLORS["danger"],size=14,symbol="star",
                    line=dict(color="white",width=1))))
    fig.update_layout(template=PLOTLY_THEME, title=title, height=440,
        xaxis_title=x_col, yaxis_title=y_col)
    return fig

def plot_pareto_3d(pf: pd.DataFrame, title="Pareto Front (3-D)"):
    x = pf.get("PI (STB/d/psi)", pf.iloc[:,0])
    y = pf.get("EUR (MMbbl)",     pf.iloc[:,1])
    z = pf.get("NPV (M$)",        pf.iloc[:,2])
    c = pf.get("Skin S_total",    pf.iloc[:,4])
    fig = go.Figure(go.Scatter3d(x=x, y=y, z=z, mode="markers",
        marker=dict(size=7, color=c, colorscale=CSCALES["skin"],
                    colorbar=dict(title="Skin"), opacity=0.85,
                    line=dict(color="white",width=0.3)),
        hovertemplate="PI=%{x:.2f}<br>EUR=%{y:.1f}<br>NPV=%{z:.1f}<extra></extra>"))
    fig.update_layout(template=PLOTLY_THEME, title=title, height=560,
        scene=dict(xaxis_title="PI (STB/d/psi)", yaxis_title="EUR (MMbbl)",
                   zaxis_title="NPV (M$)"))
    return fig

def plot_training_history(hist: dict, title="PINNs Training History"):
    e = hist.get("epoch",[])
    fig = make_subplots(rows=1, cols=2, subplot_titles=["Loss Components","Val R²"])
    for name, col in [("Data",COLORS["primary"]),("Physics",COLORS["warning"]),("Total",COLORS["danger"])]:
        key = f"loss_{name.lower()}"
        if key in hist:
            fig.add_trace(go.Scatter(x=e, y=hist[key], name=name,
                line=dict(color=col)), row=1, col=1)
    if "val_r2" in hist:
        fig.add_trace(go.Scatter(x=e, y=hist["val_r2"], name="Val R²",
            line=dict(color=COLORS["good"])), row=1, col=2)
        fig.add_hline(y=0.95, line_dash="dot", line_color="orange",
            annotation_text="R²=0.95", row=1, col=2)
    fig.update_layout(template=PLOTLY_THEME, title=title, height=400)
    return fig

def plot_predicted_vs_actual(yte, ypred, model_name="Model", target="PI", p_value=None, r2_ci=None):
    from sklearn.metrics import r2_score
    lo=min(yte.min(),ypred.min()); hi=max(yte.max(),ypred.max())
    r2 = r2_score(yte, ypred)
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=yte, y=ypred, mode="markers",
        marker=dict(color=COLORS["primary"],size=5,opacity=0.65), name="Pred"))
    fig.add_trace(go.Scatter(x=[lo,hi], y=[lo,hi], mode="lines",
        line=dict(color=COLORS["danger"],dash="dash"), name="Perfect"))
    lines = [f"R² = {r2:.4f}"]
    if r2_ci: lines.append(f"95% CI [{r2_ci.get('lower',float('nan')):.3f}, {r2_ci.get('upper',float('nan')):.3f}]")
    if p_value is not None: lines.append(f"corr. p = {p_value:.2g}")
    fig.add_annotation(x=lo+0.1*(hi-lo), y=hi-0.05*(hi-lo)*len(lines),
        text="<br>".join(lines), showarrow=False, font=dict(color="white",size=13),
        align="left", bgcolor="rgba(0,0,0,0.35)", bordercolor="white", borderwidth=1)
    fig.update_layout(template=PLOTLY_THEME, height=420,
        title=f"{model_name} — Predicted vs Actual ({target})",
        xaxis_title=f"Actual {target}", yaxis_title=f"Predicted {target}")
    return fig

def plot_feature_importance(imp, title="Feature Importance", errors=None):
    if isinstance(imp, pd.Series):
        names, vals = imp.index, imp.values
    elif isinstance(imp, pd.DataFrame):
        names, vals = imp["feature"], imp["importance_mean"]
        errors = imp["importance_std"].values if "importance_std" in imp.columns else errors
    else:
        raise TypeError("plot_feature_importance expects a pandas Series or DataFrame")
    fig = go.Figure(go.Bar(x=vals, y=names, orientation="h",
        error_x=dict(type="data", array=errors, visible=errors is not None, color=COLORS["muted"]) if errors is not None else None,
        marker=dict(color=vals, colorscale="Blues", line=dict(width=0))))
    fig.update_layout(template=PLOTLY_THEME, title=title, height=400,
        xaxis_title="Importance", yaxis=dict(autorange="reversed"))
    return fig

def plot_sobol_indices(sobol_df: pd.DataFrame, title="Sobol Sensitivity Indices"):
    fig = go.Figure()
    fig.add_trace(go.Bar(name="S1 (first-order)", x=sobol_df["feature"],
        y=sobol_df["S1"], marker_color=COLORS["primary"]))
    fig.add_trace(go.Bar(name="ST (total-order)", x=sobol_df["feature"],
        y=sobol_df["ST"], marker_color=COLORS["accent"]))
    fig.update_layout(template=PLOTLY_THEME, title=title, barmode="group", height=400,
        yaxis_title="Sensitivity Index")
    return fig

def plot_benchmark_bar(study_val, bm_dict, metric="PI", title="Benchmark Comparison"):
    labels = list(bm_dict.keys()) + ["This Study"]
    vals   = list(bm_dict.values()) + [study_val]
    colors = [COLORS["primary"]]*len(bm_dict) + [COLORS["danger"]]
    fig = go.Figure(go.Bar(x=labels, y=vals, marker_color=colors,
        text=[f"{v:.3f}" for v in vals], textposition="outside"))
    fig.update_layout(template=PLOTLY_THEME, title=title, height=420,
        yaxis_title=metric, xaxis_tickangle=-30)
    return fig

def plot_radar_benchmark(cats, study_vals, bench_vals, title="Performance Radar"):
    c = cats + [cats[0]]
    fig = go.Figure()
    fig.add_trace(go.Scatterpolar(r=study_vals+[study_vals[0]], theta=c,
        fill="toself", name="This Study", line_color=COLORS["primary"]))
    fig.add_trace(go.Scatterpolar(r=bench_vals+[bench_vals[0]], theta=c,
        fill="toself", name="Industry Benchmark", line_color=COLORS["accent"], opacity=0.70))
    fig.update_layout(polar=dict(radialaxis=dict(visible=True)),
        template=PLOTLY_THEME, title=title, height=460, showlegend=True)
    return fig

# ══════════════════════════════════════════════════════════════════════════════
#  STATISTICAL VALIDATION PLOTS (v3 — interactive companions to modules.pub_plots)
# ══════════════════════════════════════════════════════════════════════════════
def plot_cv_comparison(scores_dict: dict, metric_name="R2", pairwise_df=None,
                       title="Cross-Validated Model Comparison"):
    """Interactive box+strip plot of per-fold CV scores, one box per model,
    with significance brackets drawn between pairs flagged by
    `stats_engine.ModelComparison` (Holm-corrected p<0.05)."""
    fig = go.Figure()
    palette = [COLORS["primary"], COLORS["accent"], COLORS["secondary"], COLORS["good"], COLORS["danger"]]
    names = list(scores_dict.keys())
    for i, name in enumerate(names):
        vals = np.asarray(scores_dict[name])
        fig.add_trace(go.Box(y=vals, name=name, marker_color=palette[i % len(palette)],
                             boxpoints="all", jitter=0.4, pointpos=0, width=0.5))
    if pairwise_df is not None and len(pairwise_df) > 0:
        sig_col = [c for c in pairwise_df.columns if c.startswith("p-value (")]
        if sig_col:
            sig_col = sig_col[0]
            ymax = max(np.max(v) for v in scores_dict.values())
            ymin = min(np.min(v) for v in scores_dict.values())
            step = 0.08 * (ymax - ymin + 1e-9)
            lvl = 1
            for _, row in pairwise_df.iterrows():
                if row[sig_col] < 0.05:
                    ia, ib = names.index(row["Model A"]), names.index(row["Model B"])
                    y = ymax + step * lvl
                    fig.add_shape(type="line", x0=ia, x1=ib, y0=y, y1=y, line=dict(color="white", width=1.5))
                    fig.add_annotation(x=(ia+ib)/2, y=y, text="*", showarrow=False, font=dict(size=18, color="white"))
                    lvl += 1
    fig.update_layout(template=PLOTLY_THEME, title=title, height=460,
        yaxis_title=f"Cross-validated {metric_name}", showlegend=False)
    return fig


def plot_residuals_interactive(y_true, y_pred, model_name="Model"):
    """2-panel interactive residual check: residuals-vs-fitted + histogram
    with a normal overlay. The full 4-panel camera-ready version (with a
    Q-Q plot and scale-location panel) lives in modules.pub_plots for
    600 DPI export."""
    y_true, y_pred = np.asarray(y_true), np.asarray(y_pred)
    resid = y_true - y_pred
    fig = make_subplots(rows=1, cols=2, subplot_titles=["Residuals vs. Fitted", "Residual Distribution"])
    fig.add_trace(go.Scatter(x=y_pred, y=resid, mode="markers",
        marker=dict(color=COLORS["primary"], size=6, opacity=0.6), name="Residual"), row=1, col=1)
    fig.add_hline(y=0, line_dash="dash", line_color=COLORS["danger"], row=1, col=1)
    fig.add_trace(go.Histogram(x=resid, marker_color=COLORS["primary"], opacity=0.75,
        histnorm="probability density", name="Residuals"), row=1, col=2)
    xs = np.linspace(resid.min(), resid.max(), 100)
    from scipy import stats as _st
    fig.add_trace(go.Scatter(x=xs, y=_st.norm.pdf(xs, resid.mean(), resid.std(ddof=1)),
        line=dict(color=COLORS["accent"], width=2.5), name="Normal fit"), row=1, col=2)
    fig.update_layout(template=PLOTLY_THEME, height=420, showlegend=False,
        title=f"{model_name}: Residual Check")
    fig.update_xaxes(title_text="Fitted value", row=1, col=1)
    fig.update_yaxes(title_text="Residual", row=1, col=1)
    fig.update_xaxes(title_text="Residual", row=1, col=2)
    return fig


def plot_correlation_heatmap_stats(corr: pd.DataFrame, pvals: pd.DataFrame = None,
                                   title="Feature Correlation (significance-annotated)"):
    """Correlation heatmap with a Pearson-r colour scale and p<0.05 markers
    — the interactive companion to `pub_plots.fig_correlation_heatmap`."""
    text = corr.round(2).astype(str)
    if pvals is not None:
        for i in corr.index:
            for j in corr.columns:
                if i != j and pvals.loc[i, j] < 0.05:
                    text.loc[i, j] = text.loc[i, j] + "*"
    fig = go.Figure(go.Heatmap(z=corr.values, x=list(corr.columns), y=list(corr.index),
        colorscale="RdBu", zmid=0, zmin=-1, zmax=1, text=text.values, texttemplate="%{text}",
        colorbar=dict(title="Pearson r")))
    fig.update_layout(template=PLOTLY_THEME, title=title + "  (* p&lt;0.05)", height=500)
    return fig


def plot_learning_curve(lc_df: pd.DataFrame, model_name="Model"):
    """Train/validation R² vs. training-set size — bias/variance diagnostic."""
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=lc_df["train_size"], y=lc_df["train_mean"], mode="lines+markers",
        name="Training score", line=dict(color=COLORS["primary"])))
    fig.add_trace(go.Scatter(x=lc_df["train_size"], y=lc_df["train_mean"]+lc_df["train_std"],
        mode="lines", line=dict(width=0), showlegend=False, hoverinfo="skip"))
    fig.add_trace(go.Scatter(x=lc_df["train_size"], y=lc_df["train_mean"]-lc_df["train_std"],
        mode="lines", line=dict(width=0), fill="tonexty",
        fillcolor="rgba(46,117,182,0.15)", showlegend=False, hoverinfo="skip"))
    fig.add_trace(go.Scatter(x=lc_df["train_size"], y=lc_df["val_mean"], mode="lines+markers",
        name="Validation score", line=dict(color=COLORS["accent"])))
    fig.add_trace(go.Scatter(x=lc_df["train_size"], y=lc_df["val_mean"]+lc_df["val_std"],
        mode="lines", line=dict(width=0), showlegend=False, hoverinfo="skip"))
    fig.add_trace(go.Scatter(x=lc_df["train_size"], y=lc_df["val_mean"]-lc_df["val_std"],
        mode="lines", line=dict(width=0), fill="tonexty",
        fillcolor="rgba(255,127,14,0.15)", showlegend=False, hoverinfo="skip"))
    fig.update_layout(template=PLOTLY_THEME, height=440, title=f"{model_name}: Learning Curve",
        xaxis_title="Training set size", yaxis_title="R² score")
    return fig

# ══════════════════════════════════════════════════════════════════════════════
#  GENERIC N-DIMENSIONAL RESULTS VISUALIZATION (v3 — works on ANY dataframe:
#  raw dataset, ML predictions, NSGA-II Pareto front, or Sobol sensitivity
#  results, not just the hard-coded raw-dataset columns above)
# ══════════════════════════════════════════════════════════════════════════════
def plot_nd_bubble(df: pd.DataFrame, x: str, y: str, size_col: str = None,
                   color_col: str = None, title: str = None, size_label: str = None):
    """4-5D bubble: x, y position + size (4th dim) + colour (5th dim)."""
    d = df
    if size_col is not None and (d[size_col] < 0).any():
        # Plotly's marker-size encoding rejects negative values outright;
        # some legitimately-signed quantities (e.g. a Sobol ST-S1 gap that
        # dips slightly negative from Monte-Carlo noise around a
        # theoretical floor of zero) need clipping for display purposes.
        d = d.copy()
        d[size_col] = d[size_col].clip(lower=0)
    kwargs = dict(x=x, y=y, color=color_col, size=size_col,
                  color_continuous_scale="Viridis" if color_col and pd.api.types.is_numeric_dtype(d[color_col]) else None,
                  template=PLOTLY_THEME, height=520,
                  title=title or f"{y} vs {x}" + (f" — size={size_col}" if size_col else "") + (f", colour={color_col}" if color_col else ""))
    fig = px.scatter(d, **kwargs)
    fig.update_traces(marker=dict(line=dict(width=0.5, color="white"), opacity=0.75))
    return fig


def plot_nd_animated_bubble(df: pd.DataFrame, x: str, y: str, size_col: str,
                            color_col: str, frame_col: str, title: str = None):
    """7D: x, y, size, colour, + animation frame. `frame_col` should be a
    low-cardinality (binned/categorical) column."""
    d = df.copy()
    if frame_col in d.columns and pd.api.types.is_numeric_dtype(d[frame_col]) and d[frame_col].nunique() > 8:
        d[frame_col] = pd.qcut(d[frame_col], 6, duplicates="drop").astype(str)
    xr = [d[x].min() - 0.05 * (d[x].max() - d[x].min()), d[x].max() + 0.05 * (d[x].max() - d[x].min())]
    yr = [d[y].min() - 0.05 * (d[y].max() - d[y].min()), d[y].max() + 0.05 * (d[y].max() - d[y].min())]
    fig = px.scatter(d, x=x, y=y, size=size_col, color=color_col, animation_frame=frame_col,
                     color_continuous_scale="Turbo", template=PLOTLY_THEME, height=560,
                     range_x=xr, range_y=yr,
                     title=title or f"7-D: {y} vs {x} — size={size_col}, colour={color_col}, frame={frame_col}")
    return fig


def plot_nd_faceted(df: pd.DataFrame, x: str, y: str, color_col: str = None,
                    facet_col: str = None, facet_row: str = None, title: str = None):
    """8D: x, y, colour + 2 faceting dimensions (row & column small multiples)."""
    d = df.copy()
    for c in (facet_col, facet_row):
        if c and c in d.columns and pd.api.types.is_numeric_dtype(d[c]) and d[c].nunique() > 4:
            d[c] = pd.qcut(d[c], 3, duplicates="drop").astype(str)
    fig = px.scatter(d, x=x, y=y, color=color_col, facet_col=facet_col, facet_row=facet_row,
                     color_continuous_scale="Plasma", template=PLOTLY_THEME, height=620,
                     title=title or f"8-D Faceted: {y} vs {x}")
    fig.update_traces(marker=dict(size=7, opacity=0.7, line=dict(width=0.3, color="white")))
    return fig


def plot_nd_parallel(df: pd.DataFrame, cols: list, color_col: str = None, title: str = None):
    """9D-style parallel coordinates over an arbitrary list of numeric columns
    (e.g. all 8 NSGA-II objectives + feasibility as colour)."""
    d = df[cols].copy()
    dims = [dict(label=c, values=d[c]) for c in cols]
    line = dict(color=df[color_col], colorscale="Viridis", showscale=True,
               colorbar=dict(title=color_col)) if color_col and color_col in df.columns and \
        pd.api.types.is_numeric_dtype(df[color_col]) else dict(color=COLORS["primary"])
    fig = go.Figure(go.Parcoords(dimensions=dims, line=line))
    fig.update_layout(template=PLOTLY_THEME, height=520,
                      title=title or f"Parallel Coordinates ({len(cols)} dimensions)")
    return fig


def plot_nd_splom(df: pd.DataFrame, cols: list, color_col: str = None, title: str = None):
    """Generic scatter-plot matrix over an arbitrary list of columns."""
    fig = px.scatter_matrix(df, dimensions=cols, color=color_col,
                            color_continuous_scale="Viridis", template=PLOTLY_THEME, height=750,
                            title=title or f"Scatter Plot Matrix ({len(cols)} dimensions)")
    fig.update_traces(diagonal_visible=False, showupperhalf=False,
                      marker=dict(size=4, opacity=0.6, line=dict(width=0.2, color="white")))
    return fig


def plot_ml_predictions_multid(results: dict, models: list = None):
    """
    Multi-model predicted-vs-actual as a single N-dimensional view: x=actual,
    y=predicted, colour=model identity, size=|residual| — lets you compare
    every trained model's prediction quality (and where it struggles) in one
    chart instead of flipping between separate pages.
    """
    models = models or [k for k in results if k in ("linear", "ensemble", "kriging", "pinn")]
    from modules.ml_engine import MODEL_LABELS
    rows = []
    for k in models:
        if k not in results:
            continue
        yte, yp = results[k]["yte"], results[k]["y_pred_te"]
        for a, p in zip(yte, yp):
            rows.append({"Actual": a, "Predicted": p, "Model": MODEL_LABELS.get(k, k),
                        "Abs. Residual": abs(a - p)})
    d = pd.DataFrame(rows)
    if len(d) == 0:
        return go.Figure().update_layout(template=PLOTLY_THEME, title="No model results available", height=300)
    lo, hi = min(d["Actual"].min(), d["Predicted"].min()), max(d["Actual"].max(), d["Predicted"].max())
    fig = px.scatter(d, x="Actual", y="Predicted", color="Model", size="Abs. Residual",
                     template=PLOTLY_THEME, height=560, opacity=0.7,
                     title="Multi-Model Predicted vs. Actual — colour=model, size=|residual|")
    fig.add_trace(go.Scatter(x=[lo, hi], y=[lo, hi], mode="lines",
                             line=dict(color="white", dash="dash"), name="Perfect", showlegend=True))
    return fig


def plot_pareto_parallel_full(pareto_df: pd.DataFrame, obj_cols: list, color_col: str = "NPV (M$)"):
    """Full 8-objective (9D-style, colour-coded) parallel-coordinates view of
    the NSGA-II Pareto front — every objective at once instead of one 2D/3D
    projection at a time."""
    return plot_nd_parallel(pareto_df, obj_cols, color_col=color_col,
                            title="NSGA-II Pareto Front — All 8 Objectives (Parallel Coordinates)")
