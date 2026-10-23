"""Compute and Capability.

01 Training compute    log scatter, frontier envelope, one trend or two fitted eras
02 Same compute        score vs compute by release year (explore + animated replay)
03 Iso-capability map  fitted MMLU surface over release date x compute
04 Decomposition       waterfall: compute vs methods vs model-specific
05 Two readings
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
import analysis as A  # noqa: E402

st.set_page_config(page_title="Compute and Capability", page_icon="📈", layout="wide")

DATA = ROOT / "data" / "processed"
BLUE, ORANGE, AQUA = "#2a78d6", "#eb6834", "#1baf7a"
GREY, GREY_LIGHT = "#a3a199", "#d9d7d0"
INK, INK_2, INK_3 = "#0b0b0b", "#52514e", "#8a8983"
RAMP = ["#86b6ef", "#3987e5", "#1c5cab", "#0d366b"]   # older -> newer
SEQ = [[0, "#eef5fd"], [0.2, "#cde2fb"], [0.4, "#9ec5f4"], [0.6, "#5598e7"],
       [0.8, "#256abf"], [1, "#0d366b"]]
GRID = "#ecebe7"
SUP = str.maketrans("-0123456789", "⁻⁰¹²³⁴⁵⁶⁷⁸⁹")

st.markdown(
    """
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&family=Source+Serif+4:opsz,wght@8..60,600;8..60,700&display=swap');
html, body, [class*="css"], .stMarkdown { font-family: 'Inter', sans-serif; }
.block-container { padding-top: 2rem; max-width: 1280px; }
.hero { padding: 2.4rem 2.6rem; border-radius: 18px; color: #fff;
        background: radial-gradient(120% 140% at 0% 0%, #2a78d6 0%, #1c5cab 40%, #0d366b 100%); }
.hero h1 { font-family: 'Source Serif 4', serif; color: #fff; font-size: 2.7rem; font-weight: 700;
           line-height: 1.08; margin: 0 0 .6rem; letter-spacing: -.01em; }
.hero p { font-size: 1.05rem; opacity: .9; max-width: 720px; margin: 0; }
.kpis { display: grid; grid-template-columns: repeat(4, 1fr); gap: 14px; margin: 18px 0 8px; }
.kpi { background: #fcfcfb; border: 1px solid #e6e4de; border-radius: 14px; padding: 16px 18px; }
.kpi .v { font-family: 'Source Serif 4', serif; font-size: 2.1rem; font-weight: 700; color: #0b0b0b; line-height: 1.05; }
.kpi .l { font-size: .86rem; color: #52514e; margin-top: 4px; }
.kpi .s { font-size: .74rem; color: #8a8983; margin-top: 6px; }
.kpi.accent { border-left: 5px solid #eb6834; }
.act { font-size: .8rem; font-weight: 700; color: #2a78d6; margin-top: 2.8rem; letter-spacing: .08em; }
.act-title { font-family: 'Source Serif 4', serif; font-size: 1.9rem; font-weight: 700; color: #0b0b0b;
             margin: .05rem 0 .3rem; letter-spacing: -.01em; }
.lede { font-size: 1rem; color: #52514e; max-width: 860px; }
.callout { background: #f4f8fe; border-left: 4px solid #2a78d6; border-radius: 8px;
           padding: 12px 16px; color: #0b0b0b; margin: 8px 0; font-size: .93rem; }
.callout.orange { background: #fdf3ee; border-left-color: #eb6834; }
.side { background: #fcfcfb; border: 1px solid #e6e4de; border-radius: 14px; padding: 14px 16px; margin-bottom: 10px; }
.side h4 { margin: 0 0 4px; font-size: .86rem; color: #52514e; font-weight: 600; }
.big { font-family: 'Source Serif 4', serif; font-size: 1.7rem; font-weight: 700; color: #0b0b0b; line-height: 1.15; }
.muted { color: #8a8983; font-size: .78rem; }
@media (max-width: 900px) { .kpis { grid-template-columns: repeat(2, 1fr); } .hero h1 { font-size: 1.9rem; } }
</style>
""",
    unsafe_allow_html=True,
)


# ================================================================== data + models
@st.cache_data
def load():
    compute = pd.read_csv(DATA / "compute_over_time.csv", parse_dates=["date"])
    compute["t"] = A.frac_year(compute["date"])
    mmlu = pd.read_csv(DATA / "mmlu_vs_compute.csv")
    eci = pd.read_csv(DATA / "eci_vs_compute.csv")
    for df in (mmlu, eci):
        df["t"] = A.frac_year(df["release"])
    return compute, mmlu, eci


def subset(domains, frontier_only):
    c = load()[0]
    c = c[c["Domain"].isin(domains)]
    return c[c["Frontier model"]] if frontier_only else c


@st.cache_data
def trend(domains, frontier_only):
    c = subset(domains, frontier_only)
    return (A.fit_trend(c["t"], c["Training compute (FLOP)"]) if len(c) >= 5 else None), len(c)


@st.cache_data
def eras(domains, frontier_only):
    c = subset(domains, frontier_only)
    return A.find_breakpoint(c["t"], c["Training compute (FLOP)"]) if len(c) >= 20 else None


@st.cache_data
def capability_fit(kind):
    _, mmlu, eci = load()
    df = mmlu if kind == "mmlu" else eci
    return A.fit_capability(df["score"], df["compute"], df["t"], kind, 2021 if kind == "mmlu" else 2023)


def fmt_flop(x, unit=True):
    exp = int(np.floor(np.log10(x)))
    return f"{x / 10**exp:.1f} × 10{str(exp).translate(SUP)}" + (" FLOP" if unit else "")


def pow10(e):
    return f"10{str(int(e)).translate(SUP)}"


def to_date(t):
    return pd.to_datetime((np.asarray(t) - 1970) * 365.25, unit="D")


def style(fig, x_title, y_title, height=520):
    fig.update_layout(
        height=height, plot_bgcolor="white", paper_bgcolor="white",
        font=dict(family="Inter, sans-serif", color=INK, size=13),
        margin=dict(l=10, r=20, t=40, b=10),
        legend=dict(orientation="h", y=1.07, x=0, font=dict(color=INK_2)),
        hoverlabel=dict(bgcolor="white", font_color=INK, bordercolor=GREY_LIGHT),
    )
    fig.update_xaxes(title=x_title, gridcolor=GRID, linecolor=GREY_LIGHT, zeroline=False, title_font_color=INK_2)
    fig.update_yaxes(title=y_title, gridcolor=GRID, linecolor=GREY_LIGHT, zeroline=False, title_font_color=INK_2)
    return fig


def act(n, title, lede=""):
    st.markdown(f'<div class="act">{n:02d}</div><div class="act-title">{title}</div>'
                + (f'<div class="lede">{lede}</div>' if lede else ""), unsafe_allow_html=True)


def side(title, value, note=""):
    return (f'<div class="side"><h4>{title}</h4><div class="big">{value}</div>'
            + (f'<div class="muted">{note}</div>' if note else "") + "</div>")


compute, mmlu, eci = load()
DEFAULT_DOMAINS = ("Language", "Multimodal", "Vision")
base_trend, _ = trend(DEFAULT_DOMAINS, False)
fit_m = capability_fit("mmlu")
share = A.algo_share(fit_m["rate"], base_trend["growth"])
share_lo = A.algo_share(fit_m["rate_ci"][0], base_trend["growth"])
share_hi = A.algo_share(fit_m["rate_ci"][1], base_trend["growth"])
path80 = A.cheapest_path(mmlu, 0.80)
drop80 = path80["compute"].iloc[0] / path80["compute"].iloc[-1]
mm_names = mmlu.sort_values("t")["Model version"].tolist()
st.session_state.setdefault("pick_a", "Gopher (280B)")
st.session_state.setdefault("pick_b", "phi-4")

# ================================================================== hero
st.markdown(
    """<div class="hero"><h1>Bigger chips or smarter methods?</h1>
<p>478 notable AI models, 2010–2026. How much of the gain came from training compute,
and how much from better algorithms.</p></div>""", unsafe_allow_html=True)
st.markdown(
    f"""<div class="kpis">
  <div class="kpi"><div class="v">{base_trend['growth']:.1f}×</div>
    <div class="l">more training compute every year</div>
    <div class="s">95% CI {base_trend['growth_ci'][0]:.1f}–{base_trend['growth_ci'][1]:.1f}× · 2010–2026</div></div>
  <div class="kpi"><div class="v">{fit_m['rate']:.1f}×</div>
    <div class="l">compute saved per year by better methods</div>
    <div class="s">95% CI {fit_m['rate_ci'][0]:.1f}–{fit_m['rate_ci'][1]:.1f}× · MMLU model</div></div>
  <div class="kpi accent"><div class="v">≈{share:.0%}</div>
    <div class="l">of yearly progress comes from algorithms</div>
    <div class="s">range {share_lo:.0%}–{share_hi:.0%} · the rest is compute</div></div>
  <div class="kpi"><div class="v">{drop80:.0f}×</div>
    <div class="l">less compute to reach 80% on MMLU</div>
    <div class="s">{path80['Model version'].iloc[0]} → {path80['Model version'].iloc[-1]}</div></div>
</div>""", unsafe_allow_html=True)

# ================================================================== 01
act(1, "Training compute, 2010–2026", "Log scale: each grid line is 10× more compute.")

f1, f2, f3 = st.columns([2.2, 1.6, 0.8])
domains = tuple(sorted(f1.multiselect("Domain", sorted(compute["Domain"].unique()),
                                      default=list(DEFAULT_DOMAINS))))
mode = f2.radio("Trend", ["One trend", "Two eras (fitted break)"], index=1, horizontal=True)
frontier_only = f3.toggle("Frontier only", value=False,
                          help="Epoch AI flag: in the top 10 by training compute when released.")
view = subset(domains, frontier_only)
tr, n_fit = trend(domains, frontier_only)
er = eras(domains, frontier_only) if mode.startswith("Two") else None

chart, panel = st.columns([3.2, 1])
with chart:
    if tr is None:
        st.warning("Not enough models. Add a domain.")
    else:
        fig = go.Figure()
        t0, t1 = view["t"].min(), view["t"].max()
        if er is None:
            xs = np.linspace(t0, t1, 80)
            band = 10 ** (tr["boots"][:, [1]] + tr["boots"][:, [0]] * xs)
            lo, hi = np.percentile(band, [2.5, 97.5], axis=0)
            fig.add_trace(go.Scatter(x=np.r_[to_date(xs), to_date(xs[::-1])], y=np.r_[hi, lo[::-1]],
                                     fill="toself", fillcolor="rgba(235,104,52,0.12)", line=dict(width=0),
                                     hoverinfo="skip", name="95% CI"))
        else:
            fig.add_vrect(x0=to_date([er["year"]])[0], x1=to_date([t1 + 0.3])[0],
                          fillcolor=BLUE, opacity=0.05, line_width=0)
            fig.add_vline(x=to_date([er["year"]])[0], line=dict(color=INK_3, width=1, dash="dot"))
        for flag, colour, name, size in [(False, GREY_LIGHT, "Other notable models", 7),
                                         (True, BLUE, "Frontier models", 9)]:
            part = view[view["Frontier model"] == flag]
            fig.add_trace(go.Scatter(
                x=part["date"], y=part["Training compute (FLOP)"], mode="markers", name=name,
                marker=dict(size=size, color=colour, line=dict(width=1, color="white")),
                customdata=np.stack([part["Model"], part["Organization"].fillna("–"),
                                     part["Training compute (FLOP)"].map(fmt_flop), part["Domain"]], axis=1),
                hovertemplate="<b>%{customdata[0]}</b><br>%{customdata[1]} · %{customdata[3]}"
                              "<br>%{x|%b %Y}<br>%{customdata[2]}<extra></extra>"))
        env = view.sort_values("date").assign(top=lambda d: d["Training compute (FLOP)"].cummax())
        fig.add_trace(go.Scatter(x=env["date"], y=env["top"], mode="lines", line_shape="hv",
                                 line=dict(color=INK, width=1.5), name="Largest model so far", hoverinfo="skip"))
        if er is None:
            xs = np.linspace(t0, t1, 80)
            fig.add_trace(go.Scatter(x=to_date(xs), y=10 ** (tr["intercept"] + tr["slope"] * xs), mode="lines",
                                     line=dict(color=ORANGE, width=2.5, dash="dash"),
                                     name=f"Trend: doubles every {tr['doubling']:.1f} months", hoverinfo="skip"))
        else:
            for (a_, b_), coef, dbl in [((t0, er["year"]), er["before"], er["doubling_before"]),
                                         ((er["year"], t1), er["after"], er["doubling_after"])]:
                xs = np.linspace(a_, b_, 40)
                fig.add_trace(go.Scatter(x=to_date(xs), y=10 ** np.polyval(coef, xs), mode="lines",
                                         line=dict(color=ORANGE, width=3), showlegend=False, hoverinfo="skip"))
                mid = (a_ + b_) / 2
                fig.add_annotation(x=to_date([mid])[0], y=np.polyval(coef, mid), text=f"<b>×2 every {dbl:.1f} mo</b>",
                                   ax=0 if mid > er["year"] else 40, ay=45, arrowcolor=ORANGE,
                                   font=dict(color=ORANGE, size=12), bgcolor="rgba(255,255,255,.9)")
            fig.add_annotation(x=to_date([er["year"]])[0], y=0, yref="paper", text=f" break {er['year']:.1f}",
                               showarrow=False, xanchor="left", yshift=10, font=dict(size=11, color=INK_3))
        for name, ax, ay in [("AlexNet", 30, -40), ("Transformer", -30, -50), ("GPT-3 175B (davinci)", -60, -40),
                             ("GPT-4 (Mar 2023)", -60, -40), ("GPT-6 Astra", -70, 10)]:
            r = view[view["Model"] == name]
            if len(r):
                r = r.iloc[0]
                fig.add_annotation(x=r["date"], y=np.log10(r["Training compute (FLOP)"]),
                                   text=f"<b>{name.split(' (')[0]}</b><br>{r['date'].year}", ax=ax, ay=ay,
                                   arrowcolor=INK_3, arrowwidth=1, font=dict(size=11, color=INK_2),
                                   bgcolor="rgba(255,255,255,.85)")
        fig.update_yaxes(type="log", exponentformat="power", dtick=2)
        st.plotly_chart(style(fig, "Publication date", "Training compute (FLOP, log scale)", 560),
                        use_container_width=True)

with panel:
    if tr is not None:
        if er is None:
            st.markdown(side("Doubling time", f"{tr['doubling']:.1f} months",
                             f"95% CI {tr['doubling_ci'][0]:.1f}–{tr['doubling_ci'][1]:.1f} · 1,000 bootstraps")
                        + side("Growth per year", f"{tr['growth']:.1f}×", f"{n_fit} models"),
                        unsafe_allow_html=True)
        else:
            st.markdown(side(f"Before {er['year']:.0f}", f"×2 every {er['doubling_before']:.1f} mo",
                             f"{er['growth_before']:.1f}× per year")
                        + side(f"After {er['year']:.0f}", f"×2 every {er['doubling_after']:.1f} mo",
                               f"{er['growth_after']:.1f}× per year")
                        + side("Break year", f"{er['year']:.1f}", "least-squares search, 2012–2022"),
                        unsafe_allow_html=True)
        st.markdown('<div class="callout">GPT-4 (2023): about <b>45 million×</b> the compute '
                    'of AlexNet (2012).</div>', unsafe_allow_html=True)

# ================================================================== 02
act(2, "Same compute, better scores",
    "Lines: fitted score by release year. The gap between two lines is the gain at <b>fixed compute</b>.")
tab_explore, tab_replay = st.tabs(["Explore", "Replay 2021 → 2024"])

with tab_explore:
    b1, b2 = st.columns([1.4, 2.6])
    bench = b1.radio("Benchmark", ["MMLU (2021–2024)", "Epoch Capabilities Index (2023–2026)"],
                     label_visibility="collapsed")
    kind = "mmlu" if bench.startswith("MMLU") else "eci"
    df = mmlu if kind == "mmlu" else eci
    fit = capability_fit(kind)
    years_all = sorted(df["year"].unique())
    colours = dict(zip(years_all, RAMP[-len(years_all):]))
    years_on = b2.multiselect("Release years shown", years_all, default=years_all)
    scale = 100 if kind == "mmlu" else 1

    chart, panel = st.columns([3.2, 1])
    with chart:
        fig = go.Figure()
        xs = np.logspace(np.log10(df["compute"].min()), np.log10(df["compute"].max()), 60)
        curve_rows = {}
        for yr in years_on:
            fig.add_trace(go.Scatter(x=xs, y=A.predict(fit, xs, yr + 0.5) * scale, mode="lines",
                                     line=dict(color=colours[yr], width=2), showlegend=False, hoverinfo="skip"))
            part = df[df["year"] == yr]
            curve_rows[len(fig.data)] = part
            fig.add_trace(go.Scatter(
                x=part["compute"], y=part["score"] * scale, mode="markers", name=str(yr),
                marker=dict(size=11, color=colours[yr], line=dict(width=1.5, color="white")),
                customdata=np.stack([part["Model version"], part["organization"].fillna("–"),
                                     part["compute"].map(fmt_flop), part["release"]], axis=1),
                hovertemplate="<b>%{customdata[0]}</b><br>%{customdata[1]} · %{customdata[3]}<br>"
                              "%{customdata[2]}<br>Score: %{y:.1f}<extra></extra>"))
        labels = (["Gopher (280B)", "phi-4", "gpt-4-0314", "DeepSeek-V3"] if kind == "mmlu"
                  else ["GPT-4 (Mar 2023)", "DeepSeek V4.1 Flash", "GPT-6 Astra", "Llama 2-7B"])
        for name in labels:
            r = df[(df["Model version"] == name) & df["year"].isin(years_on)]
            if len(r):
                r = r.iloc[0]
                fig.add_annotation(x=np.log10(r["compute"]), y=r["score"] * scale, text=name, showarrow=False,
                                   yshift=15, font=dict(size=11, color=INK_2), bgcolor="rgba(255,255,255,.8)")
        if kind == "mmlu":
            fig.add_hline(y=25, line=dict(color=GREY, dash="dot"), annotation_text="Random guessing (25%)",
                          annotation_position="bottom right", annotation_font_color=INK_3)
            fig.update_yaxes(range=[15, 95])
        fig.update_xaxes(type="log", exponentformat="power", dtick=1)
        fig.update_layout(clickmode="event+select")
        event = st.plotly_chart(
            style(fig, "Training compute (FLOP, log scale)",
                  "MMLU score (%, 5-shot)" if kind == "mmlu" else "Epoch Capabilities Index", 540),
            use_container_width=True, on_select="rerun", selection_mode="points", key=f"sel_{kind}")
        if kind == "mmlu":
            st.caption("Click a dot to send it to 04 as the 'to' model. Shift-click two dots to set both.")
            picked = []
            for p in (event.selection.points if event and event.selection else []):
                rows = curve_rows.get(p.get("curve_number"))
                if rows is not None and p.get("point_index") is not None:
                    picked.append(rows.iloc[p["point_index"]])
            sig = tuple(r["Model version"] for r in picked)
            if picked and sig != st.session_state.get("last_pick"):
                st.session_state["last_pick"] = sig
                picked = sorted(picked, key=lambda r: r["t"])
                if len(picked) == 1:
                    st.session_state["pick_b"] = picked[0]["Model version"]
                else:
                    st.session_state["pick_a"] = picked[0]["Model version"]
                    st.session_state["pick_b"] = picked[-1]["Model version"]

    with panel:
        r2bar = go.Figure(go.Bar(
            x=[fit["r2_compute"], fit["r2"]], y=["Compute only", "Compute + year"], orientation="h",
            marker=dict(color=[GREY, BLUE], cornerradius=4),
            text=[f"{fit['r2_compute']:.2f}", f"{fit['r2']:.2f}"], textposition="outside",
            hovertemplate="%{y}: R² = %{x:.2f}<extra></extra>"))
        r2bar.update_xaxes(range=[0, 1.15], visible=False)
        r2bar.update_layout(height=150, margin=dict(l=0, r=0, t=26, b=0), plot_bgcolor="white",
                            paper_bgcolor="white", font=dict(family="Inter", size=12, color=INK),
                            title=dict(text="Variance explained (R²)", font=dict(size=13), x=0))
        st.plotly_chart(r2bar, use_container_width=True, config={"displayModeBar": False})
        rate_txt = f"{fit['rate']:.1f}×" if fit["rate"] < 50 else f"{fit['rate_ci'][0]:.0f}×+"
        st.markdown(side("One year of methods ≈", f"{rate_txt} compute",
                         f"95% CI {fit['rate_ci'][0]:.1f}× to {fit['rate_ci'][1]:,.0f}× · n = {fit['n']}"),
                    unsafe_allow_html=True)
        if kind == "eci":
            st.markdown('<div class="callout orange">Release year explains far more than compute here. '
                        'Post-training and reasoning compute is not counted as training compute, '
                        'so the interval is very wide.</div>', unsafe_allow_html=True)

with tab_replay:
    # cumulative frames: models released so far + the fitted curve at that date
    xs = np.logspace(22, 25.7, 60)
    steps = np.arange(2021.75, 2025.01, 0.25)
    colour_of = dict(zip(sorted(mmlu["year"].unique()), RAMP))

    def frame_data(t):
        seen = mmlu[mmlu["t"] <= t]
        return [
            go.Scatter(x=xs, y=A.predict(fit_m, xs, t) * 100, mode="lines",
                       line=dict(color=ORANGE, width=3), hoverinfo="skip"),
            go.Scatter(x=seen["compute"], y=seen["score"] * 100, mode="markers",
                       marker=dict(size=11, color=seen["year"].map(colour_of), line=dict(width=1.5, color="white")),
                       text=seen["Model version"], hovertemplate="<b>%{text}</b><br>%{y:.1f}%<extra></extra>"),
        ]

    def label(t):
        return f"{int(t)} Q{int(round((t % 1) * 4)) + 1}"

    rep = go.Figure(data=frame_data(steps[0]),
                    frames=[go.Frame(data=frame_data(t), name=label(t)) for t in steps])
    rep.update_layout(
        showlegend=False,
        updatemenus=[dict(type="buttons", x=0, y=1.16, xanchor="left", direction="left", showactive=False,
                          buttons=[dict(label="▶ Play", method="animate",
                                        args=[None, dict(frame=dict(duration=600, redraw=True),
                                                         transition=dict(duration=300), fromcurrent=True)]),
                                   dict(label="❚❚", method="animate",
                                        args=[[None], dict(mode="immediate", frame=dict(duration=0))])])],
        sliders=[dict(active=0, x=0.14, len=0.86, y=1.2, currentvalue=dict(prefix="Up to ", font=dict(size=13)),
                      steps=[dict(label=label(t), method="animate",
                                  args=[[label(t)], dict(mode="immediate", frame=dict(duration=0, redraw=True))])
                             for t in steps])])
    rep.add_hline(y=25, line=dict(color=GREY, dash="dot"))
    rep.update_xaxes(type="log", exponentformat="power", dtick=1, range=[22, 25.7])
    rep.update_yaxes(range=[15, 95])
    style(rep, "Training compute (FLOP, log scale)", "MMLU score (%, 5-shot)", 600)
    rep.update_layout(margin=dict(t=110))
    st.plotly_chart(rep, use_container_width=True)
    st.caption("Orange line: fitted MMLU score at the slider date. It lifts at every compute level as time passes.")

# ================================================================== 03
act(3, "Iso-capability map",
    "Colour is the fitted MMLU score for any release date and compute budget. Each contour line is one "
    "score level. Lines slope down: the same score needs less compute every year.")

chart, panel = st.columns([3.2, 1])
with chart:
    gx = np.linspace(2021.5, 2025.0, 120)
    gy = np.linspace(22.0, 25.7, 120)
    GX, GY = np.meshgrid(gx, gy)
    Z = A.predict(fit_m, 10 ** GY, GX) * 100
    iso = go.Figure(go.Contour(
        x=gx, y=gy, z=Z, colorscale=SEQ, zmin=25, zmax=95,
        contours=dict(start=40, end=90, size=10, showlabels=True,
                      labelfont=dict(size=12, color="white", family="Inter")),
        line=dict(color="rgba(255,255,255,.75)", width=1.2),
        colorbar=dict(title=dict(text="MMLU %", side="right"), thickness=12, len=0.8),
        hovertemplate="%{x:.1f}<br>10^%{y:.1f} FLOP<br>fitted MMLU %{z:.1f}%<extra></extra>"))
    iso.add_trace(go.Scatter(
        x=mmlu["t"], y=np.log10(mmlu["compute"]), mode="markers",
        marker=dict(size=8, color=INK, opacity=0.75, line=dict(width=1.2, color="white")),
        text=mmlu["Model version"], customdata=mmlu["score"] * 100,
        hovertemplate="<b>%{text}</b><br>actual MMLU %{customdata:.1f}%<extra></extra>", showlegend=False))
    iso.add_trace(go.Scatter(
        x=path80["t"], y=np.log10(path80["compute"]), mode="lines+markers+text",
        line=dict(color=ORANGE, width=3), marker=dict(size=12, color=ORANGE, line=dict(width=2, color="white")),
        text=path80["Model version"], textfont=dict(color=INK, size=12),
        textposition=["top right"] * (len(path80) - 1) + ["top left"],
        hovertemplate="<b>%{text}</b><extra>cheapest ≥80% so far</extra>", name="Cheapest model ≥ 80%"))
    iso.update_yaxes(tickvals=list(range(22, 26)), ticktext=[pow10(e) for e in range(22, 26)], range=[22, 25.7])
    iso.update_xaxes(tickvals=[2022, 2023, 2024, 2025], ticktext=["2022", "2023", "2024", "2025"],
                     range=[2021.5, 2025.0])
    style(iso, "Release date", "Training compute (FLOP, log scale)", 540)
    iso.update_layout(legend=dict(x=0.01, y=0.02, orientation="v", bgcolor="rgba(255,255,255,.85)"))
    st.plotly_chart(iso, use_container_width=True)
with panel:
    st.markdown(side("Contour slope", f"÷{fit_m['rate']:.1f} per year",
                     "compute needed for any fixed score")
                + side("80% MMLU", f"{drop80:.0f}× cheaper",
                       f"{path80['Model version'].iloc[0]} → {path80['Model version'].iloc[-1]}, 21 months"),
                unsafe_allow_html=True)
    st.markdown('<div class="callout">Orange path: each model that reached 80% with less compute than any '
                'earlier one. It follows the 80% contour down.</div>', unsafe_allow_html=True)

# ================================================================== 04
act(4, "Decomposing a score change",
    "Compute effect, release-date effect (methods) and model-specific residual, from the fitted MMLU model.")
p1, p2, _ = st.columns([1.3, 1.3, 1.4])
a = p1.selectbox("From model", mm_names, key="pick_a")
b = p2.selectbox("To model", mm_names, key="pick_b")
ra, rb = mmlu.set_index("Model version").loc[a], mmlu.set_index("Model version").loc[b]
c_part, a_part, o_part = A.decompose(fit_m, ra["compute"], ra["t"], rb["compute"], rb["t"], ra["score"], rb["score"])

chart, panel = st.columns([3.2, 1])
with chart:
    s_a, steps_ = ra["score"] * 100, [c_part * 100, a_part * 100, o_part * 100]
    level = np.cumsum([s_a] + steps_)
    cats = [a, "More compute", "Better methods", "Model-specific", b]
    bases = [0, *np.minimum(level[:-1], level[1:]), 0]
    heights = [s_a, *np.abs(steps_), rb["score"] * 100]
    text = [f"{s_a:.1f}%", *[f"{v:+.1f}" for v in steps_], f"{rb['score'] * 100:.1f}%"]
    wf = go.Figure(go.Bar(x=cats, y=heights, base=bases, text=text, textposition="outside", width=0.6,
                          marker=dict(color=[INK_2, BLUE, ORANGE, GREY_LIGHT, INK], cornerradius=4),
                          hovertemplate="%{x}: %{text}<extra></extra>"))
    for i, y in enumerate(level):
        wf.add_shape(type="line", x0=i + 0.3, x1=i + 0.7, y0=y, y1=y, line=dict(color=GREY, width=1, dash="dot"))
    lo = min(level.min(), rb["score"] * 100) - 12
    wf.update_yaxes(range=[max(0, lo), max(level.max(), rb["score"] * 100) + 8])
    st.plotly_chart(style(wf, "", "MMLU score (%)", 430), use_container_width=True)
with panel:
    algo_pct = a_part / (c_part + a_part) if (c_part + a_part) else np.nan
    st.markdown(side(f"{a} → {b}", f"{(rb['score'] - ra['score']) * 100:+.1f} pts",
                     f"compute ×{rb['compute'] / ra['compute']:.2f} · {rb['t'] - ra['t']:.1f} years apart")
                + side("Share of explained gain from methods", f"{algo_pct:.0%}",
                       f"explained {(c_part + a_part) * 100:+.1f} pts"),
                unsafe_allow_html=True)

# ================================================================== 05
act(5, "Two readings")
l, r = st.columns(2)
l.markdown(
    f"""<div class="callout"><b>Compute</b><br>
Grows {base_trend['growth']:.1f}× a year vs {fit_m['rate']:.1f}× from methods, and the pace picked up after
{eras(DEFAULT_DOMAINS, False)['year']:.0f}. The top scores belong to the largest models. Compute alone gives
R² = {fit_m['r2_compute']:.2f} on MMLU.</div>""", unsafe_allow_html=True)
r.markdown(
    f"""<div class="callout orange"><b>Algorithms</b><br>
At ~10²⁴ FLOP the best MMLU score went from 60% (2021) to 85% (2024). 80% got {drop80:.0f}× cheaper
in 21 months. About {share:.0%} of yearly progress, more on the newer index.</div>""", unsafe_allow_html=True)
st.markdown(
    f"""**For the department:** compute is the larger driver, but about {share:.0%} of progress comes from
methods, which do not need a data centre. A compute-only budget leaves that share unfunded.""")

with st.expander("Data, method and limits"):
    st.markdown(
        f"""
**Files** (Epoch AI, CC BY 4.0, downloaded 23 Oct 2026): `data/raw/notable_ai_models.csv`,
`data/raw/mmlu_external.csv`, `data/raw/eci_scores.csv`. Cleaning in `prepare_data.py`,
statistics in `analysis.py`, checks in `verify.py`.

**01.** Least-squares fit of log₁₀(compute) on fractional year; doubling time = 12·log₁₀2 / slope; 95% intervals
from 1,000 bootstrap resamples. Two-era mode searches break years 2012–2022 (quarterly) for the split with the
lowest total squared error.

**02–04.** `capability = b0 + bC·log10(compute) + bT·(year − base)`. MMLU is fitted on the logit of
(score − 0.25)/0.75 because the test is bounded. Exchange rate = 10^(bT/bC).
MMLU: bC = {fit_m['beta'][1]:.2f}, bT = {fit_m['beta'][2]:.2f}, n = {fit_m['n']}. The map in 03 is this model
evaluated on a grid; contour slope in log-compute per year = −bT/bC. 04 steps the prediction from
(A's compute, A's date) to (B's compute, A's date) to (B's compute, B's date).

**Share.** log(exchange rate) / (log(exchange rate) + log(compute growth)).

**Limits.** Compute values are Epoch estimates, some speculative. MMLU may be inflated by test questions leaking
into training data, which would overstate the methods effect. Release year also picks up better data and longer
training. The 72 MMLU models are mostly 2023–2024 open models, not a random sample.
""")
    st.download_button("Download MMLU table (CSV)", mmlu.to_csv(index=False), "mmlu_vs_compute.csv", "text/csv")
