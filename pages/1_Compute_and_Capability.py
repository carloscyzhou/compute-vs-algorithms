"""Compute and Capability – my contribution to the group app.

A five-act story:
  1. The compute explosion        (log scatter + frontier envelope + bootstrap trend band)
  2. Same compute, better scores  (scatter + fitted same-year curves, MMLU or ECI)
  3. The falling price of a score (cheapest-model step chart for 60/70/80% MMLU)
  4. Split any gain               (waterfall: compute vs algorithms vs model-specific)
  5. The verdict                  (both sides of the argument)
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
# palette: categorical slots 1-3 + one ordinal blue ramp (see dataviz reference palette)
BLUE, ORANGE, AQUA = "#2a78d6", "#eb6834", "#1baf7a"
GREY, GREY_LIGHT = "#a3a199", "#d9d7d0"
INK, INK_2, INK_3 = "#0b0b0b", "#52514e", "#8a8983"
RAMP = ["#86b6ef", "#3987e5", "#1c5cab", "#0d366b"]   # older -> newer
GRID = "#ecebe7"

st.markdown(
    """
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');
html, body, [class*="css"], .stMarkdown, .stMetric { font-family: 'Inter', sans-serif; }
.block-container { padding-top: 2rem; max-width: 1280px; }
.hero { padding: 2.2rem 2.4rem; border-radius: 18px; color: #fff;
        background: linear-gradient(135deg, #0d366b 0%, #1c5cab 55%, #2a78d6 100%); }
.hero .kicker { text-transform: uppercase; letter-spacing: .14em; font-size: .78rem; opacity: .8; }
.hero h1 { color: #fff; font-size: 2.5rem; font-weight: 800; line-height: 1.1; margin: .4rem 0 .6rem; }
.hero p { font-size: 1.08rem; opacity: .92; max-width: 760px; margin: 0; }
.kpis { display: grid; grid-template-columns: repeat(4, 1fr); gap: 14px; margin: 18px 0 8px; }
.kpi { background: #fcfcfb; border: 1px solid #e6e4de; border-radius: 14px; padding: 16px 18px; }
.kpi .v { font-size: 1.9rem; font-weight: 800; color: #0b0b0b; line-height: 1.1; }
.kpi .l { font-size: .86rem; color: #52514e; margin-top: 4px; }
.kpi .s { font-size: .74rem; color: #8a8983; margin-top: 6px; }
.kpi.accent { border-left: 5px solid #eb6834; }
.act { text-transform: uppercase; letter-spacing: .14em; font-size: .76rem; font-weight: 700;
       color: #2a78d6; margin-top: 2.6rem; }
.act-title { font-size: 1.75rem; font-weight: 800; color: #0b0b0b; margin: .1rem 0 .4rem; }
.lede { font-size: 1.02rem; color: #52514e; max-width: 860px; }
.callout { background: #f4f8fe; border-left: 4px solid #2a78d6; border-radius: 8px;
           padding: 12px 16px; color: #0b0b0b; margin: 8px 0; }
.callout.orange { background: #fdf3ee; border-left-color: #eb6834; }
.side { background: #fcfcfb; border: 1px solid #e6e4de; border-radius: 14px; padding: 14px 16px; }
.side h4 { margin: 0 0 6px; font-size: .95rem; }
.big { font-size: 1.6rem; font-weight: 800; color: #0b0b0b; }
.muted { color: #8a8983; font-size: .8rem; }
@media (max-width: 900px) { .kpis { grid-template-columns: repeat(2, 1fr); } .hero h1 { font-size: 1.8rem; } }
</style>
""",
    unsafe_allow_html=True,
)


# ================================================================== data + models
@st.cache_data
def load():
    compute = pd.read_csv(DATA / "compute_over_time.csv", parse_dates=["date"])
    mmlu = pd.read_csv(DATA / "mmlu_vs_compute.csv")
    eci = pd.read_csv(DATA / "eci_vs_compute.csv")
    for df in (mmlu, eci):
        df["t"] = A.frac_year(df["release"])
    return compute, mmlu, eci


@st.cache_data
def trend(domains, start_year, frontier_only):
    c, _, _ = load()
    c = c[c["Domain"].isin(domains)]
    if frontier_only:
        c = c[c["Frontier model"]]
    c = c[c["date"].dt.year >= start_year]
    if len(c) < 5:
        return None, len(c)
    return A.fit_trend(A.frac_year(c["date"]), c["Training compute (FLOP)"]), len(c)


@st.cache_data
def capability_fit(kind):
    _, mmlu, eci = load()
    df = mmlu if kind == "mmlu" else eci
    return A.fit_capability(df["score"], df["compute"], df["t"], kind, 2021 if kind == "mmlu" else 2023)


def fmt_flop(x, unit=True):
    exp = int(np.floor(np.log10(x)))
    sup = str(exp).translate(str.maketrans("-0123456789", "⁻⁰¹²³⁴⁵⁶⁷⁸⁹"))
    return f"{x / 10**exp:.1f} × 10{sup}" + (" FLOP" if unit else "")


def style(fig, x_title, y_title, height=520):
    fig.update_layout(
        height=height, plot_bgcolor="white", paper_bgcolor="white",
        font=dict(family="Inter, sans-serif", color=INK, size=13),
        margin=dict(l=10, r=20, t=40, b=10),
        legend=dict(orientation="h", y=1.06, x=0, font=dict(color=INK_2)),
        hoverlabel=dict(bgcolor="white", font_color=INK, bordercolor=GREY_LIGHT),
    )
    fig.update_xaxes(title=x_title, gridcolor=GRID, linecolor=GREY_LIGHT, zeroline=False,
                     title_font=dict(color=INK_2))
    fig.update_yaxes(title=y_title, gridcolor=GRID, linecolor=GREY_LIGHT, zeroline=False,
                     title_font=dict(color=INK_2))
    return fig


def act(n, title, lede):
    st.markdown(f'<div class="act">{n:02d}</div><div class="act-title">{title}</div>'
                + (f'<div class="lede">{lede}</div>' if lede else ''), unsafe_allow_html=True)


compute, mmlu, eci = load()
DEFAULT_DOMAINS = ("Language", "Vision", "Multimodal")
base_trend, _ = trend(DEFAULT_DOMAINS, 2010, False)
fit_m = capability_fit("mmlu")
share = A.algo_share(fit_m["rate"], base_trend["growth"])
share_lo = A.algo_share(fit_m["rate_ci"][0], base_trend["growth"])
share_hi = A.algo_share(fit_m["rate_ci"][1], base_trend["growth"])
path80 = A.cheapest_path(mmlu, 0.80)
drop80 = path80["compute"].iloc[0] / path80["compute"].iloc[-1]

# ================================================================== hero
st.markdown(
    """
<div class="hero">
  <h1>Bigger chips or smarter methods?</h1>
  <p>478 notable AI models, 2010–2026. How much of the gain came from training compute,
  and how much from better algorithms.</p>
</div>
""",
    unsafe_allow_html=True,
)
st.markdown(
    f"""
<div class="kpis">
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
</div>
""",
    unsafe_allow_html=True,
)

# ================================================================== act 1
act(1, "Training compute, 2010–2026",
    "Log scale: each grid line is 10× more compute.")

f1, f2, f3 = st.columns([2.2, 1.4, 1])
domains = f1.multiselect("Domain", sorted(compute["Domain"].unique()), default=list(DEFAULT_DOMAINS))
start_year = f2.slider("Fit trend from", 2010, 2022, 2010)
frontier_only = f3.toggle("Frontier only", value=False,
                          help="Epoch AI flag: in the top 10 by training compute when released.")

view = compute[compute["Domain"].isin(domains)] if domains else compute.iloc[0:0]
if frontier_only:
    view = view[view["Frontier model"]]
tr, n_fit = trend(tuple(sorted(domains)), start_year, frontier_only)

chart, side = st.columns([3.2, 1])
with chart:
    if tr is None:
        st.warning("Not enough models for a trend. Add a domain or move the start year back.")
    else:
        fig = go.Figure()
        xs = np.linspace(max(start_year, 2010), view["date"].dt.year.max() + view["date"].dt.dayofyear.max() / 366, 80)
        x_dates = pd.to_datetime((xs - 1970) * 365.25, unit="D")
        band = 10 ** (tr["boots"][:, [1]] + tr["boots"][:, [0]] * xs)
        lo, hi = np.percentile(band, [2.5, 97.5], axis=0)
        fig.add_trace(go.Scatter(x=np.r_[x_dates, x_dates[::-1]], y=np.r_[hi, lo[::-1]], fill="toself",
                                 fillcolor="rgba(11,11,11,0.07)", line=dict(width=0), hoverinfo="skip",
                                 name="95% CI of trend"))
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
                                 line=dict(color=INK, width=1.5), name="Largest model so far",
                                 hoverinfo="skip"))
        fig.add_trace(go.Scatter(x=x_dates, y=10 ** (tr["intercept"] + tr["slope"] * xs), mode="lines",
                                 line=dict(color=ORANGE, width=2.5, dash="dash"),
                                 name=f"Trend: doubles every {tr['doubling']:.1f} months", hoverinfo="skip"))
        for name, ax, ay in [("AlexNet", 30, -40), ("Transformer", -30, -50), ("GPT-3 175B (davinci)", -60, -40),
                             ("GPT-4 (Mar 2023)", -60, -40), ("GPT-6 Astra", -70, 10)]:
            r = view[view["Model"] == name]
            if len(r):
                r = r.iloc[0]
                fig.add_annotation(x=r["date"], y=np.log10(r["Training compute (FLOP)"]),
                                   text=f"<b>{name.split(' (')[0]}</b><br>{r['date'].year}",
                                   ax=ax, ay=ay, arrowcolor=INK_3, arrowwidth=1,
                                   font=dict(size=11, color=INK_2), bgcolor="rgba(255,255,255,.85)")
        fig.update_yaxes(type="log", exponentformat="power", dtick=2)
        st.plotly_chart(style(fig, "Publication date", "Training compute (FLOP, log scale)", 560),
                        use_container_width=True)

with side:
    if tr is not None:
        st.markdown(
            f"""<div class="side"><h4>Doubling time</h4>
<div class="big">{tr['doubling']:.1f} months</div>
<div class="muted">95% CI {tr['doubling_ci'][0]:.1f}–{tr['doubling_ci'][1]:.1f}</div><br>
<h4>Growth per year</h4><div class="big">{tr['growth']:.1f}×</div>
<div class="muted">1,000 bootstrap resamples</div><br>
<h4>Models in fit</h4><div class="big">{n_fit}</div>
<div class="muted">from {start_year}, {', '.join(domains) or '—'}</div></div>""",
            unsafe_allow_html=True)
        st.markdown('<div class="callout">GPT-4 (2023): about <b>45 million×</b> the compute '
                    'of AlexNet (2012).</div>', unsafe_allow_html=True)

# ================================================================== act 2
act(2, "Same compute, better scores",
    "Lines: fitted score by release year. The gap between two lines is the gain at "
    "<b>fixed compute</b>.")

b1, b2 = st.columns([1.4, 2.6])
bench = b1.radio("Benchmark", ["MMLU (2021–2024)", "Epoch Capabilities Index (2023–2026)"],
                 horizontal=False, label_visibility="collapsed")
kind = "mmlu" if bench.startswith("MMLU") else "eci"
df = mmlu if kind == "mmlu" else eci
fit = capability_fit(kind)
years_all = sorted(df["year"].unique())
colours = dict(zip(years_all, RAMP[-len(years_all):]))
years_on = b2.multiselect("Release years shown", years_all, default=years_all)

chart, side = st.columns([3.2, 1])
with chart:
    fig = go.Figure()
    xs = np.logspace(np.log10(df["compute"].min()), np.log10(df["compute"].max()), 60)
    for yr in years_on:
        mid_year = yr + 0.5
        yfit = A.predict(fit, xs, mid_year)
        fig.add_trace(go.Scatter(x=xs, y=yfit * (100 if kind == "mmlu" else 1), mode="lines",
                                 line=dict(color=colours[yr], width=2), showlegend=False, hoverinfo="skip"))
        part = df[df["year"] == yr]
        y = part["score"] * (100 if kind == "mmlu" else 1)
        fig.add_trace(go.Scatter(
            x=part["compute"], y=y, mode="markers", name=str(yr),
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
            fig.add_annotation(x=np.log10(r["compute"]), y=r["score"] * (100 if kind == "mmlu" else 1),
                               text=name, showarrow=False, yshift=15, font=dict(size=11, color=INK_2),
                               bgcolor="rgba(255,255,255,.8)")
    if kind == "mmlu":
        fig.add_hline(y=25, line=dict(color=GREY, dash="dot"), annotation_text="Random guessing (25%)",
                      annotation_position="bottom right", annotation_font_color=INK_3)
        fig.update_yaxes(range=[15, 95])
    fig.update_xaxes(type="log", exponentformat="power", dtick=1)
    st.plotly_chart(style(fig, "Training compute (FLOP, log scale)",
                          "MMLU score (%, 5-shot)" if kind == "mmlu" else "Epoch Capabilities Index", 540),
                    use_container_width=True)

with side:
    r2bar = go.Figure(go.Bar(
        x=[fit["r2_compute"], fit["r2"]], y=["Compute only", "Compute + year"], orientation="h",
        marker=dict(color=[GREY, BLUE], cornerradius=4), text=[f"{fit['r2_compute']:.2f}", f"{fit['r2']:.2f}"],
        textposition="outside", hovertemplate="%{y}: R² = %{x:.2f}<extra></extra>"))
    r2bar.update_xaxes(range=[0, 1.15], visible=False)
    r2bar.update_layout(height=150, margin=dict(l=0, r=0, t=26, b=0), plot_bgcolor="white",
                        paper_bgcolor="white", font=dict(family="Inter", size=12, color=INK),
                        title=dict(text="How much is explained (R²)", font=dict(size=13), x=0))
    st.plotly_chart(r2bar, use_container_width=True, config={"displayModeBar": False})
    rate_txt = f"{fit['rate']:.1f}×" if fit["rate"] < 50 else f"{fit['rate_ci'][0]:.0f}×+"
    st.markdown(
        f"""<div class="side"><h4>Exchange rate</h4>
<div class="big">1 year ≈ {rate_txt} compute</div>
<div class="muted">95% CI {fit['rate_ci'][0]:.1f}× to {fit['rate_ci'][1]:,.0f}× · n = {fit['n']} models</div></div>""",
        unsafe_allow_html=True)
    if kind == "eci":
        st.markdown('<div class="callout orange">Release year explains far more than compute here. '
                    'Post-training and reasoning compute is not counted as training compute, '
                    'so the interval is very wide.</div>', unsafe_allow_html=True)

# ================================================================== act 3
act(3, "Least compute needed to reach a score",
    "Each step is a model that reached the target with less compute than any earlier model.")

targets = {0.60: BLUE, 0.70: ORANGE, 0.80: AQUA}
chart, side = st.columns([3.2, 1])
with chart:
    fig = go.Figure()
    summary = []
    for th, colour in targets.items():
        p = A.cheapest_path(mmlu, th)
        dates = pd.to_datetime(p["release"])
        end = pd.Timestamp("2025-01-31")
        fig.add_trace(go.Scatter(
            x=list(dates) + [end], y=list(p["compute"]) + [p["compute"].iloc[-1]], mode="lines",
            line_shape="hv", line=dict(color=colour, width=2.5), name=f"≥ {th:.0%} MMLU", hoverinfo="skip",
            legendgroup=str(th)))
        fig.add_trace(go.Scatter(
            x=dates, y=p["compute"], mode="markers", showlegend=False, legendgroup=str(th),
            marker=dict(size=10, color=colour, line=dict(width=2, color="white")),
            customdata=np.stack([p["Model version"], (p["score"] * 100).round(1),
                                 p["compute"].map(fmt_flop)], axis=1),
            hovertemplate="<b>%{customdata[0]}</b><br>%{x|%b %Y}<br>MMLU %{customdata[1]}%"
                          "<br>%{customdata[2]}<extra>" + f"≥ {th:.0%}" + "</extra>"))
        last = p.iloc[-1]
        fig.add_annotation(x=pd.to_datetime(last["release"]), y=np.log10(last["compute"]),
                           text=last["Model version"], showarrow=False, yshift=-16, xshift=10,
                           font=dict(size=11, color=INK_2), xanchor="left")
        summary.append((th, p["Model version"].iloc[0], p["Model version"].iloc[-1],
                        p["compute"].iloc[0] / p["compute"].iloc[-1],
                        (dates.iloc[-1] - dates.iloc[0]).days / 30.4))
    fig.update_yaxes(type="log", exponentformat="power", dtick=1, minor=dict(showgrid=False))
    fig.update_xaxes(range=["2021-09-01", "2025-03-01"])
    st.plotly_chart(style(fig, "Release date", "Least compute needed so far (FLOP, log)", 480),
                    use_container_width=True)
with side:
    rows = "".join(f"<tr><td><b>≥{th:.0%}</b></td><td style='text-align:right'><b>{x:.0f}×</b></td>"
                   f"<td style='text-align:right'>{mo:.0f} mo</td></tr>" for th, _, _, x, mo in summary)
    st.markdown(f"""<div class="side"><h4>Compute cut to reach the target</h4>
<table style="width:100%;font-size:.95rem">
<tr class="muted"><td>Target</td><td style='text-align:right'>Less</td><td style='text-align:right'>Over</td></tr>
{rows}</table><br><div class="muted">From the first model to hit the target to the cheapest one by end of 2024.</div></div>""",
                unsafe_allow_html=True)

# ================================================================== act 4
act(4, "Decomposing a score change",
    "Compute effect, release-date effect (methods) and model-specific residual, from the fitted MMLU model.")

names = mmlu.sort_values("t")["Model version"].tolist()
p1, p2, _ = st.columns([1.3, 1.3, 1.4])
a = p1.selectbox("From model", names, index=names.index("Gopher (280B)"))
b = p2.selectbox("To model", names, index=names.index("phi-4"))
ra, rb = mmlu.set_index("Model version").loc[a], mmlu.set_index("Model version").loc[b]
c_part, a_part, o_part = A.decompose(fit_m, ra["compute"], ra["t"], rb["compute"], rb["t"],
                                     ra["score"], rb["score"])

chart, side = st.columns([3.2, 1])
with chart:
    # manual waterfall: each step is a floating bar, so each cause keeps its own colour
    s_a, steps = ra["score"] * 100, [c_part * 100, a_part * 100, o_part * 100]
    level = np.cumsum([s_a] + steps)
    cats = [f"{a}", "More compute", "Better methods", "Model-specific", f"{b}"]
    bases = [0, *np.minimum(level[:-1], level[1:]), 0]
    heights = [s_a, *np.abs(steps), rb["score"] * 100]
    text = [f"{s_a:.1f}%", *[f"{v:+.1f}" for v in steps], f"{rb['score'] * 100:.1f}%"]
    wf = go.Figure(go.Bar(
        x=cats, y=heights, base=bases, text=text, textposition="outside",
        marker=dict(color=[INK_2, BLUE, ORANGE, GREY_LIGHT, INK], cornerradius=4),
        hovertemplate="%{x}: %{text}<extra></extra>", width=0.6))
    for i, y in enumerate(level):
        wf.add_shape(type="line", x0=i + 0.3, x1=i + 0.7, y0=y, y1=y, line=dict(color=GREY, width=1, dash="dot"))
    lo = min(level.min(), rb["score"] * 100) - 12
    wf.update_yaxes(range=[max(0, lo), max(level.max(), rb["score"] * 100) + 8])
    st.plotly_chart(style(wf, "", "MMLU score (%)", 430), use_container_width=True)
with side:
    gain = (rb["score"] - ra["score"]) * 100
    explained = (c_part + a_part) * 100
    algo_pct = a_part / (c_part + a_part) if (c_part + a_part) else np.nan
    st.markdown(
        f"""<div class="side"><h4>{a} → {b}</h4>
<div class="muted">{fmt_flop(ra['compute'])} → {fmt_flop(rb['compute'])}<br>
compute ratio {rb['compute'] / ra['compute']:.2f}× · {rb['t'] - ra['t']:.1f} years apart</div><br>
<h4>Total change</h4><div class="big">{gain:+.1f} pts</div><br>
<h4>Share of explained gain from methods</h4><div class="big">{algo_pct:.0%}</div>
<div class="muted">explained gain {explained:+.1f} pts</div></div>""",
        unsafe_allow_html=True)

# ================================================================== act 5
act(5, "Two readings", "")
l, r = st.columns(2)
l.markdown(
    f"""<div class="callout"><b>Compute</b><br>
Grows {base_trend['growth']:.1f}× a year vs {fit_m['rate']:.1f}× from methods. The top scores belong
to the largest models. Compute alone gives R² = {fit_m['r2_compute']:.2f} on MMLU.</div>""",
    unsafe_allow_html=True)
r.markdown(
    f"""<div class="callout orange"><b>Algorithms</b><br>
At ~10²⁴ FLOP the best MMLU score went from 60% (2021) to 85% (2024). 80% got {drop80:.0f}× cheaper
in 21 months. About {share:.0%} of yearly progress, more on the newer index.</div>""",
    unsafe_allow_html=True)
st.markdown(
    f"""**For the department:** compute is the larger driver, but about {share:.0%} of progress comes from
methods, which do not need a data centre. A compute-only budget leaves that share unfunded.""")

with st.expander("Data, method and limits"):
    st.markdown(
        f"""
**Files** (Epoch AI, CC BY 4.0, downloaded 23 Oct 2026): `data/raw/notable_ai_models.csv`,
`data/raw/mmlu_external.csv`, `data/raw/eci_scores.csv`. Cleaning in `prepare_data.py`,
statistics in `analysis.py`, checks in `verify.py`.

**Act 1.** Least-squares fit of log₁₀(compute) on fractional year. Doubling time = 12·log₁₀2 / slope.
95% intervals from 1,000 bootstrap resamples of models.

**Act 2.** `capability = b0 + bC·log10(compute) + bT·(year − base)`. MMLU is fitted on the logit of the
above-chance score, (score − 0.25)/0.75, because the test is bounded. Exchange rate = 10^(bT/bC).
MMLU: bC = {fit_m['beta'][1]:.2f}, bT = {fit_m['beta'][2]:.2f}, n = {fit_m['n']}.

**Act 3.** For a target score, walk through models by release date and keep each one that reaches the
target with less compute than every earlier model.

**Act 4.** Predicted score at (A's compute, A's date) → (B's compute, A's date) → (B's compute, B's date).
The steps are the compute and methods parts; the remainder is model-specific.

**Act 5 share.** log(exchange rate) / (log(exchange rate) + log(compute growth)).

**Limits.** Compute values are Epoch estimates, some marked speculative. MMLU may be inflated by test
questions leaking into training data, which would flatter newer models and overstate the methods effect.
Release year is a proxy for "better methods": it also picks up better data and longer training.
The 72 MMLU models are mostly 2023–2024 open models, not a random sample.
""")
    st.download_button("Download MMLU table (CSV)", mmlu.to_csv(index=False), "mmlu_vs_compute.csv", "text/csv")
