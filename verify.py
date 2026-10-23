"""Trace numbers on the dashboard back to the raw Epoch AI files.

Run:  python verify.py
"""
import numpy as np
import pandas as pd

import analysis as A

raw = pd.read_csv("data/raw/mmlu_external.csv")

print("=== Check 1: phi-4 MMLU score (Act 2 hover, Act 3, Act 4) ===")
rows = raw[raw["Model version"] == "phi-4"]
print(rows[["Model version", "EM", "Shots", "Release date", "Training compute (FLOP)", "Source"]].to_string())
print("EM 0.848 x 100 =", round(rows["EM"].median() * 100, 1), "%  (dashboard shows 84.8%)")

print("\n=== Check 2: Gopher MMLU score (Act 4 start bar) ===")
rows = raw[(raw["Model version"] == "Gopher (280B)") & (raw["Shots"] == 5)]
print(rows[["Model version", "EM", "Shots", "Training compute (FLOP)", "Source"]].to_string())

print("\n=== Check 3: '23x less compute to reach 80%' (hero card, Act 3) ===")
gpt4 = raw[raw["Model version"] == "gpt-4-0314"]["Training compute (FLOP)"].iloc[0]
phi4 = raw[raw["Model version"] == "phi-4"]["Training compute (FLOP)"].iloc[0]
print(f"gpt-4-0314 {gpt4:.3e} / phi-4 {phi4:.3e} = {gpt4 / phi4:.1f}x")

print("\n=== Check 4: compute growth and doubling time (hero card, Act 1 default) ===")
m = pd.read_csv("data/raw/notable_ai_models.csv", low_memory=False)
m["date"] = pd.to_datetime(m["Publication date"], errors="coerce")
m = m.dropna(subset=["Training compute (FLOP)", "date"])
m = m[m["date"] >= "2010-01-01"]
m["Domain"] = m["Domain"].fillna("Other").str.split(",").str[0]
m = m[m["Domain"].isin(["Language", "Vision", "Multimodal"])]
years = m["date"].dt.year + (m["date"].dt.dayofyear - 1) / 365.25
slope = np.polyfit(years, np.log10(m["Training compute (FLOP)"]), 1)[0]
print(f"models: {len(m)}  slope: {slope:.3f}  growth/yr: {10**slope:.1f}x  "
      f"doubling: {12 * np.log10(2) / slope:.1f} months")

print("\n=== Check 5: exchange rate and algorithm share (hero cards) ===")
mm = pd.read_csv("data/processed/mmlu_vs_compute.csv")
fit = A.fit_capability(mm["score"], mm["compute"], A.frac_year(mm["release"]), "mmlu", 2021)
print(f"bC = {fit['beta'][1]:.3f}  bT = {fit['beta'][2]:.3f}  "
      f"exchange rate = 10^(bT/bC) = {fit['rate']:.2f}x")
print(f"algorithm share = ln({fit['rate']:.2f}) / (ln({fit['rate']:.2f}) + ln({10**slope:.2f})) = "
      f"{A.algo_share(fit['rate'], 10**slope):.0%}")

print("\n=== Check 6: two-era break year (chart 01, default) ===")
br = A.find_breakpoint(years, m["Training compute (FLOP)"])
print(f"break {br['year']:.1f}: doubling {br['doubling_before']:.1f} mo before, "
      f"{br['doubling_after']:.1f} mo after")
