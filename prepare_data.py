"""Turn the raw Epoch AI files into the three small tables the dashboard uses.

Raw files (downloaded 23 Oct 2026, CC BY 4.0, Epoch AI):
  data/raw/notable_ai_models.csv  <- https://epoch.ai/data/notable_ai_models.csv
  data/raw/mmlu_external.csv      <- https://epoch.ai/data/benchmark_data.zip (file inside the zip)
  data/raw/eci_scores.csv         <- same zip, epoch_capabilities_index/eci_scores.csv

Run:  python prepare_data.py
"""
from pathlib import Path

import pandas as pd

RAW = Path("data/raw")
OUT = Path("data/processed")
OUT.mkdir(parents=True, exist_ok=True)

# ---- Table 1: training compute of notable models over time ----
models = pd.read_csv(RAW / "notable_ai_models.csv", low_memory=False)
compute = models[["Model", "Organization", "Publication date", "Domain",
                  "Training compute (FLOP)", "Confidence", "Frontier model"]].copy()
compute["date"] = pd.to_datetime(compute["Publication date"], errors="coerce")
# Step 1: keep only models with a compute estimate and a valid date
compute = compute.dropna(subset=["Training compute (FLOP)", "date"])
# Step 2: keep 2010 onwards (the deep learning era)
compute = compute[compute["date"] >= "2010-01-01"]
# Step 3: first listed domain only, so each model has one colour
compute["Domain"] = compute["Domain"].fillna("Other").str.split(",").str[0]
compute["Frontier model"] = compute["Frontier model"].eq(True)
compute = compute.drop(columns="Publication date").sort_values("date")
compute.to_csv(OUT / "compute_over_time.csv", index=False)

# ---- Table 2: MMLU score vs training compute ----
mmlu = pd.read_csv(RAW / "mmlu_external.csv")
# Step 1: keep 5-shot results only, so every score uses the same test setting
mmlu = mmlu[mmlu["Shots"] == 5]
# Step 2: keep rows that have a training compute estimate
mmlu = mmlu.dropna(subset=["Training compute (FLOP)"])
# Step 3: some models are reported by several papers -> take the median score
mmlu = (mmlu.groupby("Model version")
            .agg(score=("EM", "median"),
                 reports=("EM", "size"),
                 compute=("Training compute (FLOP)", "first"),
                 release=("Release date", "first"),
                 organization=("Organization", "first"))
            .reset_index())
mmlu["score_pct"] = (mmlu["score"] * 100).round(1)
mmlu["year"] = pd.to_datetime(mmlu["release"]).dt.year
mmlu = mmlu.sort_values("compute")
mmlu.to_csv(OUT / "mmlu_vs_compute.csv", index=False)

# ---- Table 3: Epoch Capabilities Index (ECI) vs training compute, 2023-2026 ----
eci = pd.read_csv(RAW / "eci_scores.csv")
# Step 1: attach training compute by joining on the model name used in both Epoch files
eci = eci.merge(models[["Model", "Training compute (FLOP)"]], on="Model", how="inner")
# Step 2: keep models that have a compute estimate
eci = eci.dropna(subset=["Training compute (FLOP)"])
eci = eci.rename(columns={"Training compute (FLOP)": "compute", "date": "release",
                          "Organization": "organization", "Display name": "Model version"})
eci["score"] = eci["eci"]
eci["year"] = pd.to_datetime(eci["release"]).dt.year
eci = eci[["Model version", "score", "eci_ci_low", "eci_ci_high", "compute", "release",
           "organization", "year"]].sort_values("compute")
eci.to_csv(OUT / "eci_vs_compute.csv", index=False)

print(f"compute_over_time.csv: {len(compute)} models")
print(f"mmlu_vs_compute.csv:   {len(mmlu)} models")
print(f"eci_vs_compute.csv:    {len(eci)} models")
