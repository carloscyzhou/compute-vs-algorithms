# What has actually driven AI capability gains?

A Streamlit data narrative for Data Visualisation, Assignment 3.

**Question:** Has AI got better mainly because of more compute, or because of better algorithms?
**Persona:** an Australian Government department deciding whether a national AI
budget should go to compute (chips, data centres) or to research (better methods).

## My page: Compute and Capability (`pages/1_Compute_and_Capability.py`)

A five-act data story. Statistics live in `analysis.py`.

1. **The compute explosion** – 478 notable models on a log scale, with a "largest model so
   far" envelope, a fitted trend and its 95% bootstrap band. Filters for domain, trend start
   year and frontier-only. Gives doubling time (5.4 months) and growth per year (4.7×).
2. **Same compute, better scores** – MMLU (2021–24) or Epoch Capabilities Index (2023–26)
   against compute, coloured by release year. Fitted same-year curves come from the model
   `capability = b0 + bC·log10(compute) + bT·year`. Also shows R² for compute alone vs
   compute + year, and the "exchange rate" 10^(bT/bC): one year of better methods ≈ 2.8× compute.
3. **The falling price of a score** – step chart of the least compute needed to hit 60/70/80%
   MMLU. Reaching 80% got 23× cheaper (GPT-4 → phi-4).
4. **Split any gain** – pick two models; a waterfall splits the score change into compute,
   better methods and model-specific parts.
5. **The verdict** – the case for compute and the case for algorithms, side by side.

## Data

| File | Source | Licence |
|---|---|---|
| `data/raw/notable_ai_models.csv` | Epoch AI, *Data on Notable AI Models*, https://epoch.ai/data/notable_ai_models.csv | CC BY 4.0 |
| `data/raw/mmlu_external.csv` | Epoch AI, *Capabilities & benchmarking*, inside https://epoch.ai/data/benchmark_data.zip | CC BY 4.0 |
| `data/raw/eci_scores.csv` | same zip, `epoch_capabilities_index/eci_scores.csv` | CC BY 4.0 |

All downloaded 23 October 2026. `prepare_data.py` turns them into `data/processed/*.csv`
(every cleaning step is commented). `verify.py` traces dashboard numbers back to the raw files.

## Run locally

```bash
python3 -m venv .venv
.venv/bin/pip install -r requirements.txt
.venv/bin/python prepare_data.py
.venv/bin/streamlit run Home.py
```

## AI use

Claude (Anthropic) was used to find the Epoch AI datasets, draft the Python code (data cleaning,
regression, bootstrap, charts) and draft the presentation script. Every number on screen was checked against the raw CSV with
`verify.py` and by opening the raw file, and the phi-4 MMLU score was checked against the
Phi-4 Technical Report.
