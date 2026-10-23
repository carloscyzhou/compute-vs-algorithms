# What has actually driven AI capability gains?

A Streamlit data narrative for Data Visualisation, Assignment 3.

**Question:** Has AI got better mainly because of more compute, or because of better algorithms?
**Persona:** an Australian Government department deciding whether a national AI
budget should go to compute (chips, data centres) or to research (better methods).

## My page: Compute and Capability (`pages/1_Compute_and_Capability.py`)

Statistics live in `analysis.py`.

1. **Training compute, 2010–2026**: 478 notable models on a log scale with a "largest model so far"
   envelope. Either one trend with a 95% bootstrap band, or two eras split at a fitted break year
   (2019: doubling every 11.7 months before, 4.8 months after).
2. **Same compute, better scores**: MMLU (2021–24) or Epoch Capabilities Index (2023–26) against
   compute, with fitted curves per release year from
   `capability = b0 + bC·log10(compute) + bT·year`. Click dots to send them to section 4.
   A replay tab animates the fitted curve rising quarter by quarter.
3. **Iso-capability map**: the fitted MMLU surface over release date × compute as a contour map,
   with the cheapest-model-to-80% path. Contours fall about 2.8× per year.
4. **Decomposition**: a waterfall splitting the score change between any two models into compute,
   methods and model-specific parts.
5. **Two readings**: the case for compute and the case for algorithms.

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
