# 原始数据来源 / Data sources

全部来自 Epoch AI，2026 年 10 月 23 日下载，许可证 CC BY 4.0（可自由使用，需注明来源）。

| 本地文件 | 下载链接 | 说明页面 |
|---|---|---|
| `data/raw/notable_ai_models.csv` | https://epoch.ai/data/notable_ai_models.csv | https://epoch.ai/data/notable-ai-models |
| `data/raw/mmlu_external.csv` | https://epoch.ai/data/benchmark_data.zip （压缩包里的 `mmlu_external.csv`） | https://epoch.ai/benchmarks |
| `data/raw/eci_scores.csv` | 同一个压缩包里的 `epoch_capabilities_index/eci_scores.csv` | https://epoch.ai/benchmarks/eci |

用来核对 phi-4 分数的原始论文：Phi-4 Technical Report，https://arxiv.org/abs/2412.08905

## 引用格式 / Citation

- Epoch AI (2026). *Data on Notable AI Models*. Retrieved 23 October 2026 from https://epoch.ai/data/notable-ai-models
- Epoch AI (2026). *Capabilities & Benchmarking*. Retrieved 23 October 2026 from https://epoch.ai/benchmarks

## 每个文件用了哪些列

- **notable_ai_models.csv**：`Model`, `Organization`, `Publication date`, `Domain`, `Training compute (FLOP)`, `Confidence`, `Frontier model`
- **mmlu_external.csv**：`Model version`, `EM`（分数，0–1）, `Shots`, `Release date`, `Training compute (FLOP)`, `Organization`, `Source`
- **eci_scores.csv**：`Model`, `Display name`, `eci`, `eci_ci_low`, `eci_ci_high`, `date`, `Organization`

处理步骤见 `prepare_data.py`，核对脚本见 `verify.py`。
