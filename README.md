# Cross-Task Federated Backbone Aggregation with Selective State Space Models for Building Energy Analytics

Code and experiment artefacts accompanying the ACM BALANCES 2026 submission.

## What this repo does

We train a single selective state-space model (**MambaMixer**) to serve two
building-analytics tasks at once — **hourly load forecasting** and **energy
anomaly detection** — and we train it across many buildings without moving
raw meter data off-site.

The core idea: split the model into a shared **multi-scale BiMamba backbone**
and two small task-specific heads. In every federated round, the backbone is
aggregated across *all* participating clients regardless of their task, while
each head is averaged only within its own task group. This cross-task backbone
sharing gives the anomaly head a cleaner reconstruction of "normal" building
behaviour and noticeably improves precision, at a negligible cost in
forecasting accuracy.

## What's inside

- `models/mamba_mixer.py` — MambaMixer architecture (multi-scale patching,
  BiMamba selective-SSM, FiLM cross-scale gate, patch decoder, two task heads).
- `models/baselines/` — LSTM, LSTM-AE, Informer, MSD-Mixer, ANN-AE baselines.
- `trainers/centralized_trainer.py` — pooled-data training loop (upper bound).
- `trainers/multitask_fed_trainer.py` — federated training with the cross-task
  backbone-aggregation schedule; supports FedAvg and FedProx, plus the
  single-task and local-only ablations.
- `preprocess.py` — selects 50 ASHRAE buildings (forecasting) and 50 LEAD 1.0
  buildings (anomaly detection), applies `log1p`, and writes the 70/20/10
  temporal splits used everywhere else.
- `data_provider/` — per-task window datasets and federated dataloaders.
- `experiments/run_baselines.py` — driver for the centralised baselines.
- `main.py` — top-level entry point.

## Datasets

- **ASHRAE Great Energy Predictor III** — hourly electricity meters for the
  forecasting task.
- **LEAD 1.0** — expert-annotated anomaly labels over ASHRAE electricity meter
  series for the anomaly task.

Raw CSVs are not distributed here. Set `ASHRAE_RAW_CSV` and `LEAD_RAW_CSV`
before running `python preprocess.py`.

## Reproducing the headline results

```bash
python preprocess.py                 # build the 50+50 building splits
python main.py                       # cross-task FL (FedAvg),  proposed
python main.py --fl_strategy fedprox # cross-task FL with FedProx (μ=0.01)
python main.py --aggregation single_task  # single-task FL ablation
python main.py --aggregation local_only   # no aggregation baseline
python experiments/run_baselines.py       # centralised LSTM / Informer / MSD-Mixer / LSTM-AE / ANN-AE
```

All reported numbers come from the JSON files written into `results/` by these
commands.
