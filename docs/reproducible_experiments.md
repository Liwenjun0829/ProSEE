# Reproducing the archived research experiments

The entry points use the parameter values recorded in the corresponding
`seed_001_config.json` from the historical `experiment_055` through
`experiment_060` results. They default to ten seeds starting at 1518,
40 search iterations, a retraining interval of five, and 24 LM workers.
New runs for each scenario are stored under
`results/experiments/<scene>/main/` and PySR runs under
`results/experiments/<scene>/pysr/`. Existing historical files under
`outputs/` remain in place. PySR is run
independently with scene-specific entry points under `scripts/pysr/`. Each
entry point reads its matching sidecar and writes PySR results plus PySR-only
canonicalization there. For example:

Each experiment also saves its exact train/validation/test row indices in
`data/split/results/`, mirroring the experiment's path below `results/`. A fixed split
seed produces one shared file for all seeds in that experiment; runs whose
split seed follows the run seed retain separate files. Each JSON records the
split seed, source datasets, and row indices after ProSEE's input validation
and invalid-value filtering.

```bash
python scripts/pysr/run_a2g_takeoff_landing.py
python scripts/pysr/run_square1.py
```

Existing output directories are not overwritten. Regular experiment scripts
run the main search only and canonicalize only main-model results under
`main/<run>/cross_seed_formula_summary/main/`. This directory contains
cross-seed formula summaries, including formulas that occur in only one seed;
the `*_repeated_structures.csv` files and bubble plots within it are specifically
limited to structures shared by multiple seeds. The generic
`scripts/run_pysr_experiment.py --config ...` entry point remains available.
A2G cruise is temporarily disabled for PySR; its main-search experiment remains
available through `scripts/run_a2g_cruise.py`.

| Script | Historical run | Input data |
|---|---:|---|
| `scripts/run_a2g_takeoff_landing.py` | 055 | A2G takeoff and landing CSVs |
| `scripts/run_a2g_cruise.py` | 056 | A2G cruise CSV |
| `scripts/run_square1.py` | 058 | Square1 frequencies 1, 2, and 3 |
| `scripts/run_block1.py` | 059 | Preserved legacy Block1 class 1 exports |
| `scripts/run_maritime.py` | 060 | Measured MAT data and cruise CSV |

Run a script from the project root with the project Python environment, for
example:

```bash
/home/lwj/miniconda3/envs/py39/bin/python scripts/run_a2g_takeoff_landing.py
```

Each entry point accepts `--seed-count`, `--target-iterations`,
`--retrain-interval`, and `--n-jobs`. Use `--serial-lm` to run fitting serially.
The default worker count matches the archived seed configurations. Input CSV
and MAT files and the frozen formula-feature artifacts are resolved from the
ProSEE project directories; their bytes were checked against the historical
files where both copies were available.
