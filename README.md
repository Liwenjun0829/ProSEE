# ProSEE

ProSEE contains the data preparation and symbolic formula-discovery
workflow used for propagation-model studies. The repository layout separates
immutable research data, reproducible configuration, numerical code, scripts,
tests, and generated outputs.

## Project layout

```text
ProSEE/
├── configs/
│   ├── formula_discovery/       # search pipeline and generation configuration
│   └── structure_screen/        # formula-structure model CLI configuration
├── data/
│   ├── raw/                     # local-only source measurements; not distributed
│   ├── processed/               # reproducible preprocessing outputs
│   └── legacy/experiment_inputs/ # archived inputs required by old baselines
├── artifacts/
│   ├── reference_features/      # frozen formula-feature libraries
│   ├── cache_migration/         # retained legacy cache artifact
│   └── experiment_snapshots/    # preserved historical run outputs
├── preprocessing/               # dataset construction
├── scripts/                     # executable research workflows
├── src/prosee/symbolic/         # paper-facing Python API
│   ├── configuration.py
│   ├── data.py
│   ├── formulas.py
│   ├── fitting.py
│   ├── ranking.py
│   ├── baselines.py
│   ├── evaluation.py
│   ├── postprocessing.py
│   ├── experiments.py
│   └── core/                    # Python source only
├── tests/symbolic/              # regression tests
├── results/                     # new experiment outputs, grouped by workflow
└── outputs/                     # historical results retained in place
```

The public modules follow the scientific workflow: configuration, data,
formula discovery, numerical fitting, evaluation, and experiment orchestration.

## Running the pipeline

From the project root, create an environment and install the project with the
optional dependencies used for ranking and plots:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -e ".[ranking,plotting]"
```

Then run the pipeline from the same project root:

```bash
python scripts/run_formula_discovery.py \
  --config configs/formula_discovery/pipeline.json
```

The run script adds the local `src/` directory to Python's import path, so no
machine-specific `PYTHONPATH` setting is needed. The package also exposes
`prosee-symbolic` after installation.

## Reproducing archived research runs

The dataset-specific entry points are in `scripts/`: `run_a2g_takeoff_landing.py` runs the
takeoff/landing experiment 055, `run_a2g_cruise.py` runs cruise experiment
056, `run_square1.py` runs experiment 058, `run_block1.py` runs experiment
059, and `run_maritime.py` runs measured-MAT experiment 060. Their historical
parameters are stored in `configs/experiments/`; by default each uses seeds
1518–1527 and 40 search iterations. See
[`docs/reproducible_experiments.md`](docs/reproducible_experiments.md) for run
options and output locations.

## Data availability

New runs write under `results/`: dataset experiments go to
`results/experiments/<scene>/main/`, PySR baselines to
`results/experiments/<scene>/pysr/`, and the general formula-discovery pipeline
to `results/formula_discovery/`. Reserved folders for summaries, figures, and
logs are available for curated exports. Existing files under `outputs/` are
kept in place and are not moved.

Raw source measurements are not included in this project distribution and
`data/raw/` is ignored by Git. To reproduce workflows that need raw inputs,
obtain the data from its original provider, place it at the path expected by
the relevant configuration under `data/raw/`, or update that configuration to
point to your local copy. Workflows that depend on raw files cannot run until
those inputs are available. Processed datasets currently present under
`data/processed/` and `data/legacy/experiment_inputs/` are separate from the
raw-data distribution policy.

Some processed datasets can be regenerated with the scripts under
`preprocessing/` once their source data is available. For example,
`preprocessing/Maritime/build_plot_data_dataset.py` generates the Maritime
processed CSV from the local source MAT file.

The detailed historical pipeline documentation is retained in `docs/`.
