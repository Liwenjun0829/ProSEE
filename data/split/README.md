# Saved experiment splits

ProSEE writes a `split_indices.json` under this directory, mirroring the
experiment path below `results/` beneath a `results/` namespace (and retains
the existing layout for legacy paths below `outputs/`). When an experiment has an explicitly fixed
split seed, it writes one shared file for the experiment. If the split seed
defaults to each run seed, it stores a separate file for each seed. For example:

```text
data/split/results/experiments/a2g_takeoff_landing/main/mode5_experiment_003/split_indices.json
```

Each file records the split seed, fractions, source dataset, and zero-based
train, validation, and test row indices. Indices refer to rows remaining after
ProSEE validates the numeric inputs and filters invalid values; they are local
to each configured source dataset.
