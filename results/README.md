# Results directory

New generated results are grouped here. Run-specific files stay together so
each result can be traced to its seed and configuration.

- `experiments/<scene>/main/`: main formula-search experiments.
- `experiments/<scene>/pysr/`: PySR baseline runs associated with an experiment.
- `formula_discovery/`: runs from the general formula-discovery configuration.
- `baselines/`: standalone baseline runs.
- `summaries/`: curated cross-run summary tables and reports.
- `figures/`: exported figures intended for comparison or publication.
- `logs/`: run logs and diagnostic captures.

Generated files are ignored by Git; this README and directory markers keep the
standard layout available in a fresh checkout. Existing results under the
legacy `outputs/` directory are left untouched.
