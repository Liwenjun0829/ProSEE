# Dataset migration record

## Archived experiment data mappings

| Experiment | Logical dataset | ProSEE path | Equivalence evidence |
|---|---|---|---|
| 055 | takeoff, landing | `data/processed/A2G/` | files are byte-identical to the legacy CSVs |
| 056 | cruise | `data/processed/A2G/` | file is byte-identical to the legacy CSV |
| 058 | freq1 | `data/processed/Square1/standardized_pathloss_dataset_f1p4GHz_V.csv` | archived training, validation, and test hashes match |
| 058 | freq2 | `data/processed/Square1/standardized_pathloss_dataset_f28p0GHz_V.csv` | archived training, validation, and test hashes match |
| 058 | freq3 | `data/processed/Square1/standardized_pathloss_dataset_f3p5GHz_V.csv` | archived training, validation, and test hashes match |
| 059 | freq1–freq3 | `data/legacy/experiment_inputs/class1_all_freq*.csv` | preserved legacy exports; current Block1 membership differs |
| 060 | measured Maritime path loss | `data/processed/Maritime/plot_data2.csv` | generated from `data/raw/Maritime/plot_data2.mat`; CSV float round-trip and fixed-seed splits match exactly |

For experiment 058, recomputing the seeded 50/25/25 split produces the archived
hashes:

- training: `c260bf838e025bee7c300445c8492855c2a986ec37249acc1fbb60272e98dbaa`
- validation: `386d2d232fdcae919f60bf7667d1b562f959bef8467052b122f8ad78f7a7e9a0`
- test: `a282d6fcaf0f9bb60bc6eb5f1a0eb10bc889fbda23e2d97ae247da0a69c09d6c`

The redundant A2G takeoff, cruise, and landing CSV copies were removed from
`src/prosee/symbolic/core`; their canonical, byte-identical files remain in
`data/processed/A2G`. The duplicate Maritime MAT file formerly under
`data/legacy` was removed after its SHA-256 was confirmed identical to the
canonical source under `data/raw/Maritime`. The preprocessing script writes sorted finite
`distance`/`path_loss` values with 17 significant digits, preserving each
loaded float exactly.
