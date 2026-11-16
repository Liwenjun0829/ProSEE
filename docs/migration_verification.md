# Migration verification report

## Scope

The migrated scope contains the complete executable dependency closure of
`formula_discovery_pipeline.py`: formula basis generation, structure
features, ranking, lightweight GP, shared data loading, canonicalization, the
structure-screen package, configurations, frozen feature artifacts, scripts,
tests, and required datasets.

The original source tree and all original data remain unchanged and available
as the historical reference. Experiment artifacts remain in the historical
tree to avoid duplicating roughly 17 GB of immutable outputs.

## Source equivalence

`MIGRATION_MANIFEST.json` records the SHA-256 value of each migrated numerical
source. The verification command compares the 19 computational modules with
the original tree after accounting for the documented module names and path
updates:

```text
Migration equivalence: PASS (19 core files)
```

The migration updates module names, resource locations, and descriptive labels.
Formula-generation expressions, search and fitting operations, random-number
calls, multiprocessing targets, and numerical calculations remain unchanged.
The new result filename also reads an existing `dimensionless_formula_results.json`
file when the renamed result file is absent.

## Regression equivalence

The full copied test suite was run independently against the historical tree
and against ProSEE using the same Python 3.9 environment.

| Location | Passed | Failed | Warnings |
|---|---:|---:|---:|
| Historical tree | 117 | 11 | 3 |
| ProSEE | 117 | 11 | 3 |

The same 11 test names fail in both locations. One failure path attempts a
network lookup for Julia, which is rejected by the restricted environment; the
remaining failures are pre-existing expectation differences in the historical
suite. No new failure was introduced by migration.

## Data equivalence

- A2G takeoff, cruise, and landing datasets are byte-identical to their
  historical files.
- Square1 experiment 058 reproduces all archived seeded split hashes exactly.
- Experiment 059 uses byte-preserved legacy Block1 exports because current
  processed class membership differs.
- Experiment 060 uses the byte-preserved MAT file.

## Interface checks

- The structured package imports under Python 3.9.
- The command-line pipeline exposes the historical arguments.
- Both precomputation commands and the lightweight archived-experiment verifier load from
  their ProSEE locations.
- Canonical configuration resolves to existing ProSEE data and writes to
  `outputs/formula_discovery`.
