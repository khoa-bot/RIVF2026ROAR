# processingData — turning raw benchmark logs into the paper's tables
Note: this folder contains programs that may assist in cleaning and aggregating results for easy comparison. This folder does not contain any algorithm logic, or affect the result.
This folder is the **post-processing stage** of the ROAR study. The three solvers each write results
in their own format; the scripts here normalise them, average over the 10 runs, join them into one
wide table per instance partition, and compute the percentage gaps quoted in the paper.
New experiments files may be uploaded without instructions and description in this README file.
Nothing here re-runs a solver. Everything is pure parsing and arithmetic.

---

## 1. Pipeline at a glance

```
      ROAR/                    GILS_RVND/…/benchmark/         PDLSH/source/
  trp_results_*.csv              GILS_*.txt                 PDLSH_*Dataset.txt
        │                            │                             │
        └────────────────┬───────────┴─────────────────────────────┘
                         ▼
          summary_compare_fileCreator.py          ← STEP 1 (run first)
                         │
          trp_summary_regular.csv , trp_summary_large.csv
                         │
                         ├──► generate_percentage.py   → *_with_gaps.csv        (STEP 2)
                         │
                         └──► merge_trivialCombination.py → trp_summary_regular_updated.csv
                                        ▲                                       (STEP 3b)
                                        │
   trp_results_…_trivialCombination.csv ─┴─ aggregated_data_trivialCombination.py  (STEP 3a)

          log_report_ROAR_*.txt ──► loops_creator.py → roar_loops_summary.csv   (independent)
```

**Requirements:** Python 3.9+. `summary_compare_fileCreator.py` and `loops_creator.py` use only the
standard library; the other three need `pandas` (`pip install pandas`).

**All scripts read from and write to their own directory** — copy the raw logs here first, then run
each script with this folder as the working directory.

---

## 2. Input files you must supply

Copy these six files in before running anything:

| Copy from | Into this folder as | Produced by |
|---|---|---|
| `../GILS_RVND/MLP-master/benchmark/GILS_regular.txt` | `GILS_regular.txt` | `run_regular.sh` |
| `../GILS_RVND/MLP-master/benchmark/GILS_large.txt` | `GILS_large.txt` | `run_large.sh` |
| `../PDLSH/source/PDLSH_regularDataset.txt` | same name | `benchmark.py` |
| `../PDLSH/source/PDLSH_largeDataset.txt` | same name | `benchmark.py` |
| `../ROAR/trp_results_ROAR_regularDataset_modified.csv` | same name | `run_regular` |
| `../ROAR/trp_results_ROAR_largeDataset_modified.csv` | same name | `run_large` |

Optional extras:

| File | Needed for |
|---|---|
| `../ROAR/trp_results_ROAR_regularDataset_modified_trivialCombination.csv` | the ROAR-PLS ablation columns |
| `../ROAR/log_report_ROAR_{regular,large,extreme}Dataset_modified.txt` | the LNS loop-count summary |
| `../ROAR/trp_results_ROAR_extremeDataset_modified.csv` | Table III (extreme set) — see §6 |

The copies currently committed here are the exact logs behind the paper's numbers.

### Recognised input formats

**GILS-RVND** (`GILS_*.txt`) — a bare instance path, then interleaved run output:

```
regularInstances/a280.tsp
[PROGRESS] Time: 0.00s, Best Cost: 402456.0 (Seed Finish)
[PROGRESS] Epoch: 1, Interval: 1.00s, Time: 1.00s, Best Cost: 350257.0
...
COST: 347154.8
TIME: 43.52
```

A `COST:` line followed by a `TIME:` line is one run. `[PROGRESS]` lines are ignored.
**An instance header with zero `COST`/`TIME` pairs after it is interpreted as out-of-memory** — this
is how the `OOM` cells in Table II are produced.

**PDLSH** (`PDLSH_*Dataset.txt`) — blocks written by `benchmark.py`:

```
Dataset: usa13509.tsp | Execution: 3 | Nodes: 13509 | Timeout: 213.72s
Best Cost: 129847350149
Best Route: 0 4711 ...
Actual Runtime: 214.0767
```

**ROAR** (`trp_results_*.csv`) — `Dataset,Nodes,Method,RawCost,FinalCost,Time(s)`, with the header row
repeated once per execution block (the solver appends). Repeated headers and short rows are skipped.

---

## 3. The scripts

### 3.1 `summary_compare_fileCreator.py` — **run this first**

```bash
python3 summary_compare_fileCreator.py
```

Parses all three formats for both partitions, averages cost and runtime over runs, and writes:

* `trp_summary_regular.csv`
* `trp_summary_large.csv`

Columns: `Dataset, Nodes, GILS-RVND_AvgCost, GILS-RVND_AvgTime(s), ROAR_AvgCost, ROAR_AvgTime(s),
PDLSH_AvgCost, PDLSH_AvgTime(s)`. Rows are sorted by node count.

Cells with no data are labelled:

* `OOM` — GILS-RVND ran and produced nothing (memory exhaustion)
* `NO_DATA` — no runs found for any other reason

It also prints a coverage summary per algorithm, e.g.
`GILS-RVND: 3/6 datasets | OOM: bbz25234, boa28924, pla33810`.

**Configuration** (top of file):

* `ROAR_COST_FIELD = "FinalCost"` — which ROAR column counts as *the* cost. `"FinalCost"` is the
  post-search latency and is what the paper reports; `"RawCost"` is the nearest-neighbour seed.
* `FILES` — input paths, if your logs live elsewhere.
* `DATA_DIR` — defaults to the script's own directory.

**Name normalisation:** `regularInstances/a280.tsp`, `a280.tsp` and `a280` all collapse to `a280`, so
the three solvers' differing path conventions join correctly.

### 3.2 `generate_percentage.py` — gap columns

```bash
pip install pandas
python3 generate_percentage.py
```

Reads `trp_summary_regular.csv` and `trp_summary_large.csv`, appends

```
Gap_PDLSH_vs_ROAR (%)     = (PDLSH  - ROAR) / ROAR * 100
Gap_GILS-RVND_vs_ROAR (%) = (GILS   - ROAR) / ROAR * 100
```

and writes `trp_summary_regular_with_gaps.csv` / `trp_summary_large_with_gaps.csv`.

**Sign convention: positive means the competitor is worse than ROAR.** These are the numbers behind
"9.8–13.7% over GILS-RVND" and "6.0–14.7% over PDLSH" in Section V-G. `OOM` / `NO_DATA` cells become
`NaN` (via `errors='coerce'`) and yield an empty gap.

### 3.3 `aggregated_data_trivialCombination.py` — ROAR-PLS averages

```bash
python3 aggregated_data_trivialCombination.py
```

Reads `trp_results_ROAR_regularDataset_modified_trivialCombination.csv`, drops the repeated header
rows, groups by `(Dataset, Nodes, Method)`, averages `FinalCost` and `Time(s)`, and writes
`aggregated_trp_results_trivialCombination.csv` with columns `Average_Cost` / `Average_Runtime`,
sorted by node count.

### 3.4 `merge_trivialCombination.py` — join the ablation in

```bash
python3 merge_trivialCombination.py
```

Left-joins `aggregated_trp_results_trivialCombination.csv` onto `trp_summary_regular.csv` on
`(Dataset, Nodes)` and writes `trp_summary_regular_updated.csv`, adding
`ROAR_trivialCombination_AvgCost` and `ROAR_trivialCombination_AvgTime(s)`.

Those two columns are the **ROAR-PLS** column of Table I and the basis for the Section V-F ablation
percentages.

> Run order matters: `3.3` before `3.4`, and `3.1` before both.

### 3.5 `loops_creator.py` — LNS iteration counts

```bash
python3 loops_creator.py
```

Independent of the rest. Scans the three ROAR log files for
`=== Processing Dataset: <name> | Method: ... ===` followed by `Total LNS Loops Executed: <k>`, and
writes `roar_loops_summary.csv`:

```
datasets name,average loops,min-loop,max loop,stdev
mona-lisa100K,242.10,234,252,6.37
```

This quantifies Section V-I ("few loops complete within `T(N)`" as `N` grows) — e.g. `earring200K`
completes ~81 ruin-and-recreate cycles inside its 1,000 s budget.

---

## 4. Full run, from a clean state

```bash
cd processingData

# 1. bring in the raw logs (see §2)
cp ../GILS_RVND/MLP-master/benchmark/GILS_regular.txt .
cp ../GILS_RVND/MLP-master/benchmark/GILS_large.txt .
cp ../PDLSH/source/PDLSH_regularDataset.txt .
cp ../PDLSH/source/PDLSH_largeDataset.txt .
cp ../ROAR/trp_results_ROAR_*.csv .
cp ../ROAR/log_report_ROAR_*.txt .

# 2. build the joined summaries
python3 summary_compare_fileCreator.py

# 3. gaps
python3 generate_percentage.py

# 4. ablation columns
python3 aggregated_data_trivialCombination.py
python3 merge_trivialCombination.py

# 5. loop counts
python3 loops_creator.py
```

---

## 5. Output files

| File | Maps to |
|---|---|
| `trp_summary_regular.csv` | Table I, cost + time columns |
| `trp_summary_large.csv` | Table II |
| `trp_summary_regular_with_gaps.csv` | Section V-E percentages |
| `trp_summary_large_with_gaps.csv` | Section V-G percentages |
| `aggregated_trp_results_trivialCombination.csv` | ROAR-PLS means |
| `trp_summary_regular_updated.csv` | Table I including the ROAR-PLS column |
| `roar_loops_summary.csv` | Section V-I discussion |

The paper's tables divide each row by the `Scale` factor and round to five significant digits; the
CSVs here hold the raw means.

---

