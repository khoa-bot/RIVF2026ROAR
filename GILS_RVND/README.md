# GILS-RVND — baseline for the ROAR paper

This folder contains the **GILS-RVND** baseline used in the ROAR paper (Section V-A, Tables I & II).

GILS-RVND is the gold-standard TRP/MLP metaheuristic of:

> M. M. Silva, A. Subramanian, T. Vidal, L. S. Ochi,
> *"A simple and effective metaheuristic for the minimum latency problem"*,
> European Journal of Operational Research **221**(3), 513–520, 2012.

---

## 1. Provenance — read this first

The code in `MLP-master/` is **adapted from the public re-implementation by Francis Cunha**:

> **https://github.com/franciscunha/MLP**

The following files are **unmodified upstream code**:

* `src/readData.cpp`, `src/readData.h` — TSPLIB parser and `O(N^2)` distance matrix builder
  (Portuguese comments are the original author's)
* `makefile`
* the `src/ obj/` layout and build conventions

The following file was **modified for the ROAR paper**:

* `src/main.cpp` — see [§4](#4-modifications-relative-to-upstream) for the complete list of changes.

The following files were **added** for this study:

* `run_regular.sh`, `run_large.sh` — batch drivers
* `regularInstances/`, `largeInstances/` — the two instance partitions
* `benchmark/GILS_regular.txt`, `benchmark/GILS_large.txt` — captured run logs

Upstream is a faithful re-implementation of Silva et al. (2012); this fork changes only the
initialization, the stopping rule and the logging. **The search itself — RVND, the five
neighbourhoods, the `O(N^2)` re-optimization structure and the double-bridge perturbation — is
untouched.**

---

## 2. Contents

```
GILS_RVND/
├── .vscode/                     launch + build configs (local convenience)
└── MLP-master/
    ├── makefile                 upstream build (g++ -O3 -std=c++0x, srcs in src/, objs in obj/)
    ├── mlp                      checked-in Linux x86-64 binary (rebuild it)
    ├── run_regular.sh           10 runs x 20 regular instances -> benchmark/GILS_regular.txt
    ├── run_large.sh             10 runs x  6 large   instances -> benchmark/GILS_large.txt
    ├── src/
    │   ├── main.cpp             GILS-RVND driver + RVND + neighbourhoods   [MODIFIED]
    │   ├── readData.cpp/.h      TSPLIB parser, builds costM[N+1][N+1]      [upstream]
    │   ├── stable_main.txt      snapshot of an earlier main.cpp (reference only, not compiled)
    │   ├── temp                 scratch file (not compiled)
    │   └── *.o                  stale object files — `make clean` before building
    ├── obj/                     build output directory (*.o, *.d)
    ├── regularInstances/        20 .tsp — 52 to 11,849 nodes (Table I)
    ├── largeInstances/          6 .tsp  — 13,509 to 33,810 nodes (Table II)
    └── benchmark/
        ├── GILS_regular.txt     captured stdout (COST / TIME / PROGRESS lines)
        ├── GILS_large.txt       ditto
        └── extra/instlist.sh    tiny helper, unused by the benchmark
```

Only `src/*.cpp` are compiled — `stable_main.txt` and `temp` have non-`.cpp` extensions precisely so
the `$(wildcard src/*.cpp)` rule in the makefile ignores them.

---

## 3. Building and running

### Requirements

GCC with `-std=c++0x`. No external libraries. **Memory is the binding constraint** — see §6.

### Build

```bash
cd GILS_RVND/MLP-master
make clean      # important: stale .o files are checked in
make            # produces ./mlp
```

### Single instance

```bash
./mlp regularInstances/berlin52.tsp
```

The binary takes **exactly one argument**, the path to a TSPLIB `.tsp` file, and prints:

```
SEED COST: <latency of the nearest-neighbour construction>      (stderr)
[PROGRESS] Epoch: k, Interval: E(N)s, Time: t s, Best Cost: Z   (stdout, every E(N) seconds)
FINAL SOLUTION: 1 47 23 ... <n>
COST: <best depot-inclusive latency found>
TIME: <wall clock seconds>
```

### Full benchmark

```bash
chmod +x run_regular.sh run_large.sh
./run_regular.sh     # 20 instances x 10 runs, appends to benchmark/GILS_regular.txt
./run_large.sh       #  6 instances x 10 runs, appends to benchmark/GILS_large.txt
```

Both scripts `grep` for `COST|TIME|PROGRESS` and **append**. Move or delete the target file before a
fresh benchmark run. The `cd ..` at the end of each script is vestigial — the scripts never `cd`
anywhere, so run them from `MLP-master/`.

To feed `../processingData`, copy `benchmark/GILS_regular.txt` and `benchmark/GILS_large.txt` there.

---

## 4. Modifications relative to upstream

All changes live in `src/main.cpp`. Line references are to the file as committed.

### 4.1 Nearest-neighbour construction replaces GRASP (L97–150, L473)

```cpp
double R_set[] = {0.00};                 // was the 26-value set {0.00, 0.01, ..., 0.25}
```

When `alpha == 0.0`, `construction()` takes a dedicated branch: a canonical nearest-neighbour walk
from node 1, scanning candidates in ascending index order with a strict `<` so ties break to the
lowest index. The original GRASP branch (sort the candidate list by distance, pick uniformly from the
first `ceil(alpha * |CL|)`) is preserved verbatim in the `else` and is still reachable if `R_set` is
restored.

**Why:** Section V-A of the ROAR paper — "for large-scale instances, GRASP adds substantial
computational overhead and unfairly degrades performance under limited runtime". The paper reports
the substitution is benign: the baseline still reproduces the published optimum on every instance up
to 150 nodes (`pr107` excepted, within 0.07%).

**Note:** at `alpha = 0.0` the *original* GRASP branch would already reduce to nearest neighbour
(`rcl_size = max(1, 0) = 1`). The dedicated branch is therefore behaviourally equivalent but avoids
an `O(n log n)` sort at every step.

**Consequence worth knowing:** the construction is now deterministic, so all `I_MAX = 10` multi-starts
begin from the *same* seed tour. Diversification comes only from the randomized neighbourhood order
in RVND and the random double-bridge. Upstream/Silva et al. get an additional source of
diversification from the randomized construction. See [§6](#6-known-quirks-and-caveats).

### 4.2 Hard wall-clock budget via exception unwinding (L24–77, L449–454, L478–533)

`checkStrictTimeout()` throws `int(1)` once `get_search_timeout(dimension)` seconds have elapsed. It
is called at the top of every loop that can run long: `construction`, `UpdateReopt`, `apply_swap`,
`apply_flip`, `apply_reinsertion`, `RVND`, `perturb`, and the ILS loop. `main()` wraps the whole
GILS-RVND driver in a single `try { ... } catch (int e) { ... }`.

`get_search_timeout()` is **bit-identical to `T(N)` in ROAR and PDLSH**:

```
T(N) = 100                            N <= 1000
     = 100 + 900*(N-1000)/99000       1000 < N <= 100,000     (= 100 + (N-1000)/110)
     = 1000                           N > 100,000
```

### 4.3 Exception-safe incumbent tracking (L32–42, L399, L495)

`absoluteBestSolution` / `absoluteBestCost` are updated by `updateAbsoluteBest()` immediately after
the construction and after **every** improving RVND move. When the timeout exception fires mid-search
the partially-updated `solutionAlpha/Beta/Omega` are simply discarded and the tracker is reported.
This is what makes a hard-deadline run meaningful — without it, an interrupted run would report
whatever half-perturbed tour happened to be live.

`updateAbsoluteBest` is deliberately **not** called after `perturb()`, so a perturbed (worse) tour can
never become the reported best.

### 4.4 Epoch progress logging (L43–48, L61–75)

`get_epoch_interval()` mirrors ROAR's `E(N)` (Eq. 9) and prints a `[PROGRESS]` line every `E(N)`
seconds once the first seed is built. **Logging only** — unlike ROAR, GILS-RVND has *no* early-stop
on non-improving epochs; it runs until `I_MAX x I_ILS` completes or `T(N)` fires.

### 4.5 Output format (L539–545)

Prints `FINAL SOLUTION`, `COST`, `TIME` in the shape `run_*.sh` greps for.

---

## 5. Algorithm conformance to Silva et al. (2012)

Verified line by line. Everything below is **unchanged** from the published algorithm.

| Component | Paper | `src/main.cpp` |
|---|---|---|
| Outer multi-start | `I_MAX = 10` | `const int I_MAX = 10;` (L468) |
| ILS iterations | `I_ILS = min(n, 100)` | `(dimension >= 100) ? 100 : dimension` (L469) |
| Neighbourhoods | swap, 2-opt, or-opt-1/2/3 | `N1..N5` (L14, L386–392) |
| RVND | pick a random neighbourhood; on improvement restore the full list, else drop it | L378–405 |
| Move selection | best improvement within a neighbourhood, then re-enter RVND | `apply_swap` / `apply_flip` / `apply_reinsertion` each scan fully and apply the single best |
| Re-optimization | `O(1)` move evaluation via `(W, T, C)` subsequence concatenation over an `O(N^2)` table | `ReoptData` + `UpdateReopt` (L79–183) |
| Perturbation | double bridge, segment sizes in `[2, ceil(n/10)]`, non-overlapping | `perturb()` (L412–447) |
| Acceptance | `s'' <- s'` iff `f(s') < f(s'')`, reset `iterILS` | L513–517 |
| Objective | depot-inclusive latency | `reopt[0][0].W = 0`, all other `W = 1`, and `s` ends with a return to node 1 → the return leg is counted (L152–183, L488) |

**Depot-inclusive check.** With `W[j][j] = 1` and `C[j][j] = 0`, the recurrence
`C[i][j] = C[i][j-1] + W[j][j]*(T[i][j-1] + c) + C[j][j]` collapses to `C[i][j] = C[i][j-1] + T[i][j]`,
so `C[0][LAST] = sum_{j=1}^{LAST} T[0][j]` where `s[LAST] = 1` is the depot return. `W[0][0] = 0`
excludes the departure. This is exactly the metric ROAR's `calculate_trp_cost(..., depot_inclusive=true)`
computes.

**Distance rounding** (`readData.cpp`): `EUC_2D -> floor(sqrt(...) + 0.5)` (nearest integer),
`CEIL_2D -> ceil(...)`, `ATT -> TSPLIB pseudo-Euclidean`, `GEO -> TSPLIB great circle`. Identical to
ROAR's `DistanceEvaluator`, so the two are directly comparable.

---

## 6. Known quirks and caveats

1. **`O(N^2)` memory is the wall — this is the paper's Table II OOM result, not a bug.**
   `readData` allocates `costM` as `(N+1)^2` doubles, and `main()` allocates
   `reopt` as `(N+1)^2` × `sizeof(ReoptData)` (24 B). At `N = 25,234` that is
   ≈ 5.1 GB + 15.3 GB ≈ **20 GB**, on top of `vector<vector<>>` row overhead — hence OOM on a 32 GB
   machine. `usa13509` already needs ≈ 5.9 GB. Budget accordingly; the large-set script will kill the
   process on the last three instances.

2. **Deterministic seed across all 10 multi-starts.** Because `R_set = {0.00}`, every `I_MAX`
   iteration constructs the identical tour (§4.1). Under the hard deadline most instances never
   finish even one outer iteration, so this rarely bites — but if you restore GRASP, restore the full
   26-value `R_set` too.

3. **Stale `.o` files are committed** in both `src/` and `obj/`. Always `make clean` first, or the
   makefile may link an object built from an older `main.cpp`.

4. **`perturb()` can divide by zero on tiny instances.** `randomRange(1, LAST_NODE - maxSubsegSize - 1)`
   computes `rand() % (max - min + 1)`; if `dimension < maxSubsegSize + 2` the modulus is `<= 0`.
   Safe for everything here (the smallest instance is `berlin52`), but do not point this at a 10-node
   toy problem.

5. **`readData.cpp` does not clamp the `acos` argument** in `CalcDistGeo`, so a `GEO` instance could
   produce `NaN` from floating-point drift. No `GEO` instance is used in this study. ROAR's evaluator
   does clamp.

6. **`readData.cpp` has two upstream typos** in rarely used `EXPLICIT` branches: `LOWER_COL`
   (L192–197) and `LOWER_DIAG_COL` (L227–232) increment `j` in the inner loop instead of `i`, giving
   an infinite loop. Untouched from upstream and unreachable for coordinate-based instances; fix
   before using `EDGE_WEIGHT_FORMAT: LOWER_COL` data.

7. **`srand(time(NULL))`** with 1-second resolution — two runs launched inside the same second share
   an RNG stream. `run_*.sh` runs are longer than a second, so this does not affect the benchmark.

8. **`[PROGRESS]` lines are appended to the benchmark file** by `run_*.sh`'s grep. The parser in
   `../processingData/summary_compare_fileCreator.py` deliberately skips lines starting with `[`.

9. **OOM is inferred, not reported.** When `mlp` is killed by the OOM killer it prints nothing, so
   `run_*.sh` writes only the instance-name header line. `summary_compare_fileCreator.py` treats "a
   `.tsp` header with zero `COST`/`TIME` pairs after it" as `OOM`. If a run fails for any *other*
   reason it will be silently mislabelled as OOM — check `dmesg` if in doubt.

---

## 7. Citation

If you use this baseline, cite both the algorithm and the codebase it adapts:

```bibtex
@article{silva2012simple,
  title   = {A simple and effective metaheuristic for the minimum latency problem},
  author  = {Silva, Marcos Melo and Subramanian, Anand and Vidal, Thibaut and Ochi, Luiz Satoru},
  journal = {European Journal of Operational Research},
  volume  = {221}, number = {3}, pages = {513--520}, year = {2012}
}
```

Codebase adapted from: <https://github.com/franciscunha/MLP>
