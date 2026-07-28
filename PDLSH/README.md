# PDLSH — GPU baseline for the ROAR paper

This folder contains the **PDLSH** (Parallel Deterministic Local Search Heuristic) baseline used in
the ROAR paper (Sections V-A and V-C, Tables I & II).

PDLSH is due to:

> P. Yelmewad and B. Talawar,
> *"Parallel deterministic local search heuristic for minimum latency problem"*,
> Cluster Computing **24**(2), 969–995, 2021.

---

## 1. Provenance — read this first

`source/PDLSH.cu`, `source/PDLSH_v1.cu`, `source/DLSH.c`, `LICENSE` (GPL-3.0) and `.gitignore` are
the **authors' official release**, imported unmodified.

`source/time_version.cu` is the **revised file** produced for the ROAR paper. It keeps the original
solver *exactly* and changes only the objective function plus a handful of instrumentation features
(full list in [§4](#4-what-time_versioncu-changes)).

`source/benchmark.py` and `TSPLIB/` were added for this study.

> **Why a revision was needed.** The paper's Section V-C: the official implementation departs from
> the standard TRP metric of Silva et al. and Abeledo et al. in two ways — its Euclidean routine
> **truncates** `sqrtf` into an integer instead of rounding, and its scorer **omits the return to the
> depot**. Both deflate the objective, which is why the published results contain "new best-known"
> latencies for instances such as `berlin52` whose optima are *proven*. `time_version.cu` restores
> nearest-integer rounding and depot-inclusive summation so that all three methods in this repository
> are scored under one objective.

**Licensing:** PDLSH is GPL-3.0 (see `LICENSE`). `time_version.cu` is a derivative work and inherits
that licence. The rest of this repository is not GPL — keep the boundary at this folder.

---

## 2. Contents

```
PDLSH/
├── LICENSE                     GPL-3.0, from the upstream release
├── .gitignore                  upstream (CUDA intermediates)
├── source/
│   ├── PDLSH.cu                ORIGINAL author release — the reference solver
│   ├── PDLSH_v1.cu             ORIGINAL — earlier variant, float costs, sequential-order seed
│   ├── DLSH.c                  ORIGINAL — single-threaded CPU version of the same local search
│   ├── time_version.cu         REVISED — the file actually benchmarked in the ROAR paper
│   ├── time_version            checked-in Linux x86-64 binary (rebuild it)
│   ├── benchmark.py            batch driver: deadline per instance, 10 runs, parses stdout
│   ├── PDLSH_regularDataset.txt   captured results, regular set  (benchmark.py output)
│   ├── PDLSH_largeDataset.txt     captured results, large set    (benchmark.py output)
│   ├── log_PDLSH_regularDataset.txt   per-interval incumbent trace (renamed from log_PDLSH.txt)
│   └── log_PDLSH_largeDataset.txt     ditto
└── TSPLIB/                     26 .tsp — the 20 regular + 6 large instances
```

`PDLSH_v1.cu` and `DLSH.c` are kept for provenance only; neither is used in the benchmark.

---

## 3. Building and running

### Requirements

CUDA toolkit with `nvcc`, a CUDA-capable GPU, and C++11 (`<thread>`, `<atomic>`, `<chrono>`).
The paper used an NVIDIA RTX 4060 (8 GB). `time_version.cu` aborts at startup if
`cudaGetDeviceCount()` returns 0.

Device memory: the "two-pass reduction" path allocates `min_dst_data * (blk + 1)` where
`blk = ceil(n(n-1)/2 / 256)` — about 33 MB at `n = 33,810`. Modest. The unified-memory route and
coordinate arrays are `O(n)`.

### Build

```bash
cd PDLSH/source
nvcc -O3 -std=c++11 time_version.cu -o time_version
```

If your GPU needs an explicit architecture, add e.g. `-arch=sm_89` (Ada / RTX 40-series).

To build the untouched original for comparison:

```bash
nvcc -O3 PDLSH.cu -o pdlsh_original
```

### Single instance

```bash
./time_version <path/to/instance.tsp> <time_limit_seconds>
```

Example:

```bash
./time_version ../TSPLIB/berlin52.tsp 100
```

The second argument is optional and defaults to `60.0`. Output:

```
[INFO] Detected 1 CUDA-capable GPU(s). Proceeding with computations...
[LOGGER] Initialized log writer. Interval set to 1.00 seconds.
../TSPLIB/berlin52.tsp   154211  Valid
[2-OPT] Improved Cost: 151883
...
=== BEST ROUTE FOUND ===
0, 21, 30, ...
========================
--- Absolute Best Final Cost: 148325 ---
Time Elapsed: 100.004 seconds
```

A detached logger thread also appends an incumbent trace to **`log_PDLSH.txt`** in the working
directory, one line per interval (`1 s` below 1,000 nodes, scaling linearly to `30 s` at 100,000).
The two committed `log_PDLSH_*Dataset.txt` files are that file renamed after each partition.

### Full benchmark

```bash
cd PDLSH/source
python3 benchmark.py
```

`benchmark.py` runs each instance `NUM_EXECUTIONS = 10` times with
`timeout = 100 + (n - 1000)/110` (`100.0` below 1,000 nodes) — **the same `T(N)` as ROAR and
GILS-RVND** — and appends `Best Cost`, `Best Route` and `Actual Runtime` to `LOG_FILE`.

Before running, edit the two switches at the top of the file:

```python
LOG_FILE = "PDLSH_largeDataset.txt"   # or "PDLSH_regularDataset.txt"
DATASETS = [ ... ]                     # the regular list is commented out; swap the blocks
```

The regular-set list is present but commented out at lines 15–36; the large-set list is live.
Output is **appended**, so move the old file aside first.

To feed `../../processingData`, copy `PDLSH_regularDataset.txt` and `PDLSH_largeDataset.txt` there.

---

## 4. What `time_version.cu` changes

The verification target for this file is: **the solver logic must be exactly `PDLSH.cu`'s, except for
the cost function and added features.** That holds. Detailed diff below.

### 4.1 Solver structure — unchanged

Every kernel, every reduction, and the entire `main()` control flow are structurally identical to
`PDLSH.cu`:

| Kernel | Role | Status |
|---|---|---|
| `swap` / `swap_loc` / `swap_loc_one` | swap-move evaluation (built-in atomic / two-pass shared-memory / one-pass global reduction) | identical control flow |
| `two_opt` / `two_opt_loc` / `two_opt_loc_one` | 2-opt move evaluation via `get_route_dst` | identical control flow |
| `find_min`, `fill_dst_arr`, `find_min_cpu` | reduction helpers | identical |
| `nn_route` | nearest-neighbour seed | identical construction, different accumulator (§4.2) |
| `arrange_route`, `route_checker`, `print_route` | host helpers | identical |

The `(i, j)` decoding from the linear thread id, the `if(i)` / `if(i && i != j-1)` move filters, the
shared-memory tree reduction with its odd-size `fact++` fixup, the outer
`while(flag) { swap-descent; two-opt-descent; if (dst < ldst) flag = 1; }` loop, and the strictly
improving acceptance are all byte-equivalent modulo the type widening in §4.3.

### 4.2 Cost function — depot-inclusive, TSPLIB rounding  **[the intended change]**

**(a) `distD` (L138–175).** Replaced by a TSPLIB-compliant, metric-aware routine:

| | `PDLSH.cu` | `time_version.cu` |
|---|---|---|
| EUC_2D | `(long)sqrtf(dx*dx+dy*dy)` — **truncates** | `(long long)(sqrt(...) + 0.5)` — nearest integer |
| precision | `float` math | `double` math |
| CEIL_2D | not supported | `ceil(...)` |
| ATT | not supported | TSPLIB pseudo-Euclidean |
| GEO | not supported | TSPLIB great circle |

The metric is selected from the file header at startup into `__device__ __managed__ global_ewt`
(L659–680), so both host and device read the same value with no divergence inside a kernel.

**(b) Latency weights.** Under the depot-**exclusive** objective the edge entering position `t`
carries weight `n - t`; under depot-**inclusive** it carries `n - t + 1`, and a final
`c(route[n-1], route[0])` term with weight 1 is added. Every weight in the file is shifted
accordingly:

| Site | `PDLSH.cu` | `time_version.cu` |
|---|---|---|
| `nn_route` accumulator (L569–572) | `(n-j) * d(...)` | `(n-j+1) * d(...)`, then `+= d(route[n-1], route[0])` |
| `get_route_dst` prefix `d1` (L334) | `(n-y)` | `(n-y+1)` |
| `get_route_dst` suffix `d2` (L336) | `(n-y)` | `(n-y+1)` |
| `get_route_dst` reversed `d3` (L338) | `(n-z-1)` | `(n-z)` |
| `get_route_dst` return leg (L342–345) | absent | `last_node = (j == n-1) ? route[i+1] : route[n-1]`, `+ d(last_node, route[0])` |
| swap kernels, general case | `(n-i), (n-i-1), (n-j), (n-j-1)` | `(n-i+1), (n-i), (n-j+1), (n-j)` |
| swap kernels, `i == j-1 && j < n-1` | `(n-i), (n-i-2)` | `(n-i+1), (n-i-1)` |
| swap kernels, `i == j-1 && j == n-1` | one term, no depot edge | two terms, `rt[0]` as the successor |
| swap kernels, `j == n-1` in the general case | reads `rt[j+1]` out of range | `next_j = (j == n-1) ? rt[0] : rt[j+1]` |

The last row also fixes a genuine **out-of-bounds read** in the original: when `j == n - 1` the
general branch of `swap`/`swap_loc`/`swap_loc_one` dereferences `rt[n]`.

All three swap kernels received the identical edit, so the three reduction strategies remain
interchangeable.

### 4.3 Type widening

`long -> long long` throughout, and every weight multiplication is explicitly cast
(`(long long)(n-i+1) * distD(...)`). Necessary because depot-inclusive latencies on the large set
reach `1.1 x 10^12`.

### 4.4 Added features

| Feature | Lines | Notes |
|---|---|---|
| **Watchdog / deadline** | 43–50, 643–653, `CHECK_DEADLINE()` at the head of every descent loop | `argv[2]` seconds; sets an atomic `stop_flag`, and the macro `goto end_of_search`. The route array and `dst` are always in sync at those points, so the reported result is a valid tour. |
| **Interval logger** | 65–112 | Detached thread, writes `Time: t \| Cost: Z` to `log_PDLSH.txt` at `get_log_interval(n)` seconds. Mirrors ROAR's `E(N)`. |
| **GPU presence check** | 622–630 | Hard abort if no CUDA device. |
| **Non-interactive mode** | 781, 853 | `ch1 = 2; ch2 = 3;` hard-codes *Data Vector + Two-pass Reduction*. The original prompts for both on stdin, which is unusable under `subprocess`. |
| **Route + timing output** | 1169–1191 | `=== BEST ROUTE FOUND ===` block, `Absolute Best Final Cost`, `Time Elapsed`, in the shape `benchmark.py` parses. `clock()` replaced by `steady_clock` (`clock()` measures CPU time, which is wrong for a GPU-bound wall-clock budget). |
| **Header pre-scan** | 659–680 | Reads `EDGE_WEIGHT_TYPE`, then `rewind()`s so the original parser runs unchanged. |

### 4.5 Verification of the depot-inclusive rewrite

For a tour `route[0..n-1]` with edges `e_t = c(route[t-1], route[t])`:

```
Z_inclusive = sum_{t=1}^{n-1} (n - t + 1) * e_t  +  c(route[n-1], route[0])
```

* `nn_route` computes exactly this.
* `get_route_dst(i, j)` computes it for the tour with `[i+1, j]` reversed: `d1` covers positions
  `1..i` unchanged; `d3` walks the reversed block assigning new positions `i+1..j` the weights
  `n-i ... n-j+1`; `d2` covers `j+1..n-1` unchanged; the explicit return term closes the cycle.
* The swap kernels' four (or two) touched edges sit at positions `i, i+1, j, j+1`, giving weights
  `n-i+1, n-i, n-j+1, n-j`.

All three agree, so the incremental `change` deltas and the absolute `get_route_dst` recomputes stay
consistent across the descent — which matters, since the outer loop alternates between them.

---

## 5. Known quirks and caveats

1. **Reduction mode 1 (`atomicMin`) overflows above ~4.3 x 10^9.**
   `atomicMin(dst_tid, ((unsigned long long)cost << 32) | id)` packs the cost into the high 32 bits.
   Latencies on the regular set above `dsj1000` already exceed `2^32`, so this path silently
   corrupts. It is present in both the original and the revision; `time_version.cu` pins
   `ch1 = 2` and never uses it. **Do not switch to mode 1 for anything but tiny instances.**

2. **`long long sol = n * (n - 1) / 2;` is evaluated in `int` arithmetic** (L744) because `n` is an
   `int`. Safe up to `n ~ 65,535` (`33,810` gives `1.14 x 10^9`, within range), but it will overflow
   silently beyond that. Same in the original. Cast `n` if you go larger.

3. **Coordinates are stored as `float`** (`posx`, `posy`) even though `distD` now computes in
   `double`. All instances used here have coordinates below `2^24`, so they are represented exactly
   and no error is introduced — but an instance with large *fractional* coordinates would still lose
   precision relative to ROAR/GILS-RVND, which use `double` throughout.

4. **`best_route_snapshot` / `route_mutex` are dead** (L39–40, L766–769). The `SYNC_BEST_ROUTE()`
   macro was reduced to `global_best_cost.store(dst);` and no longer copies the route, contradicting
   the comment block above it (L52–58). The final route is printed straight from `route`, which is
   correct — but the comment is misleading and the vector/mutex should be deleted.

5. **`printf("[2-OPT] Improved Cost: ...")` also fires inside the swap descent** (e.g. L895, L972,
   L1093). Mislabelled; both move types print `[2-OPT]`.

6. **The logger writes to a fixed `log_PDLSH.txt`** and appends. `benchmark.py` does not rename it,
   so consecutive partitions land in the same file — rename it manually between the regular and large
   runs, as was done for the two committed logs.

7. **`route` is allocated twice** (L745–746): `malloc` then `cudaMallocManaged` over the same
   pointer, leaking the first. Inherited from the original; harmless in practice.

8. **`benchmark.py` has no upper cap on the timeout.** `get_search_timeout` implements only the
   linear branch of `T(N)`; it would exceed 1,000 s beyond 100,000 nodes. PDLSH is never run past
   33,810 nodes in the paper, so the omission does not affect the reported results.

9. **`route_checker` runs only on the seed**, not on the returned tour. Since 2-opt and swap are
   permutation-preserving this is fine, but there is no final feasibility assertion.

---

## 6. Citation

```bibtex
@article{yelmewad2021parallel,
  title   = {Parallel deterministic local search heuristic for minimum latency problem},
  author  = {Yelmewad, Pramod and Talawar, Basavaraj},
  journal = {Cluster Computing},
  volume  = {24}, number = {2}, pages = {969--995}, year = {2021}
}
```

Original source release: the authors' public repository. `time_version.cu` is a derivative work under
GPL-3.0 (`LICENSE`).
