#!/usr/bin/env python3
"""
Aggregate TRP benchmark results for three algorithms (GILS-RVND, ROAR, PDLSH)
across a "regular" and a "large" dataset partition.

For every (dataset, algorithm) pair we compute the mean cost over runs and the
mean runtime over runs, then write two wide CSV files:

    trp_summary_regular.csv
    trp_summary_large.csv

Each output row is one dataset; columns give the average cost and average
runtime for each algorithm.

Missing results are labelled explicitly:
  * "OOM"     -> the algorithm ran but ran out of memory (GILS-RVND: a dataset
                 whose name line has no COST/TIME runs after it).
  * "NO_DATA" -> the algorithm produced no runs for this dataset for another
                 reason (e.g. the ROAR large CSV containing only headers, or the
                 dataset simply absent from that algorithm's log).

All input files are read from, and both outputs are written to, the directory
this script lives in (falling back to the current working directory when run
interactively). Adjust DATA_DIR below if your files live elsewhere.
"""

import csv
import re
import statistics
from pathlib import Path

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

# Directory containing the input files AND where outputs are written.
# Defaults to the folder this script sits in; if that can't be resolved
# (e.g. interactive session), falls back to the current working directory.
try:
    DATA_DIR = Path(__file__).resolve().parent
except NameError:
    DATA_DIR = Path.cwd()

OUTPUT_DIR = DATA_DIR  # write results next to the inputs

# Which cost column of the ROAR csv to treat as "the cost".
# FinalCost = post-(restricted-2-opt) depot-inclusive cost -> matches the paper.
ROAR_COST_FIELD = "FinalCost"   # or "RawCost"

FILES = {
    "regular": {
        "GILS-RVND": DATA_DIR / "GILS_regular.txt",
        "PDLSH":     DATA_DIR / "PDLSH_regularDataset.txt",
        "ROAR":      DATA_DIR / "trp_results_ROAR_regularDataset_modified.csv",
    },
    "large": {
        "GILS-RVND": DATA_DIR / "GILS_large.txt",
        "PDLSH":     DATA_DIR / "PDLSH_largeDataset.txt",
        "ROAR":      DATA_DIR / "trp_results_ROAR_largeDataset_modified.csv",
    },
}

# Fixed algorithm column order for the output.
ALGORITHMS = ["GILS-RVND", "ROAR", "PDLSH"]

# Labels used in output cells when there is no averaged value.
LABEL_OOM = "OOM"
LABEL_NO_DATA = "NO_DATA"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def normalize_name(raw: str) -> str:
    """Normalize a dataset name to a common key across all three sources.

    'regularInstances/a280.tsp' -> 'a280'
    'a280.tsp'                   -> 'a280'
    'a280'                       -> 'a280'
    """
    name = raw.strip()
    name = name.rsplit("/", 1)[-1]          # drop any directory prefix
    if name.lower().endswith(".tsp"):
        name = name[:-4]
    return name


def mean_or_none(values):
    """Mean of a list, or None if the list is empty."""
    return statistics.fmean(values) if values else None


# ---------------------------------------------------------------------------
# Parsers
# ---------------------------------------------------------------------------
# Each parser returns:
#   {dataset_key: {"costs": [...], "times": [...], "nodes": int|None,
#                  "oom": bool}}
# "oom" is only meaningful for GILS-RVND; for the others it stays False.

def parse_gils(path: Path) -> dict:
    """Parse a GILS-RVND log.

    Runs are delimited by 'COST:'/'TIME:' pairs. A dataset whose header line is
    followed by no such pair is flagged out-of-memory (oom=True).
    """
    results = {}
    current = None
    pending_cost = None

    tsp_re = re.compile(r"\.tsp\s*$")
    cost_re = re.compile(r"^COST:\s*([-+0-9.eE]+)")
    time_re = re.compile(r"^TIME:\s*([-+0-9.eE]+)")

    with path.open(encoding="utf-8", errors="replace") as fh:
        for line in fh:
            s = line.strip()
            if not s:
                continue

            # Dataset header: a bare path ending in .tsp (not a progress or
            # COST/TIME line).
            if tsp_re.search(s) and not s.startswith("[") \
                    and not s.startswith("COST") and not s.startswith("TIME"):
                current = normalize_name(s)
                results.setdefault(
                    current, {"costs": [], "times": [], "nodes": None, "oom": False}
                )
                pending_cost = None
                continue

            m = cost_re.match(s)
            if m and current is not None:
                pending_cost = float(m.group(1))
                continue

            m = time_re.match(s)
            if m and current is not None and pending_cost is not None:
                results[current]["costs"].append(pending_cost)
                results[current]["times"].append(float(m.group(1)))
                pending_cost = None
                continue

    # Any GILS dataset that ended up with zero runs is an OOM case.
    for entry in results.values():
        if not entry["costs"]:
            entry["oom"] = True

    return results


def parse_pdlsh(path: Path) -> dict:
    """Parse a PDLSH benchmark log."""
    results = {}

    header_re = re.compile(
        r"Dataset:\s*(?P<name>\S+)\s*\|\s*Execution:\s*(?P<exe>\d+)\s*\|\s*"
        r"Nodes:\s*(?P<nodes>\d+)"
    )
    cost_re = re.compile(r"^Best Cost:\s*([-+0-9.eE]+)")
    time_re = re.compile(r"^Actual Runtime:\s*([-+0-9.eE]+)")

    current = None
    pending_cost = None

    with path.open(encoding="utf-8", errors="replace") as fh:
        for line in fh:
            s = line.strip()
            if not s:
                continue

            m = header_re.search(s)
            if m:
                current = normalize_name(m.group("name"))
                entry = results.setdefault(
                    current, {"costs": [], "times": [], "nodes": None, "oom": False}
                )
                entry["nodes"] = int(m.group("nodes"))
                pending_cost = None
                continue

            m = cost_re.match(s)
            if m and current is not None:
                pending_cost = float(m.group(1))
                continue

            m = time_re.match(s)
            if m and current is not None and pending_cost is not None:
                results[current]["costs"].append(pending_cost)
                results[current]["times"].append(float(m.group(1)))
                pending_cost = None
                continue

    return results


def parse_roar(path: Path) -> dict:
    """Parse a ROAR results CSV (header line may repeat; may contain no data)."""
    results = {}

    with path.open(newline="", encoding="utf-8", errors="replace") as fh:
        reader = csv.reader(fh)
        for row in reader:
            if not row:
                continue
            if row[0].strip() == "Dataset":     # skip repeated headers
                continue
            if len(row) < 6:                    # expect 6 columns
                continue

            name = normalize_name(row[0])
            entry = results.setdefault(
                name, {"costs": [], "times": [], "nodes": None, "oom": False}
            )
            try:
                entry["nodes"] = int(float(row[1]))
            except ValueError:
                pass

            cost_idx = 4 if ROAR_COST_FIELD == "FinalCost" else 3
            try:
                cost = float(row[cost_idx])
                time = float(row[5])
            except ValueError:
                continue
            entry["costs"].append(cost)
            entry["times"].append(time)

    return results


PARSERS = {
    "GILS-RVND": parse_gils,
    "PDLSH":     parse_pdlsh,
    "ROAR":      parse_roar,
}


# ---------------------------------------------------------------------------
# Aggregation + output
# ---------------------------------------------------------------------------

def build_partition(partition: str):
    """Parse all three algorithms for one partition.

    Returns (ordered_dataset_keys, per_algo_results, nodes_lookup).
    """
    per_algo = {}
    nodes_lookup = {}
    dataset_order = []
    seen = set()

    for algo in ALGORITHMS:
        path = FILES[partition][algo]
        if path.exists():
            parsed = PARSERS[algo](path)
        else:
            print(f"  [warn] missing input file for {algo} ({partition}): {path}")
            parsed = {}
        per_algo[algo] = parsed

        for name, entry in parsed.items():
            if entry.get("nodes") is not None:
                nodes_lookup.setdefault(name, entry["nodes"])
            if name not in seen:
                seen.add(name)
                dataset_order.append(name)

    # Sort datasets by node count (unknown node counts sort last, then by name).
    dataset_order.sort(key=lambda n: (nodes_lookup.get(n, float("inf")), n))
    return dataset_order, per_algo, nodes_lookup


def cell_values(entry):
    """Return (cost_cell, time_cell) strings for one (dataset, algo) entry."""
    if entry and entry["costs"]:
        return (f"{mean_or_none(entry['costs']):.4f}",
                f"{mean_or_none(entry['times']):.4f}")
    # No averaged value: decide the label.
    if entry and entry.get("oom"):
        return (LABEL_OOM, LABEL_OOM)
    return (LABEL_NO_DATA, LABEL_NO_DATA)


def write_summary(partition: str):
    dataset_order, per_algo, nodes_lookup = build_partition(partition)

    header = ["Dataset", "Nodes"]
    for algo in ALGORITHMS:
        header.append(f"{algo}_AvgCost")
        header.append(f"{algo}_AvgTime(s)")

    out_path = OUTPUT_DIR / f"trp_summary_{partition}.csv"
    with out_path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.writer(fh)
        writer.writerow(header)
        for name in dataset_order:
            row = [name, nodes_lookup.get(name, "")]
            for algo in ALGORITHMS:
                cost_cell, time_cell = cell_values(per_algo[algo].get(name))
                row.append(cost_cell)
                row.append(time_cell)
            writer.writerow(row)

    return out_path, dataset_order, per_algo


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    for partition in ("regular", "large"):
        out_path, dataset_order, per_algo = write_summary(partition)
        print(f"\n=== {partition.upper()} -> {out_path} ===")
        print(f"{len(dataset_order)} datasets")
        for algo in ALGORITHMS:
            oom, nodata, have = [], [], 0
            for d in dataset_order:
                entry = per_algo[algo].get(d)
                if entry and entry["costs"]:
                    have += 1
                elif entry and entry.get("oom"):
                    oom.append(d)
                else:
                    nodata.append(d)
            line = f"  {algo:10s}: {have}/{len(dataset_order)} datasets"
            if oom:
                line += f" | OOM: {', '.join(oom)}"
            if nodata:
                line += f" | NO_DATA: {', '.join(nodata)}"
            print(line)


if __name__ == "__main__":
    main()