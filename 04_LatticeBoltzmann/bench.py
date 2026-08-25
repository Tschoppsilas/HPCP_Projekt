#!/usr/bin/env python3
"""
bench.py -- run baseline vs. an optimized LBM implementation, compare timing
and correctness, and keep a running log across repeated runs.

Usage
-----
    python bench.py --case cylinder --size small
    python bench.py --case cavity --size small --label numba
    python bench.py --case cylinder --size medium --optimized-script optimized/lbm_d2q9_numba.py

What it does, every time you run it:
  1. Baseline reference for this case/size is (re)used from data/ref_<case>_<size>.npz
     if it already exists, otherwise it runs the baseline once to create it.
  2. Runs your optimized script fresh, writes its result to a NEW timestamped file
     under results/ (so old runs are never overwritten -- you keep a history).
  3. Runs validate.py to check correctness against the baseline.
  4. Prints a summary (runtime, MLUPS, speedup, correctness) and appends one row
     to results/benchmark_log.csv, so you have every run logged for the write-up.

Nothing here changes the physics or the two solver scripts -- this is only a
harness around them.
"""

import argparse
import csv
import re
import subprocess
import sys
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parent

RUNTIME_RE = re.compile(r"runtime\s*:\s*([\d.]+)\s*s")
MLUPS_RE = re.compile(r"MLUPS\s*:\s*([\d.]+)")
REGRESSION_RE = re.compile(r"^\s*(\w+):\s*(PASS|FAIL)\s+max rel err\s+([\d.eE+-]+)", re.MULTILINE)


def run_solver(script, case, size, out_path, quiet=True):
    """Run a solver script (baseline or optimized), return (ok, stdout, runtime_s, mlups)."""
    cmd = [sys.executable, str(script), "--case", case, "--size", size, "--out", str(out_path)]
    if quiet:
        cmd.append("--quiet")
    print(f"  $ {' '.join(cmd)}")
    result = subprocess.run(cmd, capture_output=True, text=True, cwd=ROOT)
    ok = result.returncode == 0
    stdout = result.stdout
    if not ok:
        print(stdout)
        print(result.stderr)
        return False, stdout, None, None

    rt = RUNTIME_RE.search(stdout)
    ml = MLUPS_RE.search(stdout)
    runtime_s = float(rt.group(1)) if rt else None
    mlups = float(ml.group(1)) if ml else None
    return True, stdout, runtime_s, mlups


def run_validate(baseline_npz, candidate_npz, rtol):
    """Run validate.py, return (ok, stdout, per_field_errors)."""
    validate_script = ROOT / "baseline" / "validate.py"
    cmd = [sys.executable, str(validate_script),
           "--baseline", str(baseline_npz), "--candidate", str(candidate_npz),
           "--rtol", str(rtol)]
    print(f"  $ {' '.join(cmd)}")
    result = subprocess.run(cmd, capture_output=True, text=True, cwd=ROOT)
    ok = result.returncode == 0
    errors = {m.group(1): (m.group(2), float(m.group(3))) for m in REGRESSION_RE.finditer(result.stdout)}
    return ok, result.stdout, errors


def append_log(log_path, row, header):
    new_file = not log_path.exists()
    with open(log_path, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=header)
        if new_file:
            writer.writeheader()
        writer.writerow(row)


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--case", choices=["cylinder", "cavity"], default="cylinder")
    p.add_argument("--size", choices=["tiny", "small", "medium", "large"], default="small")
    p.add_argument("--baseline-script", default="baseline/lbm_d2q9.py")
    p.add_argument("--optimized-script", default="optimized/lbm_d2q9_numba.py")
    p.add_argument("--label", default="numba", help="tag used in output filenames and the log")
    p.add_argument("--rtol", type=float, default=1e-6)
    p.add_argument("--force-baseline", action="store_true",
                    help="re-run the baseline even if a reference file already exists")
    p.add_argument("--verbose", action="store_true", help="show per-step progress from the solvers")
    args = p.parse_args()

    data_dir = ROOT / "data"
    results_dir = ROOT / "results"
    data_dir.mkdir(exist_ok=True)
    results_dir.mkdir(exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    baseline_npz = data_dir / f"ref_{args.case}_{args.size}.npz"
    candidate_npz = results_dir / f"{args.case}_{args.size}_{args.label}_{timestamp}.npz"

    quiet = not args.verbose

    # 1) baseline reference (reused if it already exists)
    if baseline_npz.exists() and not args.force_baseline:
        print(f"[1/3] Baseline reference already exists, reusing: {baseline_npz.name}")
        base_ok, base_runtime, base_mlups = True, None, None
    else:
        print(f"[1/3] Running baseline ({args.case}, {args.size}) ...")
        base_ok, _, base_runtime, base_mlups = run_solver(
            ROOT / args.baseline_script, args.case, args.size, baseline_npz, quiet=quiet)
        if not base_ok:
            print("Baseline run FAILED -- aborting.")
            sys.exit(1)
        print(f"      runtime {base_runtime:.2f} s | {base_mlups:.3f} MLUPS")

    # 2) optimized run, always fresh, always a new file
    print(f"[2/3] Running optimized ({args.label}, {args.case}, {args.size}) ...")
    opt_ok, _, opt_runtime, opt_mlups = run_solver(
        ROOT / args.optimized_script, args.case, args.size, candidate_npz, quiet=quiet)
    if not opt_ok:
        print("Optimized run FAILED.")
        row = {"timestamp": timestamp, "case": args.case, "size": args.size, "label": args.label,
               "baseline_runtime_s": base_runtime, "baseline_mlups": base_mlups,
               "opt_runtime_s": None, "opt_mlups": None, "speedup": None,
               "correctness": "ERROR", "candidate_file": str(candidate_npz.name)}
        append_log(results_dir / "benchmark_log.csv", row, list(row.keys()))
        sys.exit(1)
    print(f"      runtime {opt_runtime:.2f} s | {opt_mlups:.3f} MLUPS")

    # 3) correctness check against the baseline reference
    print("[3/3] Validating against baseline ...")
    val_ok, val_stdout, errors = run_validate(baseline_npz, candidate_npz, args.rtol)
    print(val_stdout)

    speedup = (base_runtime / opt_runtime) if (base_runtime and opt_runtime) else None

    print("=" * 60)
    print(f"case={args.case} size={args.size} label={args.label}")
    if base_runtime:
        print(f"baseline : {base_runtime:.2f} s  ({base_mlups:.3f} MLUPS)")
    else:
        print("baseline : reused reference, no fresh timing this run")
    print(f"optimized: {opt_runtime:.2f} s  ({opt_mlups:.3f} MLUPS)")
    if speedup:
        print(f"speedup  : {speedup:.2f}x")
    print(f"correctness: {'OK' if val_ok else 'FAILED'}")
    print(f"result file: {candidate_npz}")
    print("=" * 60)

    row = {"timestamp": timestamp, "case": args.case, "size": args.size, "label": args.label,
           "baseline_runtime_s": base_runtime, "baseline_mlups": base_mlups,
           "opt_runtime_s": opt_runtime, "opt_mlups": opt_mlups,
           "speedup": round(speedup, 3) if speedup else None,
           "correctness": "OK" if val_ok else "FAILED",
           "candidate_file": str(candidate_npz.name)}
    append_log(results_dir / "benchmark_log.csv", row, list(row.keys()))
    print(f"Logged to {results_dir / 'benchmark_log.csv'}")


if __name__ == "__main__":
    main()
