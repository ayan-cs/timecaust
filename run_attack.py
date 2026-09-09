"""
run_attack.py
Single entry point for the TimeCAT attack. Reads RUN_DATASETS and RUN_SEEDS
from config.py -- nothing is passed on the command line. Each dataset must
already have a trained surrogate (python run_model.py) before it can be
attacked; attack.py reads that dataset's registry.json to find which
checkpoint to use.

    python run_attack.py

Same in-process-if-one-job / subprocess-per-job dispatch as run_model.py;
see that file's docstring for the reasoning, including why subprocess runs
just inherit this process's own stdout/stderr instead of being captured to
a logs_all/seed<S>/ file -- attack.py's own run_attack_grid already writes
every line to that run's own artifacts/.../global_logs.log via
utils.common.get_logger (which dual-writes to both stdout and that file),
so a second copy here was pure duplication, and was silently never created
at all whenever RUN_DATASETS/RUN_SEEDS was trimmed to a single job (the
in-process branch below never touched it either).
"""

from __future__ import annotations

import os
import subprocess
import sys
import traceback

import config
from utils.common import Timer, fmt_seconds, timestamp


def _jobs() -> list[tuple[str, int]]:
    unknown = [d for d in config.RUN_DATASETS if d not in config.DATASETS]
    if unknown:
        raise ValueError(f"RUN_DATASETS contains unknown dataset(s) {unknown}; "
                          f"known datasets: {sorted(config.DATASETS)}")
    return [(d, s) for d in config.RUN_DATASETS for s in config.RUN_SEEDS]


def _run_inprocess(dataset: str, seed: int) -> int:
    import attack  # local import: only needed for the in-process path
    try:
        summary = attack.run_dataset_seed(dataset, seed)
        print(f"[done] {dataset} seed{seed}: {summary['n_combinations']} combinations x "
              f"{len(summary['target_dims'])} dims, {summary['n_failed']} failed "
              f"-> {summary.get('checkpoint_path')}")
        return 0
    except Exception:
        print(f"[FAILED] {dataset} seed{seed}\n{traceback.format_exc()}", file=sys.stderr)
        return 1


def _run_subprocess(dataset: str, seed: int, env: dict) -> int:
    job_env = {**env, "TCAT_DATASET": dataset, "TCAT_SEED": str(seed)}
    print(f"\nAttacking: {dataset} | seed {seed} | started {timestamp()}", flush=True)
    # No stdout/stderr redirection: the subprocess inherits this process's
    # own streams, so its output (and that run's own global_logs.log,
    # written by attack.py itself) is the only place it's recorded -- see
    # the module docstring above.
    proc = subprocess.Popen([sys.executable, "-m", "attack"], env=job_env)
    return proc.wait()


def main() -> int:
    jobs = _jobs()
    if not jobs:
        print("no (dataset, seed) jobs queued; check RUN_DATASETS / RUN_SEEDS in config.py")
        return 1

    if len(jobs) == 1:
        dataset, seed = jobs[0]
        return _run_inprocess(dataset, seed)

    print(f"{len(jobs)} jobs queued: {jobs}", flush=True)
    env = os.environ.copy()
    if ":" in config.DEVICE:
        env["CUDA_VISIBLE_DEVICES"] = config.DEVICE.split(":", 1)[1]
        env["TCAT_DEVICE"] = "cuda"
    failed = []

    with Timer() as total:
        for dataset, seed in jobs:
            try:
                with Timer() as job_timer:
                    code = _run_subprocess(dataset, seed, env)
            except KeyboardInterrupt:
                print("\n[INTERRUPTED] Ctrl+C received. Stopping run_attack.py.", flush=True)
                return 130
            if code == 0:
                print(f"[OK] {dataset} seed{seed} finished in {fmt_seconds(job_timer.elapsed)}", flush=True)
            else:
                print(f"[ERROR] {dataset} seed{seed} failed (exit code {code}) in {fmt_seconds(job_timer.elapsed)}", flush=True)
                failed.append((dataset, seed))

    print(f"\ntotal wall time: {fmt_seconds(total.elapsed)} for {len(jobs)} job(s)", flush=True)
    print(f"failed jobs: {failed if failed else None}", flush=True)
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
