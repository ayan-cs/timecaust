"""
run_model.py
Single entry point for surrogate-model (cLSTM) training. Reads RUN_DATASETS
and RUN_SEEDS from config.py -- nothing is passed on the command line.

    python run_model.py

If exactly one (dataset, seed) job is queued, it trains in this process --
the same as calling train.run_dataset_seed directly, so a single run is easy
to step through with a debugger and its traceback (if any) prints straight
to this terminal.

If more than one job is queued (multiple datasets and/or multiple seeds),
jobs run sequentially, each as its own `python -m crvae_model.train`
subprocess (dataset/seed passed via TCAT_DATASET / TCAT_SEED environment
variables, since this project has no argparse) -- replacing the old
ztimecat_run_all.py, which drove seven separate per-dataset scripts the
same way. Each subprocess inherits this process's own stdout/stderr, so its
output (including a crash -- CUDA OOM, a missing dependency, anything)
prints live to this terminal exactly as it would running that dataset
in-process; nothing is captured to a separate log file here, because
crvae_model.train's own run_grid already writes every line to that run's
own artifacts/.../run.log via crvae_model.utils.common.get_logger (which
dual-writes to both stdout and that file) -- a second copy in a
logs_all/seed<S>/ directory used to exist here too, but it was pure
duplication of that same run.log whenever it was actually created, and
whenever RUN_DATASETS/RUN_SEEDS was trimmed to a single job (i.e. this file
took the in-process branch below instead of the subprocess one) it was
never created in the first place -- see this project's README for the full
writeup.

This file is training-side tooling (it only ever launches
crvae_model.train), so -- like crvae_model itself -- it imports its generic
infra (Timer/fmt_seconds/timestamp) from crvae_model.utils.common, not from
the attack stage's utils/ package at the project root. run_attack.py is its
mirror image on the attack side.
"""

from __future__ import annotations

import os
import subprocess
import sys
import traceback

import config
from crvae_model.utils.common import Timer, fmt_seconds, timestamp


def _jobs() -> list[tuple[str, int]]:
    unknown = [d for d in config.RUN_DATASETS if d not in config.DATASETS]
    if unknown:
        raise ValueError(f"RUN_DATASETS contains unknown dataset(s) {unknown}; "
                          f"known datasets: {sorted(config.DATASETS)}")
    return [(d, s) for d in config.RUN_DATASETS for s in config.RUN_SEEDS]


def _run_inprocess(dataset: str, seed: int) -> int:
    from crvae_model import train  # local import: only needed for the in-process path
    try:
        entry = train.run_dataset_seed(dataset, seed)
        print(f"[done] {dataset} seed{seed}: best_val_loss={entry['best_val_loss']:.6f} "
              f"-> {entry['checkpoint_path']}")
        return 0
    except Exception:
        print(f"[FAILED] {dataset} seed{seed}\n{traceback.format_exc()}", file=sys.stderr)
        return 1


def _run_subprocess(dataset: str, seed: int, env: dict) -> int:
    job_env = {**env, "TCAT_DATASET": dataset, "TCAT_SEED": str(seed)}
    print(f"\nRunning: {dataset} | seed {seed} | started {timestamp()}", flush=True)
    # No stdout/stderr redirection: the subprocess inherits this process's
    # own streams, so its output (and that dataset's own run.log, written by
    # crvae_model.train itself) is the only place it's recorded -- see the
    # module docstring above.
    proc = subprocess.Popen([sys.executable, "-m", "crvae_model.train"], env=job_env)
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
                print("\n[INTERRUPTED] Ctrl+C received. Stopping run_model.py.", flush=True)
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
