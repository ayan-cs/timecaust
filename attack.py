"""
attack.py
TimeCAT attack driver, consolidated from TimeCAT_new's seven near-identical
timecat_<dataset>.py scripts (same duplication pattern as
crvae_model/train.py's seven train_<dataset>.py: only the dataset/model-
artifact strings and the combination index to load differed). For a trained
surrogate checkpoint (found automatically by reading the dataset's
registry.json -- written by crvae_model.train, but read here with nothing
but a plain json.load -- instead of a hand-typed model_artifact string +
checkpoint_comb<i>.pt path), this runs the PGD attack from
utils.attack_utils under three channel-selection conditions -- parent-only
(TimeCAT itself), non-parent-only, and several random-mask control trials --
for every (target channel, hyperparameter combination) in
config.ATTACK_PARAM_GRID, on both the train and val splits.

This file, and everything it imports from utils/, is the attack stage. It
never imports crvae_model.train or crvae_model.utils -- the only things it
takes from the surrogate-training side are crvae_model.model (the cLSTM
class itself, needed to instantiate the architecture a checkpoint's weights
get loaded into; see utils.data_utils.load_model) and the checkpoint.pt
file on disk. See utils/common.py's module docstring for the full
training/attack split.

One call to `run_attack_grid` (via `run_dataset_seed`) produces:

artifacts/seed<S>/artifacts_<dataset>/attacks/timecat__grid__seed<S>__<timestamp>/
    global_logs.log            every combination's log lines, in one file (the
                                original timecat_<dataset>.py scripts redirected
                                sys.stdout to a fresh dim_<j>/logs/attack_comb<k>.log
                                per combination instead; this project uses one
                                combined log per grid everywhere, same as
                                crvae_model/train.py's run.log)
    experiment_summary.json    which surrogate run was attacked, grid size, total runtime
    dim_<j>/
        metadata/
            comb<k>_metadata.json      full per-combination results (both splits,
                                        all three conditions, ordering checks)
            all_metadata.jsonl         same, one JSON line per combination
            *_vectors.npz              per-instance inputs/attack_vectors/
                                        ground_truths/pred_clean/pred_adv for
                                        each (combination, split, condition)

Run `python analyze_results.py --exp_dir <that folder>` afterwards to rank
combinations and get results_*.csv / best_combinations_*.json.
"""

from __future__ import annotations

import traceback
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset

import config
from utils import common, data_utils
from utils.attack_utils import run_condition, save_attack_artifacts


def _windows_to_loader(windows, batch_size):
    lefts, rights = zip(*(data_utils.split_window(np.array(w)) for w in windows))
    lefts, rights = np.array(lefts), np.array(rights)
    dl = DataLoader(TensorDataset(torch.FloatTensor(lefts), torch.FloatTensor(rights)), batch_size=batch_size)
    return dl, rights


def _aggregate_random_trials(trial_list):
    agg = {}
    for key in ['css', 'cga', 'pe', 'mse_tgt', 'mae_tgt', 'prs', 'mse_nt', 'mae_nt']:
        vals = [r[key] for r in trial_list]
        agg[f'{key}_mean'] = float(np.mean(vals))
        agg[f'{key}_std'] = float(np.std(vals))
    return agg


def run_attack_grid(dataset: str, seed: int, artifacts_root: str | Path) -> dict:
    ds_cfg = config.DATASETS[dataset]
    npz_path, context = ds_cfg["npz"], ds_cfg["context"]

    ds_dir = common.dataset_artifact_dir(artifacts_root, dataset)
    registry_path = ds_dir / "registry.json"
    if not registry_path.exists():
        raise RuntimeError(f"no surrogate registry at {registry_path}; "
                            f"train a surrogate first (python run_model.py) for dataset={dataset} seed={seed}")
    registry = common.load_json(registry_path)
    surrogate_entry = registry["selected"]
    checkpoint_path = surrogate_entry["checkpoint_path"]

    run_dir = common.unique_dir(ds_dir / "attacks" / common.build_run_name("timecat", config.ATTACK_SEC_TAG, seed))
    run_dir.mkdir(parents=True, exist_ok=True)
    logger = common.get_logger(f"timecat.{dataset}.seed{seed}", run_dir / "global_logs.log")
    device = common.resolve_device(config.DEVICE)
    logger.info("=" * 88)
    logger.info(f"TimeCauST attack | dataset={dataset} | seed={seed} | device={device} | started {common.timestamp()}")
    logger.info(f"run_dir={run_dir}")
    logger.info(f"attacking surrogate run={surrogate_entry['run_name']} | checkpoint={checkpoint_path}")

    common.set_deterministic(seed)
    model = data_utils.load_model(checkpoint_path, device)
    n_dim = model.n_dim
    causal_graph = model.causal_graph

    X, _meta = data_utils.load_trajectory(npz_path, version=config.CRVAE_TRAIN["data_version"])
    X_train, X_val, _X_test = data_utils.create_split_windows(X, context=context, val_frac=config.CRVAE_TRAIN["val_frac"], test_frac=config.CRVAE_TRAIN["test_frac"], logger=logger)

    trials = config.expand_attack_grid()
    target_dims = config.ATTACK_CONFIG["target_dims"] or list(range(n_dim))
    n_random_trials = config.ATTACK_CONFIG["n_random_trials"]
    logger.info(f"n_dim={n_dim} | target_dims={target_dims} | {len(trials)} hyperparameter combinations | "
                f"{n_random_trials} random trials/combination")

    n_failed = 0
    with common.Timer() as exp_timer:
        for j in target_dims:
            parent_mask = data_utils.get_parent_mask(causal_graph, j)
            nonparent_mask = data_utils.get_nonparent_mask(causal_graph, j)
            n_parents = int(parent_mask.sum())

            dim_dir = run_dir / f"dim_{j}"
            metadata_dir = dim_dir / "metadata"
            metadata_dir.mkdir(parents=True, exist_ok=True)

            logger.info("=" * 60)
            logger.info(f"target dim {j} | parents={np.where(parent_mask)[0].tolist()} ({n_parents} channels)")

            dim_metadata_list = []
            for i, hp in enumerate(trials, start=1):
                lam_ntgt, lam_smooth, epsilon, alpha, pgd_steps, batch_size = (
                    hp["lam_ntgt"], hp["lam_smooth"], hp["epsilon"], hp["alpha"], hp["pgd_steps"], hp["batch_size"]
                )
                logger.info(f"combination {i:04d}/{len(trials)} | dim {j} | hp={hp}")

                try:
                    train_dl, X_train_right = _windows_to_loader(X_train, batch_size)
                    val_dl, X_val_right = _windows_to_loader(X_val, batch_size)
                    hparams = dict(epsilon=epsilon, alpha=alpha, pgd_steps=pgd_steps,
                                   lam_ntgt=lam_ntgt, lam_smooth=lam_smooth, device=device)

                    parent_tr = run_condition(model, train_dl, X_train_right, j, parent_mask, "parent_only", **hparams)
                    parent_val = run_condition(model, val_dl, X_val_right, j, parent_mask, "parent_only", **hparams)
                    save_attack_artifacts(parent_tr, metadata_dir, f"comb{i}_train_parent")
                    save_attack_artifacts(parent_val, metadata_dir, f"comb{i}_val_parent")

                    nonparent_tr = run_condition(model, train_dl, X_train_right, j, nonparent_mask, "nonparent_only", **hparams)
                    nonparent_val = run_condition(model, val_dl, X_val_right, j, nonparent_mask, "nonparent_only", **hparams)
                    save_attack_artifacts(nonparent_tr, metadata_dir, f"comb{i}_train_nonparent")
                    save_attack_artifacts(nonparent_val, metadata_dir, f"comb{i}_val_nonparent")

                    rand_tr_list, rand_val_list = [], []
                    for t in range(n_random_trials):
                        random_mask = data_utils.get_random_mask(n_dim, n_parents, exclude_dim=j, seed=t * 100 + j)
                        r_tr = run_condition(model, train_dl, X_train_right, j, random_mask, f"random_trial_{t+1}", **hparams)
                        r_val = run_condition(model, val_dl, X_val_right, j, random_mask, f"random_trial_{t+1}", **hparams)
                        save_attack_artifacts(r_tr, metadata_dir, f"comb{i}_train_rand{t+1}")
                        save_attack_artifacts(r_val, metadata_dir, f"comb{i}_val_rand{t+1}")
                        rand_tr_list.append(r_tr)
                        rand_val_list.append(r_val)

                    rand_agg_tr = _aggregate_random_trials(rand_tr_list)
                    rand_agg_val = _aggregate_random_trials(rand_val_list)

                    def _ordering(metric, p, np_, r):
                        return {"parent": p[metric], "nonparent": np_[metric], "random_mean": r[f"{metric}_mean"],
                                "expected_P_gt_R_gt_NP": p[metric] > r[f"{metric}_mean"] > np_[metric]}

                    comb_metadata = {
                        "combination": i, "dataset": dataset, "target_dim": j,
                        "parent_channels": np.where(parent_mask)[0].tolist(), "n_dim": n_dim,
                        "hyperparameters": {**hp, "n_random_trials": n_random_trials},
                        "results_train_set": {"parent_only": parent_tr, "nonparent_only": nonparent_tr,
                                               "random_trials": rand_tr_list, "random_aggregated": rand_agg_tr},
                        "results_val_set": {"parent_only": parent_val, "nonparent_only": nonparent_val,
                                             "random_trials": rand_val_list, "random_aggregated": rand_agg_val},
                        "ordering_train_set": {m: _ordering(m, parent_tr, nonparent_tr, rand_agg_tr) for m in ["css", "pe"]},
                        "ordering_val_set": {m: _ordering(m, parent_val, nonparent_val, rand_agg_val) for m in ["css", "pe"]},
                    }
                    logger.info(f"  VAL parent   css={parent_val['css']:.4f} pe={parent_val['pe']:.4f} "
                                f"mse_tgt={parent_val['mse_tgt']:.6f}")
                    logger.info(f"  VAL nonparent css={nonparent_val['css']:.4f} pe={nonparent_val['pe']:.4f} "
                                f"mse_tgt={nonparent_val['mse_tgt']:.6f}")
                    logger.info(f"  VAL random    css={rand_agg_val['css_mean']:.4f}+-{rand_agg_val['css_std']:.4f} "
                                f"pe={rand_agg_val['pe_mean']:.4f}+-{rand_agg_val['pe_std']:.4f}")

                    common.save_json(comb_metadata, metadata_dir / f"comb{i}_metadata.json")
                    dim_metadata_list.append(comb_metadata)
                except Exception:
                    logger.error(f"[FAILED] dim {j} combination {i:04d} | hp={hp}\n{traceback.format_exc()}")
                    n_failed += 1
                    continue
                finally:
                    if device.type == "cuda":
                        torch.cuda.empty_cache()

            with open(metadata_dir / "all_metadata.jsonl", "w") as f:
                for obj in dim_metadata_list:
                    f.write(common.dumps(obj) + "\n")

    exp_summary = {
        "dataset": dataset, "seed": seed, "surrogate_run": surrogate_entry["run_name"],
        "checkpoint_path": checkpoint_path, "npz": npz_path, "n_dim": n_dim,
        "target_dims": target_dims, "n_combinations": len(trials), "n_random_trials": n_random_trials,
        "n_failed": n_failed, "total_time_s": round(exp_timer.elapsed, 2),
        "param_grid": config.ATTACK_PARAM_GRID, "env": common.env_info(),
    }
    common.save_json(exp_summary, run_dir / "experiment_summary.json")
    logger.info("=" * 88)
    logger.info(f"attack grid finished in {common.fmt_seconds(exp_timer.elapsed)} | "
                f"{n_failed} failed cells | artifacts at {run_dir}")
    return exp_summary


def run_dataset_seed(dataset: str, seed: int, artifacts_root: str | Path | None = None) -> dict:
    if dataset not in config.DATASETS:
        raise ValueError(f"unknown dataset {dataset!r}; expected one of {sorted(config.DATASETS)}")
    root = Path(artifacts_root) if artifacts_root else common.seed_artifact_root(seed, base=config.ARTIFACTS_DIR)
    return run_attack_grid(dataset, seed, root)


if __name__ == "__main__":
    import os
    _dataset = os.environ.get("TCAT_DATASET", config.RUN_DATASETS[0])
    _seed = int(os.environ.get("TCAT_SEED", config.RUN_SEEDS[0]))
    run_dataset_seed(_dataset, _seed)
