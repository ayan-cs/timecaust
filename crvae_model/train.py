from __future__ import annotations

import traceback
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.optim import AdamW
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader, TensorDataset

import config
from crvae_model.model import cLSTM
from crvae_model.utils import common, data_utils

def train_epoch(model, train_dl, val_dl, optimizer, beta_kl, device):
    model.train()
    agg = {"loss": 0.0, "mse": 0.0, "kl": 0.0}
    n = 0
    for X_l, X_r in train_dl:
        X_l, X_r = X_l.to(device), X_r.to(device)
        optimizer.zero_grad()
        pred, kl = model(X_l, future=X_r.shape[1])
        mse = F.mse_loss(pred, X_r)
        loss = mse + beta_kl * kl
        loss.backward()
        optimizer.step()
        bs = X_l.size(0)
        agg["loss"] += loss.item() * bs
        agg["mse"] += mse.item() * bs
        agg["kl"] += kl.item() * bs
        n += bs
    train_agg = {k: v / n for k, v in agg.items()}

    model.eval()
    agg = {"loss": 0.0, "mse": 0.0, "kl": 0.0}
    n = 0
    with torch.no_grad():
        for X_l, X_r in val_dl:
            X_l, X_r = X_l.to(device), X_r.to(device)
            pred, kl = model(X_l, future=X_r.shape[1])
            mse = F.mse_loss(pred, X_r)
            loss = mse + beta_kl * kl
            bs = X_l.size(0)
            agg["loss"] += loss.item() * bs
            agg["mse"] += mse.item() * bs
            agg["kl"] += kl.item() * bs
            n += bs
    val_agg = {k: v / n for k, v in agg.items()}

    return train_agg, val_agg


@torch.no_grad()
def eval_epoch(model, loader, beta_kl, device):
    model.eval()
    agg = {"loss": 0.0, "mse": 0.0, "kl": 0.0}
    n = 0
    for X_l, X_r in loader:
        X_l, X_r = X_l.to(device), X_r.to(device)
        pred, kl = model(X_l, future=X_r.shape[1])
        mse = F.mse_loss(pred, X_r)
        loss = mse + beta_kl * kl
        bs = X_l.size(0)
        agg["loss"] += loss.item() * bs
        agg["mse"] += mse.item() * bs
        agg["kl"] += kl.item() * bs
        n += bs
    return {k: v / n for k, v in agg.items()}

def _make_loader(windows, batch_size, shuffle):
    lefts, rights = zip(*(data_utils.split_window(np.array(w)) for w in windows))
    ds = TensorDataset(torch.FloatTensor(np.array(lefts)), torch.FloatTensor(np.array(rights)))
    return DataLoader(ds, batch_size=batch_size, shuffle=shuffle, num_workers=0)


def run_trial(hp: dict, dataset: str, seed: int, n_dim: int, causal_graph, X_train, X_val, X_test, device, logger) -> dict:
    lr, batch_size, hidden_size, num_layers, dropout, beta_kl = (
        hp["lr"], hp["batch_size"], hp["hidden_size"], hp["num_layers"], hp["dropout"], hp["beta_kl"]
    )
    logger.info(f"    lr={lr}  batch_size={batch_size}  hidden_size={hidden_size}  "
                f"num_layers={num_layers}  dropout={dropout}  beta_kl={beta_kl}")

    train_dl = _make_loader(X_train, batch_size, shuffle=False)
    val_dl = _make_loader(X_val, batch_size, shuffle=False)

    model = cLSTM(n_dim=n_dim, hidden_size=hidden_size, causal_graph=causal_graph).to(device)
    optimizer = AdamW(model.parameters(), lr=lr)
    scheduler = ReduceLROnPlateau(optimizer, mode="min", factor=config.CRVAE_TRAIN["lr_factor"], patience=config.CRVAE_TRAIN["lr_patience"])

    history: list[dict] = []
    best_epoch, best_val_loss, best_state = 0, float("inf"), None
    step_counter = 0
    epoch = -1

    with common.Timer() as trial_timer:
        for epoch in range(config.CRVAE_TRAIN["epochs"]):
            with common.Timer() as epoch_timer:
                train_agg, val_agg = train_epoch(model, train_dl, val_dl, optimizer, beta_kl, device)

            history.append({
                "epoch": epoch + 1, "lr": scheduler.get_last_lr()[0],
                "train_loss": train_agg["loss"], "train_mse": train_agg["mse"], "train_kl": train_agg["kl"],
                "val_loss": val_agg["loss"], "val_mse": val_agg["mse"], "val_kl": val_agg["kl"],
                "epoch_time_s": round(epoch_timer.elapsed, 4),
            })

            if epoch % 25 == 0 or epoch == 0:
                logger.info(f"    epoch {epoch + 1:5d} | lr {scheduler.get_last_lr()[0]:.2e} "
                            f"| train_loss {train_agg['loss']:.6f} | val_loss {val_agg['loss']:.6f} "
                            f"| {epoch_timer.elapsed:.2f}s/epoch")

            if val_agg["loss"] < best_val_loss:
                best_val_loss, step_counter, best_epoch = val_agg["loss"], 0, epoch + 1
                best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
                logger.info(f"    new best | val_loss={val_agg['loss']:.6f} @ epoch {epoch + 1}")
            else:
                step_counter += 1

            scheduler.step(val_agg["loss"])

            if step_counter >= config.CRVAE_TRAIN["patience"]:
                logger.info(f"    early stop at epoch {epoch + 1} (best epoch {best_epoch})")
                break
            if device.type == "cuda":
                torch.cuda.empty_cache()

    train_time = trial_timer.elapsed
    if best_state is None:
        raise RuntimeError(f"trial produced no valid checkpoint (val_loss never improved from +inf) "
                            f"after {epoch + 1} epoch(s); hp={hp}")
    model.load_state_dict(best_state)
    best_record = history[best_epoch - 1]

    test_dl = _make_loader(X_test, batch_size, shuffle=False)
    with common.Timer() as test_timer:
        test_agg = eval_epoch(model, test_dl, beta_kl, device)
    test_block = {**test_agg, "n_test": len(X_test), "eval_time_s": round(test_timer.elapsed, 4)}
    logger.info(f"    test | loss={test_block['loss']:.6f} | {common.fmt_seconds(test_block['eval_time_s'])}")

    model_hp = {"n_dim": n_dim, "hidden_size": hidden_size, "causal_graph": np.asarray(causal_graph).tolist()}
    metadata = {
        "dataset": dataset, "train_seed": seed, "n_dim": n_dim,
        "model": {**model_hp, "batch_size": batch_size, "num_layers": num_layers, "dropout": dropout},
        "initial_lr": lr, "earlystopper_patience": config.CRVAE_TRAIN["patience"],
        "lr_step": config.CRVAE_TRAIN["lr_patience"], "beta_kl": beta_kl,
        "final_epoch": epoch, "optimal_epoch": best_epoch, "best_val_loss": best_val_loss,
        "training_time": common.fmt_seconds(train_time), "avg_epoch_sec": train_time / (epoch + 1),
        "n_epochs_recorded": len(history), "test": test_block,
    }

    return {"model": model, "model_hp": model_hp, "metadata": metadata, "history": history, "train_time_s": train_time}


def run_grid(dataset: str, seed: int, artifacts_root: str | Path) -> dict:
    trials = config.expand_crvae_grid()
    ds_cfg = config.DATASETS[dataset]
    npz_path, context = ds_cfg["npz"], ds_cfg["context"]

    ds_dir = common.dataset_artifact_dir(artifacts_root, dataset)
    run_dir = common.unique_dir(ds_dir / common.build_run_name("crvae" or config.BASE_TAG, "grid" or config.CRVAE_SEC_TAG, seed))
    trials_dir, best_dir = run_dir / "trials", run_dir / "best"
    trials_dir.mkdir(parents=True, exist_ok=True)
    best_dir.mkdir(parents=True, exist_ok=True)

    logger = common.get_logger(f"crvae.{dataset}.seed{seed}", run_dir / "run.log")
    device = common.resolve_device(config.DEVICE)
    logger.info("=" * 88)
    logger.info(f"TimeCAT surrogate training | dataset={dataset} | seed={seed} | device={device} | started {common.timestamp()}")
    logger.info(f"run_dir={run_dir}")
    logger.info(f"npz={npz_path} | context={context} | data_version={config.CRVAE_TRAIN['data_version']} | "
                f"{len(trials)} combinations queued")

    common.set_deterministic(seed)
    X, _meta = data_utils.load_trajectory(npz_path, version=config.CRVAE_TRAIN["data_version"])
    X_train, X_val, X_test = data_utils.create_split_windows(
        X, context=context, val_frac=config.CRVAE_TRAIN["val_frac"], test_frac=config.CRVAE_TRAIN["test_frac"],
        logger=logger,
    )
    n_dim = len(X_train[0][0])
    logger.info(f"n_dim={n_dim} | context={context} | n_train={len(X_train)} | n_val={len(X_val)} | n_test={len(X_test)}")

    causal_graph = data_utils.load_ground_truth(npz_path)
    logger.info(f"causal graph loaded | shape={causal_graph.shape}")

    test_path = run_dir / "test_windows.npz"
    np.savez_compressed(test_path, X_test=np.array(X_test, dtype=np.float32))

    csv_path = run_dir / "grid_results.csv"
    results, best, failed = [], None, []
    row_keys = ["combination", "status", *config.CRVAE_GRID_KEYS,
                "best_epoch", "final_epoch", "best_val_loss", "test_loss", "train_time_s", "error"]

    with common.Timer() as grid_timer:
        for i, hp in enumerate(trials, start=1):
            logger.info("-" * 88)
            logger.info(f"combination {i:04d}/{len(trials)} | started {common.timestamp()} | hp={hp}")
            trial_dir = trials_dir / f"comb_{i:04d}"
            trial_dir.mkdir(parents=True, exist_ok=True)

            try:
                out = run_trial(hp=hp, dataset=dataset, seed=seed, n_dim=n_dim, causal_graph=causal_graph, X_train=X_train, X_val=X_val, X_test=X_test, device=device, logger=logger)
            except Exception as e:
                logger.error(f"[FAILED] combination {i:04d} | hp={hp}\n{traceback.format_exc()}")
                row = {k: None for k in row_keys}
                row.update({"combination": i, "status": "FAILED", **hp, "error": f"{type(e).__name__}: {e}"})
                common.append_csv_row(csv_path, {k: row[k] for k in row_keys})
                failed.append(i)
                continue

            history_path = trial_dir / "history.npz"
            common.save_history(history_path, out["history"])
            out["metadata"]["history_path"] = history_path.as_posix()
            common.save_checkpoint(trial_dir / "checkpoint.pt", out["model"], out["model_hp"])
            common.save_json(out["metadata"], trial_dir / "metadata.json")

            row = {k: None for k in row_keys}
            row.update({
                "combination": i, "status": "OK", **hp,
                "best_epoch": out["metadata"]["optimal_epoch"], "final_epoch": out["metadata"]["final_epoch"],
                "best_val_loss": round(out["metadata"]["best_val_loss"], 6),
                "test_loss": round(out["metadata"]["test"]["loss"], 6),
                "train_time_s": round(out["train_time_s"], 2),
            })
            common.append_csv_row(csv_path, {k: row[k] for k in row_keys})
            results.append(row)

            logger.info(f"combination {i:04d} done | best_val_loss={out['metadata']['best_val_loss']:.6f} "
                        f"| test_loss={out['metadata']['test']['loss']:.6f} "
                        f"| epochs={out['metadata']['final_epoch'] + 1} | {common.fmt_seconds(out['train_time_s'])}")

            score = out["metadata"]["best_val_loss"]
            if best is None or score < best["score"] - 1e-9:
                best = {"score": score, "combination": i, "hp": hp, "out": out}

    if best is None:
        logger.error(f"[ABORT] every combination failed for {dataset} seed {seed}; see errors above")
        raise RuntimeError(f"all {len(trials)} combinations failed for {dataset} seed {seed}; see {run_dir / 'run.log'}")

    best_history_path = best_dir / "history.npz"
    common.save_history(best_history_path, best["out"]["history"])
    best_meta = {**best["out"]["metadata"], "history_path": best_history_path.as_posix()}
    common.save_checkpoint(best_dir / "checkpoint.pt", best["out"]["model"], best["out"]["model_hp"])
    common.save_json(best_meta, best_dir / "metadata.json")

    meta = {
        "dataset": dataset, "seed": seed, "npz": npz_path, "context": context,
        "n_dim": n_dim, "device": str(device), "run_dir": run_dir.as_posix(),
        "n_train": len(X_train), "n_val": len(X_val), "n_test": len(X_test),
        "test_windows_path": test_path.as_posix(),
        "n_combinations": len(trials), "n_ok": len(results), "n_failed": len(failed),
        "failed_combinations": failed,
        "grid_time_s": round(grid_timer.elapsed, 2), "grid_time_h": round(grid_timer.elapsed / 3600, 4),
        "best_combination": best["combination"], "best_hp": best["hp"], "best_val_loss": best["score"],
        "param_grid": config.CRVAE_PARAM_GRID, "train_config": config.CRVAE_TRAIN,
        "env": common.env_info(),
    }
    common.save_json(meta, run_dir / "meta.json")

    logger.info("=" * 88)
    logger.info(f"BEST | combination {best['combination']} | hp={best['hp']} | val_loss={best['score']:.6f}")
    logger.info(f"grid finished in {common.fmt_seconds(grid_timer.elapsed)} | "
                f"{len(failed)}/{len(trials)} failed | artifacts at {run_dir}")

    entry = {
        "dataset": dataset, "seed": seed, "run_dir": run_dir.as_posix(),
        "checkpoint_path": (best_dir / "checkpoint.pt").as_posix(),
        "history_path": best_history_path.as_posix(),
        "hp": best["hp"], "best_val_loss": best["score"], "test_loss": best_meta["test"]["loss"],
        "n_dim": n_dim, "context": context,
        "n_train": len(X_train), "n_val": len(X_val), "n_test": len(X_test),
        "n_combinations": len(trials), "n_failed": len(failed),
        "grid_time_s": round(grid_timer.elapsed, 2), "npz": npz_path,
    }
    common.update_registry(ds_dir / "registry.json", run_dir.name, entry, promote=config.PROMOTE, logger=logger)
    return entry


def run_dataset_seed(dataset: str, seed: int, artifacts_root: str | Path | None = None) -> dict:
    """The one function both run_model.py (in-process, single job) and this
    file's __main__ block (one subprocess per job) call."""
    if dataset not in config.DATASETS:
        raise ValueError(f"unknown dataset {dataset!r}; expected one of {sorted(config.DATASETS)}")
    root = Path(artifacts_root) if artifacts_root else common.seed_artifact_root(seed, base=config.ARTIFACTS_DIR)
    return run_grid(dataset, seed, root)


if __name__ == "__main__":
    import os
    _dataset = os.environ.get("TCAT_DATASET", config.RUN_DATASETS[0])
    _seed = int(os.environ.get("TCAT_SEED", config.RUN_SEEDS[0]))
    run_dataset_seed(_dataset, _seed)
