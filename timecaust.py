import torch, os, json, sys, time, gc
from itertools import product
import numpy as np
from torch.utils.data import DataLoader, TensorDataset

from utils import load_data, load_model, getExperimentName, getCausalMatrix, epoch_time
from helpers import *
from crvae_model.model import cLSTM

def simulate_attack(param_grid, dataset, model, exp_details, log_ptr):
    dataset_name, dataset_artifact, X_train_left, X_train_right, X_val_left, X_val_right = dataset
    model_type, model_artifact, model = model
    exp_name, exp_dir = exp_details
    print(f"Using dataset: {dataset_artifact}", flush=True)
    print(f"Using model: '{model_type}' from artifact: '{model_artifact}'", flush=True)

    n_dim = metadata['model']['input_size']
    GC = getCausalMatrix(n_dim=n_dim, data=dataset_name)

    param_combinations = list(product(
        *[param_grid[k] for k in param_grid.keys() if k not in ['target_dims', 'n_random_trials']]
    ))

    target_dims = param_grid['target_dims']
    n_random_trials = param_grid.get('n_random_trials', [3])[0]
 
    print(f"Experiment: {exp_name}", flush=True)
    print(f"Target dims: {target_dims}", flush=True)
    print(f"Hyperparameter combinations: {len(param_combinations)}", flush=True)
    print(f"Random trials per condition: {n_random_trials}\n", flush=True)

    start_exp = time.time()
    for j in target_dims:
        parent_mask = get_parent_mask(GC, j)
        nonparent_mask = get_nonparent_mask(GC, j)
        n_parents = int(parent_mask.sum())
 
        dim_dir = os.path.join(exp_dir, f'dim_{j}')
        os.mkdir(dim_dir)
        logs_path = os.path.join(dim_dir, 'logs')
        os.mkdir(logs_path)
        metadata_path = os.path.join(dim_dir, 'metadata')
        os.mkdir(metadata_path)
 
        print(f"\n{'='*60}", flush=True)
        print(f"Target dimension: {j}  |  Parents: {np.where(parent_mask)[0].tolist()}  ({n_parents} channels)", flush=True)
        print(f"{'='*60}", flush=True)
 
        combination = 0
        dim_metadata_list = []

        for lam_ntgt, lam_smooth, epsilon, alpha, pgd_steps, batch_size in param_combinations:
            combination += 1

            # To be printed inside Global log
            print(f"\nCombination {combination} ::", flush=True)
            print(f"lam_ntgt: {lam_ntgt}\tlam_smooth: {lam_smooth}\tepsilon: {epsilon}\talpha: {alpha}\tpgd_steps: {pgd_steps}\tbatch_size: {batch_size}", flush=True)

            # Experiment log
            sys.stdout = open(os.path.join(logs_path, f'attack_comb{combination}.log'), 'w')
            print(f"Using dataset: {dataset_artifact}", flush=True)
            print(f"Target dim: {j}  |  Parents: {np.where(parent_mask)[0].tolist()}", flush=True)
            print(f"\nCombination {combination} ::", flush=True)
            print(f"lam_ntgt: {lam_ntgt}\tlam_smooth: {lam_smooth}\tepsilon: {epsilon}\talpha: {alpha}\tpgd_steps: {pgd_steps}\tbatch_size: {batch_size}", flush=True)
 
            # Prepare dataloaders
            X_tl = np.array(X_train_left)
            X_tr = np.array(X_train_right)
            X_vl = np.array(X_val_left)
            X_vr = np.array(X_val_right)
            train_dl = DataLoader(
                TensorDataset(torch.FloatTensor(X_tl), torch.FloatTensor(X_tr)),
                batch_size=batch_size
            )
            val_dl = DataLoader(
                TensorDataset(torch.FloatTensor(X_vl), torch.FloatTensor(X_vr)),
                batch_size=batch_size
            )
            del X_tl, X_tr, X_vl, X_vr
            gc.collect()
 
            hparams_common = {
                'epsilon': epsilon, 'alpha': alpha, 'pgd_steps': pgd_steps,
                'lam_ntgt': lam_ntgt, 'lam_smooth': lam_smooth
            }
 
            # ============== Condition 1: Parent-only (TimeCauST) ==============
            print(f"\n--- Condition: PARENT-ONLY ---", flush=True)
            parent_results_tr = run_condition(
                model, train_dl, X_train_right, j, parent_mask,
                'parent_only', **hparams_common
            )
            parent_results_val = run_condition(
                model, val_dl, X_val_right, j, parent_mask,
                'parent_only', **hparams_common
            )

            save_attack_artifacts(results=parent_results_tr, metadata_dir=metadata_path, prefix=f'comb{combination}_train_parent')
            save_attack_artifacts(results=parent_results_val, metadata_dir=metadata_path, prefix=f'comb{combination}_val_parent')

            # CHANGED: updated key from 'adv_mse_tgt' to 'mse_tgt', added all metrics
            print(f"TRN:: Parent-only  MSEtgt: {parent_results_tr['mse_tgt']:.6f}  CSS: {parent_results_tr['css']:.4f}  CGA: {parent_results_tr['cga']:.4f}  PE: {parent_results_tr['pe']:.4f}  PRS: {parent_results_tr['prs']:.6f}", flush=True)
            print(f"VAL:: Parent-only  MSEtgt: {parent_results_val['mse_tgt']:.6f}  CSS: {parent_results_val['css']:.4f}  CGA: {parent_results_val['cga']:.4f}  PE: {parent_results_val['pe']:.4f}  PRS: {parent_results_val['prs']:.6f}", flush=True)
 
            # ============== Condition 2: Non-parent-only ==============
            print(f"\n--- Condition: NON-PARENT-ONLY ---", flush=True)
            nonparent_results_tr = run_condition(
                model, train_dl, X_train_right, j, nonparent_mask,
                'nonparent_only', **hparams_common
            )
            nonparent_results_val = run_condition(
                model, val_dl, X_val_right, j, nonparent_mask,
                'nonparent_only', **hparams_common
            )

            save_attack_artifacts(results=nonparent_results_tr, metadata_dir=metadata_path, prefix=f'comb{combination}_train_nonparent')
            save_attack_artifacts(results=nonparent_results_val, metadata_dir=metadata_path, prefix=f'comb{combination}_val_nonparent')

            # CHANGED: updated key and added all metrics
            print(f"TRN:: Non-parent   MSEtgt: {nonparent_results_tr['mse_tgt']:.6f}  CSS: {nonparent_results_tr['css']:.4f}  CGA: {nonparent_results_tr['cga']:.4f}  PE: {nonparent_results_tr['pe']:.4f}  PRS: {nonparent_results_tr['prs']:.6f}", flush=True)
            print(f"VAL:: Non-parent   MSEtgt: {nonparent_results_val['mse_tgt']:.6f}  CSS: {nonparent_results_val['css']:.4f}  CGA: {nonparent_results_val['cga']:.4f}  PE: {nonparent_results_val['pe']:.4f}  PRS: {nonparent_results_val['prs']:.6f}", flush=True)
 
            # ============== Condition 3: Random (multiple trials) ==============
            random_trial_results_tr = []
            random_trial_results_val = []
            for trial in range(n_random_trials):
                random_mask = get_random_mask(n_dim, n_parents, exclude_dim=j, seed=trial*100+j)
                print(f"\n--- Condition: RANDOM (trial {trial+1}/{n_random_trials}, channels={np.where(random_mask)[0].tolist()}) ---", flush=True)
                rand_res_tr = run_condition(
                    model, train_dl, X_train_right, j, random_mask,
                    f'random_trial_{trial+1}', **hparams_common
                )
                rand_res_val = run_condition(
                    model, val_dl, X_val_right, j, random_mask,
                    f'random_trial_{trial+1}', **hparams_common
                )

                save_attack_artifacts(results=rand_res_tr, metadata_dir=metadata_path, prefix=f'comb{combination}_train_rand{trial+1}')
                save_attack_artifacts(results=rand_res_val, metadata_dir=metadata_path, prefix=f'comb{combination}_val_rand{trial+1}')

                # CHANGED: updated key and added all metrics
                print(f"TRN:: Random trial {trial+1}  MSEtgt: {rand_res_tr['mse_tgt']:.6f}  CSS: {rand_res_tr['css']:.4f}  CGA: {rand_res_tr['cga']:.4f}  PE: {rand_res_tr['pe']:.4f}  PRS: {rand_res_tr['prs']:.6f}", flush=True)
                print(f"VAL:: Random trial {trial+1}  MSEtgt: {rand_res_val['mse_tgt']:.6f}  CSS: {rand_res_val['css']:.4f}  CGA: {rand_res_val['cga']:.4f}  PE: {rand_res_val['pe']:.4f}  PRS: {rand_res_val['prs']:.6f}", flush=True)
                random_trial_results_tr.append(rand_res_tr)
                random_trial_results_val.append(rand_res_val)
 
            def aggregate_random_trials(trial_list):
                agg = {}
                for key in ['css', 'cga', 'pe', 'mse_tgt', 'mae_tgt', 'prs', 'mse_nt', 'mae_nt']:
                    vals = [r[key] for r in trial_list]
                    agg[f'{key}_mean'] = float(np.mean(vals))
                    agg[f'{key}_std'] = float(np.std(vals))
                return agg

            random_agg_tr = aggregate_random_trials(random_trial_results_tr)
            random_agg_val = aggregate_random_trials(random_trial_results_val)
 
            # ============== Compile metadata ==============
            comb_metadata = {
                'combination': combination,
                'dataset': dataset_artifact,
                'model_artifact': model_artifact,
                'target_dim': j,
                'parent_channels': np.where(parent_mask)[0].tolist(),
                'n_dim': n_dim,
                'hyperparameters': {
                    'lam_ntgt': lam_ntgt,
                    'lam_smooth': lam_smooth,
                    'epsilon': epsilon,
                    'alpha': alpha,
                    'pgd_steps': pgd_steps,
                    'batch_size': batch_size,
                    'n_random_trials': n_random_trials
                },
                'results_train_set': {
                    'parent_only': parent_results_tr,
                    'nonparent_only': nonparent_results_tr,
                    'random_trials': random_trial_results_tr,
                    'random_aggregated': random_agg_tr
                },
                'results_val_set': {
                    'parent_only': parent_results_val,
                    'nonparent_only': nonparent_results_val,
                    'random_trials': random_trial_results_val,
                    'random_aggregated': random_agg_val
                },
                'ordering_train_set': {
                    'css': {
                        'parent': parent_results_tr['css'],
                        'nonparent': nonparent_results_tr['css'],
                        'random_mean': random_agg_tr['css_mean'],
                        'expected_P_gt_R_gt_NP': parent_results_tr['css'] > random_agg_tr['css_mean'] > nonparent_results_tr['css']
                    },
                    'pe': {
                        'parent': parent_results_tr['pe'],
                        'nonparent': nonparent_results_tr['pe'],
                        'random_mean': random_agg_tr['pe_mean'],
                        'expected_P_gt_R_gt_NP': parent_results_tr['pe'] > random_agg_tr['pe_mean'] > nonparent_results_tr['pe']
                    },
                    'cga': {
                        'parent': parent_results_tr['cga'],
                        'nonparent': nonparent_results_tr['cga'],
                        'random_mean': random_agg_tr['cga_mean']
                    },
                    'prs': {
                        'parent': parent_results_tr['prs'],
                        'nonparent': nonparent_results_tr['prs'],
                        'random_mean': random_agg_tr['prs_mean']
                    }
                },
                'ordering_val_set': {
                    'css': {
                        'parent': parent_results_val['css'],
                        'nonparent': nonparent_results_val['css'],
                        'random_mean': random_agg_val['css_mean'],
                        'expected_P_gt_R_gt_NP': parent_results_val['css'] > random_agg_val['css_mean'] > nonparent_results_val['css']
                    },
                    'pe': {
                        'parent': parent_results_val['pe'],
                        'nonparent': nonparent_results_val['pe'],
                        'random_mean': random_agg_val['pe_mean'],
                        'expected_P_gt_R_gt_NP': parent_results_val['pe'] > random_agg_val['pe_mean'] > nonparent_results_val['pe']
                    },
                    'cga': {
                        'parent': parent_results_val['cga'],
                        'nonparent': nonparent_results_val['cga'],
                        'random_mean': random_agg_val['cga_mean']
                    },
                    'prs': {
                        'parent': parent_results_val['prs'],
                        'nonparent': nonparent_results_val['prs'],
                        'random_mean': random_agg_val['prs_mean']
                    }
                }
            }
 
            print(f"\n{'#'*80}", flush=True)
            print(f"SUMMARY on TRAIN SET :: Combination {combination}, dim {j}", flush=True)
            print(f"  Parent     | CSS: {parent_results_tr['css']:.4f}  PE: {parent_results_tr['pe']:.4f}  CGA: {parent_results_tr['cga']:.4f}  PRS: {parent_results_tr['prs']:.6f}  MSEtgt: {parent_results_tr['mse_tgt']:.6f}  MSEnt: {parent_results_tr['mse_nt']:.6f}", flush=True)
            print(f"  Non-parent | CSS: {nonparent_results_tr['css']:.4f}  PE: {nonparent_results_tr['pe']:.4f}  CGA: {nonparent_results_tr['cga']:.4f}  PRS: {nonparent_results_tr['prs']:.6f}  MSEtgt: {nonparent_results_tr['mse_tgt']:.6f}  MSEnt: {nonparent_results_tr['mse_nt']:.6f}", flush=True)
            print(f"  Random     | CSS: {random_agg_tr['css_mean']:.4f}±{random_agg_tr['css_std']:.4f}  PE: {random_agg_tr['pe_mean']:.4f}±{random_agg_tr['pe_std']:.4f}", flush=True)
            print(f"  CSS ordering (P>R>NP): {comb_metadata['ordering_train_set']['css']['expected_P_gt_R_gt_NP']}", flush=True)
            print(f"  PE  ordering (P>R>NP): {comb_metadata['ordering_train_set']['pe']['expected_P_gt_R_gt_NP']}", flush=True)
            print(f"\n{'#'*80}", flush=True)
            print(f"SUMMARY on VAL SET :: Combination {combination}, dim {j}", flush=True)
            print(f"  Parent     | CSS: {parent_results_val['css']:.4f}  PE: {parent_results_val['pe']:.4f}  CGA: {parent_results_val['cga']:.4f}  PRS: {parent_results_val['prs']:.6f}  MSEtgt: {parent_results_val['mse_tgt']:.6f}  MSEnt: {parent_results_val['mse_nt']:.6f}", flush=True)
            print(f"  Non-parent | CSS: {nonparent_results_val['css']:.4f}  PE: {nonparent_results_val['pe']:.4f}  CGA: {nonparent_results_val['cga']:.4f}  PRS: {nonparent_results_val['prs']:.6f}  MSEtgt: {nonparent_results_val['mse_tgt']:.6f}  MSEnt: {nonparent_results_val['mse_nt']:.6f}", flush=True)
            print(f"  Random     | CSS: {random_agg_val['css_mean']:.4f}±{random_agg_val['css_std']:.4f}  PE: {random_agg_val['pe_mean']:.4f}±{random_agg_val['pe_std']:.4f}", flush=True)
            print(f"  CSS ordering (P>R>NP): {comb_metadata['ordering_val_set']['css']['expected_P_gt_R_gt_NP']}", flush=True)
            print(f"  PE  ordering (P>R>NP): {comb_metadata['ordering_val_set']['pe']['expected_P_gt_R_gt_NP']}", flush=True)
            print(f"{'#'*80}\n", flush=True)
 
            sys.stdout = log_ptr
            dim_metadata_list.append(comb_metadata)

            print(f"\n{'*'*80}", flush=True)
            print(f"SUMMARY on TRAIN SET :: Combination {combination}, dim {j}", flush=True)
            print(f"  Parent     | CSS: {parent_results_tr['css']:.4f}  PE: {parent_results_tr['pe']:.4f}  CGA: {parent_results_tr['cga']:.4f}  PRS: {parent_results_tr['prs']:.6f}", flush=True)
            print(f"  Non-parent | CSS: {nonparent_results_tr['css']:.4f}  PE: {nonparent_results_tr['pe']:.4f}  CGA: {nonparent_results_tr['cga']:.4f}  PRS: {nonparent_results_tr['prs']:.6f}", flush=True)
            print(f"  Random     | CSS: {random_agg_tr['css_mean']:.4f}±{random_agg_tr['css_std']:.4f}  PE: {random_agg_tr['pe_mean']:.4f}±{random_agg_tr['pe_std']:.4f}", flush=True)
            print(f"  CSS ordering (P>R>NP): {comb_metadata['ordering_train_set']['css']['expected_P_gt_R_gt_NP']}", flush=True)
            print(f"\n{'*'*80}", flush=True)
            print(f"SUMMARY on VAL SET :: Combination {combination}, dim {j}", flush=True)
            print(f"  Parent     | CSS: {parent_results_val['css']:.4f}  PE: {parent_results_val['pe']:.4f}  CGA: {parent_results_val['cga']:.4f}  PRS: {parent_results_val['prs']:.6f}", flush=True)
            print(f"  Non-parent | CSS: {nonparent_results_val['css']:.4f}  PE: {nonparent_results_val['pe']:.4f}  CGA: {nonparent_results_val['cga']:.4f}  PRS: {nonparent_results_val['prs']:.6f}", flush=True)
            print(f"  Random     | CSS: {random_agg_val['css_mean']:.4f}±{random_agg_val['css_std']:.4f}  PE: {random_agg_val['pe_mean']:.4f}±{random_agg_val['pe_std']:.4f}", flush=True)
            print(f"  CSS ordering (P>R>NP): {comb_metadata['ordering_val_set']['css']['expected_P_gt_R_gt_NP']}", flush=True)
            print(f"{'*'*80}\n", flush=True)
 
            with open(os.path.join(metadata_path, f'comb{combination}_metadata.json'), 'w') as f:
                json.dump(comb_metadata, f, indent=4)
 
            torch.cuda.empty_cache()
 
        # Dimension-level summary
        with open(os.path.join(metadata_path, 'all_metadata.jsonl'), 'w') as f:
            for obj in dim_metadata_list:
                f.write(json.dumps(obj) + '\n')
 
    end_exp = time.time()
    h, m, s = epoch_time(start_exp, end_exp)
    print(f"\nTotal experiment time : {h}hrs. {m}mins. {s:.2f}sec.", flush=True)
    print(f"{'%'*50}\n")
 
    exp_summary = {
        'experiment': exp_name,
        'dataset': dataset_artifact,
        'model_artifact': model_artifact,
        'target_dims': target_dims,
        'n_combinations': len(param_combinations),
        'total_time': {'hr': h, 'mins': m, 'sec': s}
    }
    with open(os.path.join(exp_dir, 'experiment_summary.json'), 'w') as f:
        json.dump(exp_summary, f, indent=4)
 
    return exp_summary

if __name__ == "__main__":
    torch.manual_seed(42)
    np.random.seed(42)

    BASE_DIR = os.path.abspath('')

    ####################################################################
    ### Manually change this place for other datasets
    dataset_name = 'henon'
    dataset_artifact = 'henon_10001_10_50.h5'
    model_artifact = 'crvae___henon_10001_10_50___17-04-2026_17-09-26'
    COMBINATION = 1
    context = 50
    ####################################################################

    datapath = os.path.join(BASE_DIR, 'data', dataset_artifact)
    checkpoint_path = os.path.join(BASE_DIR, f'artifacts_{dataset_name}', model_artifact, 'checkpoints', f'checkpoint_comb{COMBINATION}.pt')
    metadata_path = os.path.join(BASE_DIR, f'artifacts_{dataset_name}', model_artifact, 'metadata', f'metadata_comb{COMBINATION}.json')

    exp_name = getExperimentName()
    exp_dir = os.path.join(BASE_DIR, f'artifacts_{dataset_name}', model_artifact, exp_name)
    os.mkdir(exp_dir)

    print(f"Loading dataset: '{dataset_name}' ...", flush=True)
    X_train_left, X_train_right, X_val_left, X_val_right = load_data(datapath)
    print(f"Dataset loaded.", flush=True)

    print(f"Loading metadata first: 'metadata_comb{COMBINATION}' from artifact: '{model_artifact}' ...", flush=True)
    with open(metadata_path, 'r') as f:
        metadata = json.load(f)

    print(f"Dataset loaded.", flush=True)

    print(f"Loading trained model: 'checkpoint_comb{COMBINATION}' from artifact: '{model_artifact}' ...", flush=True)
    model_type = model_artifact.split('_')[0]
    model = load_model(path=checkpoint_path, metadata=metadata, model_type=model_type, dataset_name=dataset_name)
    print(f"Model loaded.", flush=True)

    global_log_f = open(os.path.join(exp_dir, 'global_logs.log'), 'w')
    sys.stdout = global_log_f
    print(f"Dataset: '{dataset_name}' Loaded.", flush=True)
    print(f"Loaded metadata: 'metadata_comb{COMBINATION}' from artifact: '{model_artifact}.'", flush=True)
    print(f"Loaded trained model: 'checkpoint_comb{COMBINATION}' from artifact: '{model_artifact}'.\n", flush=True)
    print(f"{'#'*80}\n{'#'*80}\n{'#'*80}")

    param_grid = {
        'target_dims': list(range(metadata['model']['input_size'])),
        'lam_ntgt': [0.5],
        'lam_smooth': [0.1, 0.5],
        'epsilon': [0.1, 0.5],
        'alpha': [0.05],
        'pgd_steps': [100],
        'batch_size': [4096],
        'n_random_trials': [3]
    }

    _ = simulate_attack(
        param_grid=param_grid,
        dataset=(dataset_name, dataset_artifact.split('.')[0], X_train_left, X_train_right, X_val_left, X_val_right),
        model=(model_type, model_artifact, model),
        exp_details=(exp_name, exp_dir),
        log_ptr=global_log_f
    )

    sys.stdout = sys.__stdout__