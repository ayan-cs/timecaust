import torch, os, json, sys, time, gc
import torch.nn.functional as f
from torch.utils.data import DataLoader, TensorDataset
from torch.optim import AdamW
from torch.optim.lr_scheduler import ReduceLROnPlateau
import numpy as np
from itertools import product

# Considering BASE folder as 'timecaust' with empty __init__.py
# Code execution has to be done from 'timecaust' folder for relative imports to work
from utils import load_data, getModelName, epoch_time, getCausalMatrix
from crvae_model.model import cLSTM

def train_epoch(model, train_dl, val_dl, optimizer, beta_kl):
    total_train_loss = 0
    total_train_mse = 0
    total_train_kl = 0
    model.train()
    acc_steps = 1
    for idx, (X_l, X_r) in enumerate(train_dl):
        X_l = X_l.cuda()
        X_r = X_r.cuda()
        optimizer.zero_grad()
        pred, train_kl = model(X_l, future=X_r.shape[1])
        train_mse = f.mse_loss(pred, X_r)
        loss = train_mse + beta_kl * train_kl
        loss /= acc_steps
        loss.backward()
        optimizer.step()
        total_train_loss += loss.item() * X_l.size(0)
        total_train_mse += train_mse.item() * X_l.size(0)
        total_train_kl += train_kl.item() * X_l.size(0)

    train_loss = total_train_loss / len(train_dl.dataset)
    train_mse = total_train_loss / len(train_dl.dataset)
    train_kl = total_train_kl / len(train_dl.dataset)
    torch.cuda.empty_cache()

    model.eval()
    total_val_loss = 0
    total_val_mse = 0
    total_val_kl = 0
    with torch.no_grad():
        for X_l, X_r in val_dl:
            X_l, X_r = X_l.cuda(), X_r.cuda()
            pred, val_kl = model(X_l, future=X_r.shape[1])
            val_mse = f.mse_loss(pred, X_r)
            loss = val_mse + beta_kl * val_kl
            total_val_loss += loss.item() * X_l.size(0)
            total_val_mse += val_mse.item() * X_l.size(0)
            total_val_kl += val_kl.item() * X_l.size(0)
    
    val_loss = total_val_loss / len(val_dl.dataset)
    val_mse = total_val_mse / len(val_dl.dataset)
    val_kl = total_val_kl / len(val_dl.dataset)
    torch.cuda.empty_cache()
    
    return train_loss, val_loss, train_mse, train_kl, val_mse, val_kl

def trainer(dataset_name, param_grid, X_train_left1, X_train_right1, X_val_left1, X_val_right1, patience, step_size):
    BASE_DIR = os.path.abspath('')

    dataset_name, data_artifact = dataset_name
    print(f"Using dataset: {data_artifact}", flush=True)
    if not os.path.exists(os.path.join(BASE_DIR, f'artifacts_{dataset_name}')):
        os.mkdir(os.path.join(BASE_DIR, f'artifacts_{dataset_name}'))
    model_name = getModelName(dataset=data_artifact, model_type='crvae')

    artifact_path = os.path.join(BASE_DIR, f'artifacts_{dataset_name}', model_name)
    os.mkdir(artifact_path)
    checkpoints_path = os.path.join(artifact_path, 'checkpoints')
    os.mkdir(checkpoints_path)
    metadata_path = os.path.join(artifact_path, 'metadata')
    os.mkdir(metadata_path)
    logs_path = os.path.join(artifact_path, 'logs')
    os.mkdir(logs_path)

    input_size = len(X_train_left1[0][0]) # X_train_left.shape[-1]
    seq_len = len(X_train_left1[0]) + 1 # X_train_left.shape[-2] + 1
    GC = getCausalMatrix(n_dim=input_size, data=dataset_name)

    param_combinations = list(product(
    param_grid['lr'],
    param_grid['batch_size'],
    param_grid['hidden_size'],
    param_grid['num_layers'],
    param_grid['dropout'],
    param_grid['beta_kl']
    ))

    # best_val_loss = np.inf # For global best model (runtime error)
    epochs = 5000
    combination = 0
    metadata_list = []
    start_gs = time.time()
    for lr, batch_size, hidden_size, num_layers, dropout, beta_kl in param_combinations:
        combination += 1
        sys.stdout = open(os.path.join(logs_path, f'train_comb{combination}.log'), 'w')
        print(f"Using dataset: {data_artifact}", flush=True)
        print(f"\nTraining with combination {combination} ::\nInitial LR: {lr}\tBatch size: {batch_size}\tnum_layers: {num_layers}\tDropout: {dropout}\tbeta_kl: {beta_kl}", flush=True)
        X_train_left = np.array(X_train_left1)
        X_train_right = np.array(X_train_right1)
        train_dl = DataLoader(TensorDataset(torch.FloatTensor(X_train_left), torch.FloatTensor(X_train_right)), batch_size=batch_size, shuffle=False)
        del X_train_left, X_train_right
        gc.collect()
        
        X_val_left = np.array(X_val_left1)
        X_val_right = np.array(X_val_right1)
        val_dl = DataLoader(TensorDataset(torch.FloatTensor(X_val_left), torch.FloatTensor(X_val_right)), batch_size=batch_size)
        del X_val_left, X_val_right
        gc.collect()

        model = cLSTM(n_dim=input_size, hidden_size=hidden_size, causal_graph=GC).cuda()
        optimizer = AdamW(model.parameters(), lr=lr)
        scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=step_size)

        train_loss_list = {
            'loss':[], 'mse':[], 'kl':[]
        }
        val_loss_list = {
            'loss':[], 'mse':[], 'kl':[]
        }

        best_epoch = 0
        best_val_loss = np.inf # For best model in each combo (runs perfectly)
        step_counter = 0
        start = time.time()
        for epoch in range(epochs):
            print(f"\nEpoch {epoch+1}\tLearning rate : {scheduler.get_last_lr()}\n", flush=True)
            start_ep = time.time()
            train_loss, val_loss, train_mse, train_kl, val_mse, val_kl = train_epoch(model, train_dl, val_dl, optimizer, beta_kl)
            end_ep = time.time()
            train_loss_list['loss'].append(train_loss)
            val_loss_list['loss'].append(val_loss)
            train_loss_list['mse'].append(train_mse)
            train_loss_list['kl'].append(train_kl)
            val_loss_list['mse'].append(val_mse)
            val_loss_list['kl'].append(val_kl)
        
            print(f"Train loss: {train_loss:10.6f}\tTrain MSE: {train_mse:10.6f}\tTrain KL: {train_kl:10.6f}", flush=True)
            print(f"Validation loss: {val_loss:10.6f}\tVal MSE: {val_mse:10.6f}\tVal KL: {val_kl:10.6f}", flush=True)

            _, mn, sc = epoch_time(start_ep, end_ep)
            print(f"Epoch execution time : {mn}min. {sc:.6f}sec.", flush=True)

            if val_loss < best_val_loss :
                best_val_loss = val_loss
                step_counter = 0
                best_epoch = epoch+1
                metadata = {
                    'combination' : combination,
                    'dataset' : data_artifact,
                    'n_dim' : input_size,
                    'seq_len' : seq_len,
                    'artifact' : model_name,
                    'model' : {
                        'batch_size' : batch_size,
                        'input_size' : input_size,
                        'hidden_size' : hidden_size,
                        'num_layers' : num_layers,
                        'dropout' : dropout
                    },
                    'max_epochs' : epoch,
                    'initial_lr' : lr,
                    'earlystopper_patience' : patience,
                    'lr_step' : step_size,
                    'beta_kl' : beta_kl
                }
                torch.save(model.state_dict(), os.path.join(checkpoints_path, f'checkpoint_comb{combination}.pt'))
                print(f"Model recorded with Val loss : {val_loss}", flush=True)
                best_epoch = epoch
            else:
                step_counter += 1
                # step_counter = 0 # For runtime error (no improvement at all)
            
            scheduler.step(val_loss)
            if step_counter >= patience:
                print(f"Model not improving. Moving on to next combination ...", flush=True)
                break
            torch.cuda.empty_cache()
        
        end = time.time()
        h, m, s = epoch_time(start, end)
        metadata['final_epoch'] = epoch
        metadata['optimal_epoch'] = best_epoch
        metadata['best_val_loss'] = best_val_loss
        metadata['training_time'] = {'hr' : h, 'mins' : m, 'sec' : s}
        metadata['avg_epoch_sec'] = (end - start)/(epoch+1)
        metadata['train_loss_list'] = train_loss_list
        metadata['val_loss_list'] = val_loss_list
        print(f"Total training time : {h}hrs. {m}mins. {s}sec.", flush=True)
        print("\n"+"#"*100+"\n"+"#"*100+"\n"+"#"*100+"\n", flush=True)
        sys.stdout = sys.__stdout__
        torch.cuda.empty_cache()
        metadata_list.append(metadata)
        with open(os.path.join(metadata_path, f'metadata_comb{combination}.json'), 'w') as f:
            json.dump(metadata, f, indent=4)
    
    end_gs = time.time()
    h, m, s = epoch_time(start_gs, end_gs)
    print(f"Total Grid Search training time : {h}hrs. {m}mins. {s}sec.", flush=True)
    metadata_list.append({'grid_search_time' : {'hr' : h, 'mins' : m, 'sec' : s}})
    # sys.stdout = sys.__stdout__
    return metadata_list


if __name__=="__main__":
    torch.manual_seed(42)
    np.random.seed(42)

    BASE_DIR = os.path.abspath('')

    ####################################################################
    ### Manually change this place for other datasets
    dataset_name = 'henon'
    dataset_artifact = 'henon_10001_10_50.h5'
    context = 50
    datapath = os.path.join(BASE_DIR, 'data', dataset_artifact)
    ####################################################################

    print(f"Loading dataset : {dataset_name} ...", flush=True)
    X_train_left, X_train_right, X_val_left, X_val_right = load_data(datapath)
    print(f"Dataset loaded.", flush=True)

    # param_grid = {
    # "lr" : [0.001, 0.005],
    # "batch_size" : [256, 512, 1024],
    # "hidden_size" : [64, 128, 256],
    # "num_layers" : [1],
    # "dropout" : [0],
    # "beta_kl": [0.005, 0.01, 0.05, 0.1]
    # }

    # Optimal Config after Grid Search
    param_grid = {
    "lr" : [0.005],
    "batch_size" : [512],
    "hidden_size" : [128],
    "num_layers" : [1],
    "dropout" : [0],
    "beta_kl": [0.05]
    }

    metadata_list = trainer(
        dataset_name=(dataset_name, dataset_artifact.split('.')[0]),
        param_grid=param_grid,
        X_train_left1=X_train_left, X_train_right1=X_train_right, X_val_left1=X_val_left, X_val_right1=X_val_right,
        patience=100, step_size=20
    )

    with open(os.path.join(BASE_DIR, f'artifacts_{dataset_name}', metadata_list[0]['artifact'], 'train_metadata_all.jsonl'), 'w') as f:
        for obj in metadata_list:
            f.write(json.dumps(obj)+'\n')