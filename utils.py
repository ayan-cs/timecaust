import h5py, torch
from datetime import datetime
import numpy as np

from crvae_model.model import cLSTM

def epoch_time(start_time, end_time):
    elapsed_time = end_time - start_time
    elapsed_hrs = int(elapsed_time / 3600)
    elapsed_mins = int((elapsed_time - elapsed_hrs * 3600) / 60)
    elapsed_secs = elapsed_time - (elapsed_mins * 60 + elapsed_hrs * 3600)
    return elapsed_hrs, elapsed_mins, elapsed_secs

def getModelName(dataset, model_type):
    now = str(datetime.now())
    date, time = now.split()[0], now.split()[1]
    date = date.split('-')
    date.reverse()
    date = '-'.join(date)
    time = time.replace(':', '-')[:8]

    model_name = f"{model_type}___{dataset}___{date}_{time}"
    return model_name

def load_data(path):
    with h5py.File(path, 'r') as f:
        X_train_left = f['X_train_left']
        X_train_right = f['X_train_right']
        X_val_left = f['X_val_left']
        X_val_right = f['X_val_right']
        return list(X_train_left), list(X_train_right), list(X_val_left), list(X_val_right)

def load_single_sample(path):
    with h5py.File(path, 'r') as f:
        x_l = list(f['x_l'])
        x_r = list(f['x_r'])
        return np.array(x_l), np.array(x_r)

def load_model(path, metadata, model_type='crvae', dataset_name='henon'):
    if model_type=='crvae':
        hyperparameters = {
            'n_dim': metadata['model']['input_size'],
            'hidden_size': metadata['model']['hidden_size'],
            'num_layers': metadata['model']['num_layers'],
            'dropout': metadata['model']['dropout']
        }
        GC = getCausalMatrix(n_dim=hyperparameters['n_dim'], data=dataset_name)
        model = cLSTM(n_dim=hyperparameters['n_dim'], hidden_size=hyperparameters['hidden_size'], causal_graph=GC).cuda()
        model.load_state_dict(torch.load(path))
    
    else:
        raise ValueError(f"Invalid model_type: {model_type}. Valid choices: ['crvae']")
    
    model.eval()
    for p in model.parameters():
        p.requires_grad_(False)
    # for module in model.modules():
    #     print(module)
    #     if isinstance(module, (torch.nn.LSTM, torch.nn.GRU, torch.nn.RNN)):
    #         module.train()
    return model

def getExperimentName():
    now = str(datetime.now())
    date, time = now.split()[0], now.split()[1]
    date = date.split('-')
    date.reverse()
    date = '-'.join(date)
    time = time.replace(':', '-')[:8]
    model_name = f"attack___{date}_{time}"
    return model_name

def getCausalMatrix(n_dim=None, data='henon'):
    if data=='henon':
        assert n_dim is not None
        GC = np.zeros([n_dim, n_dim])
        for i in range(n_dim):
            GC[i,i] = 1
            if i!=0:
                GC[i,i-1] = 1
        return GC

    if data=='lorenz':
        assert n_dim is not None
        GC = np.zeros((n_dim, n_dim), dtype=int)
        for i in range(n_dim):
            GC[i, i] = 1
            GC[i, (i + 1) % n_dim] = 1
            GC[i, (i - 1) % n_dim] = 1
            GC[i, (i - 2) % n_dim] = 1
        return GC
    
    if data=='ecoli':
        GC = np.array(
            [[1., 1., 0., 0., 0., 0., 0., 0., 0., 0.],
            [0., 1., 1., 0., 1., 0., 0., 0., 0., 1.],
            [0., 0., 1., 0., 0., 0., 0., 0., 0., 0.],
            [0., 1., 0., 1., 0., 0., 0., 0., 0., 0.],
            [0., 0., 0., 0., 1., 0., 0., 0., 0., 0.],
            [0., 1., 0., 0., 0., 1., 0., 0., 0., 0.],
            [0., 1., 0., 0., 0., 0., 1., 0., 0., 0.],
            [0., 1., 0., 0., 0., 0., 0., 1., 0., 0.],
            [0., 1., 0., 0., 0., 0., 0., 0., 1., 0.],
            [0., 0., 0., 0., 0., 0., 0., 0., 0., 1.]]
        )
        return GC
    
    if data=='yeast':
        GC = np.array(
            [[1., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
            [1., 1., 1., 1., 0., 0., 0., 0., 0., 0.],
            [0., 0., 1., 0., 0., 0., 0., 0., 0., 0.],
            [0., 0., 0., 1., 0., 0., 0., 0., 0., 0.],
            [0., 0., 0., 1., 1., 1., 0., 0., 0., 0.],
            [0., 0., 0., 0., 0., 1., 0., 0., 0., 0.],
            [0., 0., 0., 0., 0., 1., 1., 1., 0., 0.],
            [0., 0., 0., 0., 0., 0., 0., 1., 0., 0.],
            [0., 0., 0., 0., 0., 0., 0., 1., 1., 0.],
            [0., 0., 0., 0., 0., 0., 0., 0., 1., 1.]]
        )
        return GC
    
    if data=='synhill':
        GC = np.array(
            [[1., 0., 0., 0., 0., 0., 0., 1., 0., 0.],
            [0., 1., 0., 0., 0., 0., 0., 1., 0., 0.],
            [0., 0., 1., 0., 0., 0., 1., 0., 1., 1.],
            [0., 0., 0., 1., 0., 1., 0., 1., 0., 0.],
            [0., 0., 1., 0., 1., 0., 1., 0., 0., 1.],
            [0., 1., 1., 0., 1., 1., 0., 0., 0., 0.],
            [0., 0., 1., 0., 0., 0., 1., 0., 0., 1.],
            [0., 0., 1., 1., 0., 0., 0., 1., 0., 1.],
            [0., 0., 0., 1., 0., 1., 0., 1., 1., 0.],
            [1., 0., 0., 0., 0., 0., 1., 1., 0., 1.]]
        )
        return GC

def preprocessSyn(X, context, test_size=0.2):

    def createChunks(data, context):
        if context > data.shape[0]:
            context = data.shape[0]
        X_left = []
        X_right = []
        # context_l = int(context/2)
        context_l = context - 1 # For CauFR-TS
        for i in range(len(data) - context + 1):
            X_left.append(data[i : i+context_l].tolist())
            X_right.append(data[i+context_l : i+context].tolist())
        return np.array(X_left), np.array(X_right)

    # if data is not None:
    X_left, X_right = createChunks(X, context)
    num_train = int(X_left.shape[0]*(1-test_size))
    X_train_left, X_train_right, X_test_left, X_test_right = X_left[:num_train, :, :], X_right[:num_train, :, :], X_left[num_train:, : ,:], X_right[num_train:, :, :]
    return X_train_left, X_train_right, X_test_left, X_test_right
    
    # else:
    #     parent = os.path.abspath('')
    #     dataset = os.path.join(parent, 'datasets', f"{config['dataset']}.npy")
    #     context = config['chunksize']
    #     X = np.load(dataset).T

    #     X_left, X_right = createChunks(X, context)
    #     X_train_left, X_train_right, X_test_left, X_test_right = createSplit(X_left, X_right, test_size=config['test_split'], shuffle=True)
    #     return X_train_left, X_train_right, X_test_left, X_test_right