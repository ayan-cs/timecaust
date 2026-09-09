"""
crvae_model/model.py
The surrogate forecaster attacked by TimeCAT: a causally-masked VAE-LSTM
(cLSTM). One shared LSTMEncoder turns the input window into a Gaussian
latent (z_h, z_c) for the whole system; one LSTMDecoder per target channel
then forecasts that channel from the latent plus only its *parents'* values
(per the dataset's ground-truth causal graph), so an attacker who wants to
move channel j's forecast is, by the model's own construction, forced to
perturb j's causal parents to have any legitimate structural effect.

Ported from crvae_model/model.py (TimeCAT_new) unchanged in math -- same
forward pass, same losses, same architecture -- with one real bug fixed:

- LSTMEncoder.forward hardcoded `device='cuda'` when building its initial
  hidden/cell state, so the model could only ever run on a GPU machine (a
  CPU forward pass would immediately fail on a device mismatch). It now uses
  the input tensor's own device, matching utils.common.resolve_device's
  design (see that module's docstring on the same class of bug in the
  sibling codebase_caufrts project). On a CUDA machine this resolves to
  exactly the same device as before, so it changes nothing there.

One thing this file does *not* fix, on purpose: cLSTM's constructor only
ever took `n_dim`, `hidden_size`, and `causal_graph` -- `num_layers` and
`dropout` were accepted by the old train_<dataset>.py scripts' hyperparameter
grids and written into metadata.json, but LSTMEncoder/LSTMDecoder never
actually received them (both hardcode a single-layer nn.LSTM with no dropout
argument). Sweeping those two keys in the original grids was therefore a
no-op -- every "different num_layers/dropout" combination trained the exact
same architecture. Left as-is here to keep this port bit-identical to the
original model; config.py's CRVAE_PARAM_GRID keeps both as fixed singletons
([1] and [0]) rather than pretending they are still swept.
"""

import torch, torch.nn as nn
import numpy as np


class LSTMEncoder(nn.Module):
    def __init__(self, n_dim, hidden_size):
        super(LSTMEncoder, self).__init__()
        self.n_dim = n_dim
        self.hidden_size = hidden_size
        self.lstm = nn.LSTM(n_dim, hidden_size, batch_first=True)
        self.lstm.flatten_parameters()

        self.fc_mu_h = nn.Linear(hidden_size, hidden_size)
        self.fc_mu_c = nn.Linear(hidden_size, hidden_size)
        self.fc_logvar_h = nn.Linear(hidden_size, hidden_size)
        self.fc_logvar_c = nn.Linear(hidden_size, hidden_size)

    def forward(self, X_left):
        # Initial hidden/cell state, both zero, on the same device as the
        # input (was hardcoded 'cuda' -- see this file's module docstring).
        h_0 = torch.zeros(1, X_left.shape[0], self.hidden_size, device=X_left.device)
        _, (hidden_out, cell_out) = self.lstm(X_left, (h_0, h_0))
        hidden_out = hidden_out[-1].unsqueeze(0)
        cell_out = cell_out[-1].unsqueeze(0)

        mu_h = self.fc_mu_h(hidden_out)
        mu_c = self.fc_mu_c(cell_out)
        logvar_h = self.fc_logvar_h(hidden_out)
        logvar_c = self.fc_logvar_c(cell_out)
        std_h = torch.exp(0.5 * logvar_h)
        std_c = torch.exp(0.5 * logvar_c)
        eps_h = torch.randn_like(std_h)
        eps_c = torch.randn_like(std_c)

        z_h = mu_h + std_h * eps_h
        z_c = mu_c + std_c * eps_c

        return z_h, z_c, mu_h, logvar_h, mu_c, logvar_c


class LSTMDecoder(nn.Module):
    def __init__(self, n_dim, hidden_size):
        super(LSTMDecoder, self).__init__()
        self.n_dim = n_dim
        self.hidden_size = hidden_size
        self.lstm = nn.LSTM(n_dim, hidden_size, batch_first=True)
        self.fc = nn.Linear(hidden_size, 1)
        self.lstm.flatten_parameters()

    def forward(self, X_right, z, causal_graph):
        X_right = X_right[:, :, np.where(causal_graph != 0)[0]]
        X_right_pred, (hidden_out, cell_out) = self.lstm(X_right, z)
        X_right_pred = self.fc(X_right_pred).squeeze(-1)
        return X_right_pred, (hidden_out, cell_out)


class cLSTM(nn.Module):
    """One shared LSTMEncoder over the full (isolated-channel-free) input
    window, one LSTMDecoder per target channel reading off the shared latent
    plus only that channel's parents (causal_graph[d]). A decoder's input
    width is therefore int(causal_graph[d].sum()) -- the architecture itself
    is shaped by the causal graph, which is why `causal_graph` is part of the
    model's own hyperparameters (see utils.data_utils.load_model) rather
    than something a caller supplies separately at inference time."""

    def __init__(self, n_dim, hidden_size, causal_graph):
        super(cLSTM, self).__init__()
        self.n_dim = n_dim
        self.hidden_size = hidden_size
        self.causal_graph = causal_graph
        self.encoder = LSTMEncoder(n_dim, hidden_size)
        self.decoder = nn.ModuleList([LSTMDecoder(int(causal_graph[d].sum()), hidden_size) for d in range(n_dim)])

    # Dual KL-Divergence for Hidden state and Cell state
    def __kl_loss(self, mu_h, logvar_h, mu_c, logvar_c):
        kl_h = -0.5 * torch.sum(1 + logvar_h - mu_h.pow(2) - logvar_h.exp())
        kl_c = -0.5 * torch.sum(1 + logvar_c - mu_c.pow(2) - logvar_c.exp())
        return kl_h + kl_c

    def forward(self, X_left, future):  # X_right=None
        z_h, z_c, mu_h, logvar_h, mu_c, logvar_c = self.encoder(X_left[:, :-1, :])
        d_kl = self.__kl_loss(mu_h, logvar_h, mu_c, logvar_c)
        hidden_list = [z_h for _ in range(self.n_dim)]
        cell_list = [z_c for _ in range(self.n_dim)]
        X_pred = [X_left[:, -1, :]]
        for _ in range(future):
            hidden_curr = []
            cell_curr = []
            for d, d_head in enumerate(self.decoder):
                op, (h_t, c_t) = d_head(X_pred[-1].unsqueeze(1), z=(hidden_list[d], cell_list[d]), causal_graph=self.causal_graph[d])
                if d == 0:
                    X_t = op
                else:
                    X_t = torch.cat((X_t, op), dim=-1)
                hidden_curr.append(h_t)
                cell_curr.append(c_t)
            hidden_list = hidden_curr
            cell_list = cell_curr
            X_pred.append(X_t)
        X_pred = torch.stack(X_pred, dim=1)[:, 1:, :]
        return X_pred, d_kl
