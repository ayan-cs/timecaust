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
        h_0 = torch.zeros(1, X_left.shape[0], self.hidden_size, device='cuda') # Initial Hidden and Cell states, both Zero
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
        X_right = X_right[:, :, np.where(causal_graph!=0)[0]]
        # first = torch.zeros_like(X_right[:, 0:1, :])
        # X_right = torch.cat((first, X_right[:, :-1, :]), dim=1)
        X_right_pred, (hidden_out, cell_out) = self.lstm(X_right, z)
        X_right_pred = self.fc(X_right_pred).squeeze(-1)
        return X_right_pred, (hidden_out, cell_out)

class cLSTM(nn.Module):
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
    
    def forward(self, X_left, future): # X_right=None
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
                if d==0:
                    X_t = op
                else:
                    X_t = torch.cat((X_t, op), dim=-1)
                hidden_curr.append(h_t)
                cell_curr.append(c_t)
            hidden_list = hidden_curr
            cell_list = cell_curr
            X_pred.append(X_t)
            # print(f"X_t shape: {X_t.shape}\tX_pred[-1].unsqueeze: {X_pred[-1].unsqueeze(1).shape}")
        X_pred = torch.stack(X_pred, dim=1)[:, 1:, :]
        # print(f"X_pred shape: {X_pred.shape}")
        return X_pred, d_kl