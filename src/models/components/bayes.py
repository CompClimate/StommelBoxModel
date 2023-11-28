import torch.nn as nn
import models.components.bayes_layer as bl


class BayesMLPModel(nn.Module):
    """Implements a simple Bayesian neural network."""

    def __init__(self, input_dim, hidden_dim, n_hidden, num_MC=1):
        super().__init__()
        self.mc_exp = bl.MC_ExpansionLayer(num_MC=num_MC, input_dim=2)
        self.input_block = nn.Sequential(
            bl.BayesLinear(input_dim, hidden_dim),
            nn.ReLU(),
        )
        self.hidden_layers = [
            nn.Sequential(
                bl.BayesLinear(hidden_dim, hidden_dim),
                nn.ReLU(),
            )
        ] * n_hidden
        self.hidden_layers = nn.Sequential(*self.hidden_layers)
        self.output_block = bl.BayesLinear(hidden_dim, 1)
        self.explain_mode = False

    def forward(self, x):
        x = self.mc_exp(x)
        x = self.input_block(x)
        x = self.hidden_layers(x)
        x = self.output_block(x)
        mu, std = x.mean(dim=0).squeeze(), x.std(dim=0).squeeze()
        if self.explain_mode:
            return mu.unsqueeze(-1)
        else:
            return mu, std
