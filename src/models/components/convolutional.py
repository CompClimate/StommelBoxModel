import torch.nn as nn
from captum.attr._utils.lrp_rules import EpsilonRule


class ConvModel(nn.Module):
    def __init__(self, input_dim, hidden_dim, n_hidden):
        super().__init__()
        self.explain_mode = False
        self.input_block = nn.Conv1d(1, 16, 3)
        self.hidden_layers = nn.Conv1d(16, 16, 3)
        self.output_block = nn.Linear(6, 1)

    def forward(self, x):
        x = self.input_block(x.unsqueeze(1))
        x = self.hidden_layers(x)
        x = self.output_block(x)
        mu, std = x.mean(dim=1), x.std(dim=1)
        if self.explain_mode:
            return mu.unsqueeze(-1)
        else:
            return mu, std
