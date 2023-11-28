import torch
import torch.nn as nn


class MLPModel(nn.Module):
    """Implements a simple MLP."""

    def __init__(self, input_dim, hidden_dim, n_hidden):
        super().__init__()
        self.input_block = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
        )
        self.hidden_layers = [
            nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim),
                nn.ReLU(),
            )
        ] * n_hidden
        self.hidden_layers = nn.Sequential(*self.hidden_layers)
        self.output_block = nn.Linear(hidden_dim, 1)
        self.explain_mode = False

    def forward(self, x):
        x = self.input_block(x)
        x = self.hidden_layers(x)
        x = self.output_block(x)
        if self.explain_mode:
            return x.unsqueeze(-1)
        else:
            return x, torch.zeros_like(x)
