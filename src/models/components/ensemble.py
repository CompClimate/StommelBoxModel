import torch
import torch.nn as nn

import src


class TorchEnsemble(nn.Module):
    """Implements a deep ensemble given a list of models."""

    def __init__(self, model_cls, num_models, input_dim, hidden_dim, n_hidden):
        super().__init__()
        self.models = [
            eval(model_cls)(input_dim, hidden_dim, n_hidden) for _ in range(num_models)
        ]
        self.explain_mode = False
        for i, model in enumerate(self.models):
            self.register_module(f"model_{i}", model)

    def forward(self, x):
        preds = torch.stack([model(x)[0] for model in self.models], dim=-1)
        mu, std = preds.mean(dim=-1), preds.std(dim=-1)
        if self.explain_mode:
            return mu.unsqueeze(-1)
        else:
            return mu, std
