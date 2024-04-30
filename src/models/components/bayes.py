import torch.nn as nn
import torchbnn


class BNNTorch(nn.Module):
    def __init__(
        self, input_dim, hidden_dim, n_hidden, num_MC, prior_mu=0, prior_sigma=0.1
    ):
        super().__init__()
        self.input_block = nn.Sequential(
            torchbnn.BayesLinear(prior_mu, prior_sigma, input_dim, hidden_dim),
            nn.ReLU(),
        )

        self.hidden_layers = []
        for _ in range(n_hidden):
            self.hidden_layers.append(
                nn.Sequential(
                    torchbnn.BayesLinear(prior_mu, prior_sigma, hidden_dim, hidden_dim),
                    nn.ReLU(),
                )
            )
        self.hidden_layers = nn.Sequential(*self.hidden_layers)
        self.output_block = torchbnn.BayesLinear(prior_mu, prior_sigma, hidden_dim, 1)
        self.num_MC = num_MC
        self.explain_mode = False

    def forward(self, x):
        x = self.input_block(x)
        x = self.hidden_layers(x)
        x = self.output_block(x)

        if self.explain_mode:
            return x.unsqueeze(-1)
        else:
            return x
