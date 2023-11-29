import torch
import torch.nn as nn


class RNNModel(nn.Module):
    """Implements a reccurent model."""

    def __init__(
        self,
        input_dim,
        hidden_dim,
        output_dim,
        num_layers,
        recurrent_type="rnn",
        rnn_dropout=0.3,
        bidirectional=False,
        quantify_uncertainty=0.0,
    ):
        super().__init__()

        self.input_dim = input_dim
        self.hidden_size = hidden_dim
        self.output_dim = output_dim
        self.num_layers = num_layers
        self.rnn_dropout = rnn_dropout
        self.bidirectional = bidirectional
        self.quantify_uncertainty = quantify_uncertainty

        if recurrent_type == "rnn":
            rnn_cls = nn.RNN
        elif recurrent_type == "lstm":
            rnn_cls = nn.LSTM
        elif recurrent_type == "gru":
            rnn_cls = nn.GRU

        self.rnn = rnn_cls(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=rnn_dropout,
            bidirectional=bidirectional,
        )
        # if quantify_uncertainty > 0.0:
        # self.init_second_rnn_()
        self.fc = nn.Linear(hidden_dim * 2 if bidirectional else hidden_dim, output_dim)
        self.explain_mode = False

    def init_second_rnn_(self):
        self.rnn2 = nn.LSTM(
            input_size=self.hidden_size,
            hidden_size=self.hidden_size,
            num_layers=self.num_layers,
            batch_first=True,
            dropout=self.rnn_dropout,
            bidirectional=self.bidirectional,
        )

    def forward(self, x):
        x, _ = self.rnn(x)

        # if self.quantify_uncertainty > 0.0:
        #     x = F.dropout(x, self.quantify_uncertainty, training=True)
        #     x, _ = self.rnn2(x)
        #     x = F.dropout(x, self.quantify_uncertainty, training=True)

        # x = x[:, -1, :]

        x = self.fc(x)
        mu, std = x, torch.zeros_like(x)
        if self.explain_mode:
            return mu.unsqueeze(-1)
        else:
            return mu, std

    @torch.no_grad()
    def quantify(self, x, p, k):
        if not hasattr(self, "rnn2"):
            self.init_second_rnn_()

        quant_old = self.quantify_uncertainty
        self.quantify_uncertainty = p

        preds = torch.hstack([self(x) for _ in range(k)])

        self.quantify_uncertainty = quant_old

        return preds.mean(dim=-1), preds.std(dim=-1)
