"""Implements things related to torch-based neural networks."""

import captum
import captum.attr as attr
import lightning
import lightning.pytorch
import lightning.pytorch as pl
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from sklearn.preprocessing import MinMaxScaler
from torch.utils.data import DataLoader, TensorDataset
from torchmetrics.regression import MeanSquaredError
import ruptures as rpt
import copy
from landscape import RandomCoordinates, PCACoordinates, LossSurface
import networkx as nx

import bayes_layer as bl
from utils import explain_captum, heatmap, explain_mode


def sliding_windows(data, seq_length):
    """
    Transforms a 1d time series into sliding windows of length `seq_length`.
    """
    x = []
    y = []

    for i in range(len(data) - seq_length - 1):
        _x = data[i : (i + seq_length)]
        _y = data[i + seq_length]
        x.append(_x)
        y.append(_y)

    return np.array(x), np.array(y)


def prepare_data(train, test, seq_len, scale=False):
    """From a train/test split, optionally scaled the data and returns sliding windows."""
    if scale:
        sc = MinMaxScaler()
        train = sc.fit_transform(train)
        test = sc.fit_transform(test)

    X_train, y_train = sliding_windows(train, seq_len)
    X_test, y_test = sliding_windows(test, seq_len)

    X_train, X_test = X_train.astype(np.float32), X_test.astype(np.float32)
    y_train, y_test = y_train.astype(np.float32), y_test.astype(np.float32)

    return (X_train, y_train), (X_test, y_test)


class RNNModel(nn.Module):
    """Implements a reccurent model."""

    def __init__(
        self,
        input_size,
        hidden_size,
        output_size,
        num_layers,
        recurrent_type="rnn",
        rnn_dropout=0.3,
        bidirectional=False,
        quantify_uncertainty=0.0,
    ):
        super().__init__()

        self.output_size = output_size
        self.num_layers = num_layers
        self.input_size = input_size
        self.hidden_size = hidden_size
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
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=rnn_dropout,
            bidirectional=bidirectional,
        )
        # if quantify_uncertainty > 0.0:
        # self.init_second_rnn_()
        self.fc = nn.Linear(
            hidden_size * 2 if bidirectional else hidden_size, output_size
        )
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

        return preds.mean(dim=-1), 6 * preds.std(dim=-1)


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


class ConvModel(nn.Module):
    def __init__(self):
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


class TorchEnsemble(nn.Module):
    """Implements a deep ensemble given a list of models."""

    def __init__(self, models):
        super().__init__()
        self.models = models
        self.explain_mode = False
        for i, model in enumerate(self.models):
            self.register_module(f"model_{i}", model)

    def forward(self, x):
        preds = torch.hstack([model(x)[0] for model in self.models])
        mu, std = preds.mean(dim=-1), preds.std(dim=-1)
        if self.explain_mode:
            return mu.unsqueeze(-1)
        else:
            return mu, std


class Model(pl.LightningModule):
    """A Pytorch Lightning wrapper around a torch model."""

    def __init__(
        self, torch_model, loss_fun, lr, alpha=1e-3, lr_scheduler=None, log_bias=False
    ):
        super().__init__()
        self.model = torch_model
        self.loss_fun = loss_fun
        self.lr = lr
        self.alpha = alpha
        self.lr_scheduler = lr_scheduler
        self.train_loss = MeanSquaredError()
        self.val_loss = MeanSquaredError()
        self.log_bias = log_bias

    def forward(self, x):
        return self.model.forward(x)

    def training_step(self, batch, batch_idx):
        x, y = batch
        pred = self(x)

        if isinstance(pred, tuple):
            pred, _ = pred
        pred = pred.squeeze()
        loss = self.loss_fun(pred, y)

        self.train_loss(pred, y)
        self.log(
            "train_loss",
            self.train_loss,
            on_step=False,
            on_epoch=True,
            prog_bar=True,
            sync_dist=True,
        )

        if self.log_bias:
            bias = (pred - y).mean()
            self.log(
                "train_bias",
                bias,
                on_step=False,
                on_epoch=True,
                prog_bar=True,
                sync_dist=True,
            )

        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        pred = self(x)

        if isinstance(pred, tuple):
            pred, _ = pred
        pred = pred.squeeze()
        loss = self.loss_fun(pred, y)

        self.val_loss(pred, y)
        self.log(
            "val_loss",
            self.val_loss,
            on_step=False,
            on_epoch=True,
            prog_bar=True,
            sync_dist=True,
        )

        if self.log_bias:
            bias = (pred - y).mean()
            self.log(
                "train_bias",
                bias,
                on_step=False,
                on_epoch=True,
                prog_bar=True,
                sync_dist=True,
            )

        return loss

    def configure_optimizers(self):
        optimizer = optim.Adam(
            self.model.parameters(),
            lr=self.lr,
            weight_decay=self.alpha,
        )
        lr_scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=10)
        # return dict(
        #     optimizer=optimizer,
        #     lr_scheduler=lr_scheduler,
        # )
        return optimizer


def train(
    X_train,
    y_train,
    X_test,
    y_test,
    cfg,
):
    """Trains a torch model given a training configuration."""
    model_type = cfg["model_type"]

    if model_type == "ensemble":
        torch_model = TorchEnsemble(
            [
                MLPModel(
                    cfg["input_dim"],
                    cfg["hidden_size"],
                    cfg["num_layers"],
                )
                for _ in range(cfg["num_models"])
            ]
        )
    elif model_type == "bnn":
        torch_model = BayesMLPModel(
            cfg["input_dim"],
            cfg["hidden_size"],
            cfg["num_layers"],
            num_MC=cfg["num_models"],
        )
    elif model_type == "mlp":
        torch_model = MLPModel(
            cfg["input_dim"],
            cfg["hidden_size"],
            cfg["num_layers"],
        )
    elif model_type == "conv":
        torch_model = ConvModel()
    elif model_type == "rnn":
        torch_model = RNNModel(
            cfg["input_dim"],
            cfg["hidden_size"],
            1,
            cfg["num_layers"],
            recurrent_type=cfg["recurrent_type"],
        )

    criterion = nn.MSELoss()

    train_set = TensorDataset(X_train, y_train)
    test_set = TensorDataset(X_test, y_test)

    train_loader = DataLoader(
        train_set,
        batch_size=cfg["batch_size"],
    )
    test_loader = DataLoader(
        test_set,
        batch_size=cfg["batch_size"],
    )

    trainer = pl.Trainer(
        max_epochs=cfg["num_epochs"],
        logger=[
            eval(logger_name)(**logger_kwargs)
            for logger_name, logger_kwargs in cfg["loggers"].items()
        ],
    )

    pl_model = Model(
        torch_model,
        criterion,
        cfg["lr"],
        alpha=cfg["l2_alpha"],
        log_bias=cfg["compute_bias"],
    )

    trainer.fit(pl_model, train_dataloaders=train_loader, val_dataloaders=test_loader)
    return pl_model


def get_loss_landscape(pl_model, X, y, loss_fun, **landscape_kwargs):
    pcoords = RandomCoordinates(copy.deepcopy(list(pl_model.parameters())))
    loss_landscape = LossSurface(pl_model, X, y)

    c_range = 1
    loss_landscape.compile(
        loss_fun=loss_fun,
        **landscape_kwargs,
        # points=100,
        # coords=pcoords,
        # range_=c_range,
        # loss_fun=loss_fun,
        # surf=False,
        # device='mps',
    )

    return loss_landscape
    # _, ax = loss_surface.plot('Loss Landscape')


def plot_gt_pred(pl_model, X_train, y_train, X_test, y_test, cfg):
    with torch.no_grad():
        pl_model = pl_model.to("cpu")
        train_pred_mean, train_pred_std = pl_model(X_train)
        train_pred_mean = train_pred_mean.view(-1)
        train_pred_std = train_pred_std.view(-1)

        test_pred_mean, test_pred_std = pl_model(X_test)
        test_pred_mean = test_pred_mean.view(-1)
        test_pred_std = test_pred_std.view(-1)

    if cfg["show_change_points"]:
        algo = rpt.Pelt(model="rbf").fit(y_test)
        result = algo.predict(pen=10)

    fig, ax = plt.subplots(ncols=2)

    ax[0].plot(y_train, label="Ground Truth")
    ax[0].plot(train_pred_mean, label="Prediction")
    ax[0].fill_between(
        list(range(1, len(train_pred_mean) + 1)),
        train_pred_mean - train_pred_std,
        train_pred_mean + train_pred_std,
        alpha=0.3,
    )
    ax[0].set_title("Training Set")
    ax[0].legend()

    ax[1].plot(y_test, label="Ground Truth")
    ax[1].plot(test_pred_mean, label="Prediction")
    ax[1].fill_between(
        list(range(1, len(test_pred_mean) + 1)),
        test_pred_mean - test_pred_std,
        test_pred_mean + test_pred_std,
        alpha=0.3,
    )
    if cfg["show_change_points"]:
        ax[1].vlines(result, y_test.min(), y_test.max(), ls="--")
    ax[1].set_title("Test Set")
    ax[1].legend()

    return fig


def compute_bias(pl_model, X_train, y_train, X_test, y_test):
    with torch.no_grad():
        pred_train = pl_model(X_train)[0].reshape(-1)
        pred_test = pl_model(X_test)[0].reshape(-1)
        train_bias = pred_train - y_train
        test_bias = pred_test - y_test

    fig, ax_bias = plt.subplots(ncols=2)
    ax_bias[0].plot(train_bias)
    ax_bias[0].set_title("Training Bias")
    ax_bias[1].plot(test_bias)
    ax_bias[1].set_title("Test Bias")

    return fig


def attribute(pl_model, alg_cls, X_test, feature_names, ylabel):
    with explain_mode(pl_model.model):
        attrs = explain_captum(
            pl_model.model,
            alg_cls,
            X_test,
            feature_names,
        )

    fig, ax = plt.subplots()
    heatmap(attrs, fig=fig, ax=ax, ylabel=ylabel, show=False)

    return fig
