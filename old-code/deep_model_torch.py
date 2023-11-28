"""Implements things related to torch-based neural networks."""

import captum
import captum.attr as attr
import lightning.pytorch
import lightning.pytorch as pl
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from torchmetrics.regression import MeanSquaredError
from torchmetrics import MeanMetric
import ruptures as rpt
from landscape import RandomCoordinates, PCACoordinates, LossSurface

import bayes_layer as bl


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

        return preds.mean(dim=-1), preds.std(dim=-1)


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
        self,
        torch_model,
        loss_fun,
        lr,
        alpha=1e-3,
        lr_scheduler=None,
        log_bias=False,
        log_grad_norm=False,
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
        self.log_grad_norm = log_grad_norm
        self.grad_norm_metric = MeanMetric()

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

        if self.log_grad_norm and batch_idx > 0:
            grads = [
                param.grad.detach().flatten()
                for param in self.model.parameters()
                if param.grad is not None
            ]
            grad_norm = torch.cat(grads).norm()
            self.grad_norm_metric.update(grad_norm)
            self.log(
                "train_grad_norm",
                self.grad_norm_metric.compute().item(),
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
    torch_model = torch_model_from_cfg(cfg)
    criterion = eval(cfg["loss_fun"])(**(cfg["loss_fun_kwargs"] or {}))

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
        log_grad_norm=cfg["log_grad_norm"],
    )

    trainer.fit(pl_model, train_dataloaders=train_loader, val_dataloaders=test_loader)
    return pl_model


def get_loss_landscape(pl_model, X, y, loss_fun, **landscape_kwargs):
    # pcoords = RandomCoordinates(copy.deepcopy(list(pl_model.parameters())))
    loss_landscape = LossSurface(pl_model, X, y)
    loss_landscape.compile(
        loss_fun=loss_fun,
        **landscape_kwargs,
        # points=100,
        # coords=pcoords,
        # range_=1,
        # loss_fun=loss_fun,
        # surf=False,
        # device='mps',
    )

    return loss_landscape
    # _, ax = loss_surface.plot('Loss Landscape')


def plot_gt_pred(pl_model, X_train, y_train, X_test, y_test, cfg):
    with torch.no_grad():
        train_pred_mean, train_pred_std = pl_model(X_train)
        train_pred_mean = train_pred_mean.view(-1)
        train_pred_std = train_pred_std.view(-1)

        test_pred_mean, test_pred_std = pl_model(X_test)
        test_pred_mean = test_pred_mean.view(-1)
        test_pred_std = test_pred_std.view(-1)

    if cfg["show_change_points"]:
        algo = rpt.Pelt(model="rbf").fit(y_test)
        result = algo.predict(pen=10)

    fig, ax = plt.subplots()
    xs_time_train = list(range(1, len(train_pred_mean) + 1))
    xs_time_test = list(
        range(len(train_pred_mean), len(train_pred_mean) + len(test_pred_mean))
    )

    ax.plot(
        xs_time_train, y_train, label="Ground Truth: Training Set", color="tab:blue"
    )
    ax.plot(
        xs_time_train,
        train_pred_mean,
        label="Prediction: Training Set",
        color="tab:blue",
        linestyle=":",
    )
    ax.plot(xs_time_test, y_test, label="Ground Truth: Test Set", color="tab:orange")
    ax.plot(
        xs_time_test,
        test_pred_mean,
        label="Prediction: Test Set",
        color="tab:orange",
        linestyle=":",
    )
    ax.fill_between(
        xs_time_train,
        train_pred_mean - train_pred_std,
        train_pred_mean + train_pred_std,
        alpha=0.3,
    )
    ax.fill_between(
        xs_time_test,
        test_pred_mean - test_pred_std,
        test_pred_mean + test_pred_std,
        alpha=0.3,
    )

    if cfg["show_change_points"]:
        ax.vlines(result, y_test.min(), y_test.max(), ls="--")

    ax.set_xlabel("\(t\)")
    ax.set_ylabel("\(q\) (Sv)")
    ax.legend()

    return fig


def compute_bias(pl_model, X_train, y_train, X_test, y_test):
    with torch.no_grad():
        train_pred_mean, train_pred_std = pl_model(X_train)
        test_pred_mean, test_pred_std = pl_model(X_test)

        train_pred_mean, test_pred_mean = (
            train_pred_mean.squeeze(),
            test_pred_mean.squeeze(),
        )
        train_pred_std, test_pred_std = (
            train_pred_std.squeeze(),
            test_pred_std.squeeze(),
        )

        train_bias = train_pred_mean - y_train
        test_bias = test_pred_mean - y_test

    fig, ax = plt.subplots(figsize=(10, 8))

    xs_time_train = list(range(1, len(y_train) + 1))
    xs_time_test = list(range(len(y_train), len(y_train) + len(y_test)))

    ax.plot(xs_time_train, train_bias.cpu(), label="Bias: Training Set")
    ax.plot(xs_time_test, test_bias.cpu(), label="Bias: Test Set")

    ax.fill_between(
        xs_time_train,
        (train_bias - train_pred_std).cpu(),
        (train_bias + train_pred_std).cpu(),
        alpha=0.3,
    )
    ax.fill_between(
        xs_time_test,
        (test_bias - test_pred_std).cpu(),
        (test_bias + test_pred_std).cpu(),
        alpha=0.3,
    )

    ax.set_xlabel("\(t\)")
    ax.set_ylabel(r"\(\hat{q} - q\)")
    ax.legend()

    return fig


def torch_model_from_cfg(cfg):
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
    elif model_type in ["rnn", "lstm", "gru"]:
        torch_model = RNNModel(
            cfg["input_dim"],
            cfg["hidden_size"],
            1,
            cfg["num_layers"],
            recurrent_type=model_type,
        )

    return torch_model
