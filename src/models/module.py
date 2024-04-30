from typing import Any, Union

import hydra
import lightning.pytorch as pl
import torch
import torch.nn as nn
import torchbnn
from torchmetrics import MeanMetric
from torchmetrics.regression import MeanSquaredError


class Model(pl.LightningModule):
    """A Pytorch Lightning wrapper around a torch model."""

    def __init__(
        self,
        net: nn.Module,
        optimizer: torch.optim.Optimizer,
        loss_fun: nn.Module,
        scheduler: torch.optim.lr_scheduler = None,
        log_bias: bool = False,
        log_grad_norm: bool = False,
        kl_weight: Union[None, float] = None,
    ):
        super().__init__()

        self.save_hyperparameters(logger=False)

        self.net = net
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.loss_fun = loss_fun
        self.train_loss = MeanSquaredError()
        self.val_loss = MeanSquaredError()
        self.log_bias = log_bias
        self.log_grad_norm = log_grad_norm
        self.grad_norm_metric = MeanMetric()
        self.kl_weight = kl_weight

        if kl_weight is not None:
            self.kl_loss = torchbnn.BKLLoss(reduction="mean", last_layer_only=False)

        self.log_kwargs = dict(
            on_step=False,
            on_epoch=True,
            prog_bar=True,
            sync_dist=True,
        )

    def forward(self, x):
        return self.net.forward(x)

    def training_step(self, batch, batch_idx):
        x, y = batch
        pred = self(x)

        if isinstance(pred, tuple):
            pred, _ = pred
        pred = pred.squeeze()
        loss = self.loss_fun(pred, y)

        if self.kl_weight is not None:
            kl = self.kl_loss(self.net)
            loss += (self.kl_weight * kl)[0]

        self.train_loss(pred, y)
        self.log(
            "train_loss",
            self.train_loss,
            **self.log_kwargs,
        )

        if self.log_bias:
            bias = (pred - y).mean()
            self.log(
                "train_bias",
                bias,
                **self.log_kwargs,
            )

        if self.log_grad_norm and batch_idx > 0:
            grads = [
                param.grad.detach().flatten()
                for param in self.net.parameters()
                if param.grad is not None
            ]
            grad_norm = torch.cat(grads).norm()
            self.grad_norm_metric.update(grad_norm)
            self.log(
                "train_grad_norm",
                self.grad_norm_metric.compute().item(),
                **self.log_kwargs,
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
            **self.log_kwargs,
        )

        if self.log_bias:
            bias = (pred - y).mean()
            self.log(
                "val_bias",
                bias,
                **self.log_kwargs,
            )

        return loss

    def configure_optimizers(self):
        optimizer = self.hparams.optimizer(params=self.trainer.model.parameters())
        if self.hparams.scheduler is not None:
            scheduler = self.hparams.scheduler(optimizer=optimizer)
            return {
                "optimizer": optimizer,
                "lr_scheduler": {
                    "scheduler": scheduler,
                    "monitor": "val/loss",
                    "interval": "epoch",
                    "frequency": 1,
                },
            }
        return {"optimizer": optimizer}


class EnsembleModel(pl.LightningModule):
    def __init__(
        self,
        net_cfg,
        num_models: int,
        optimizer: torch.optim.Optimizer,
        loss_fun: nn.Module,
        scheduler: torch.optim.lr_scheduler = None,
        log_bias: bool = False,
        log_grad_norm: bool = False,
    ):
        super().__init__()

        self.save_hyperparameters(logger=False)
        self.automatic_optimization = False

        try:
            net_cfg["_target_"] = net_cfg.pop("__target__")
        except BaseException:
            pass
        self.nets = nn.ModuleList(
            [
                hydra.utils.instantiate(net_cfg).to(self.device)
                for _ in range(num_models)
            ]
        )
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.loss_fun = loss_fun
        self.train_loss = MeanSquaredError()
        self.val_loss = MeanSquaredError()
        self.log_bias = log_bias
        self.log_grad_norm = log_grad_norm
        self.grad_norm_metric = MeanMetric()

        self.explain_mode = False

        self.log_kwargs = dict(
            on_step=False,
            on_epoch=True,
            prog_bar=True,
            sync_dist=True,
        )

    def to(self, *args: Any, **kwargs: Any):
        device, dtype = torch._C._nn._parse_to(*args, **kwargs)[:2]
        self.nets = self.nets.to(device=device, dtype=dtype)
        return super().to(*args, **kwargs)

    def cuda(self, device=None):
        if device is None:
            device = torch.device("cuda", torch.cuda.current_device())
        elif isinstance(device, int):
            device = torch.device("cuda", index=device)
        self.nets = self.nets.to(device=device)
        return super().cuda(device=device)

    def cpu(self):
        device = torch.device("cpu")
        self.nets = self.nets.to(device=device)
        return super().cpu()

    def forward(self, x):
        preds = [net.forward(x) for net in self.nets]

        convert = isinstance(preds[0], tuple)
        for i in range(len(preds)):
            if convert:
                preds[i] = preds[i][0]
            if len(preds[i].size()) == 3:
                preds[i] = preds[i].squeeze(-1)

        preds = torch.stack(preds, dim=-1)
        mean = preds.mean(dim=-1)
        std = preds.std(dim=-1)

        if self.explain_mode:
            return mean.unsqueeze(-1)
        else:
            return mean, std

    def training_step(self, batch, batch_idx):
        x, y = batch

        preds = []

        for net, optimizer in zip(self.nets, self.optimizers()):
            self.toggle_optimizer(optimizer)
            pred, _ = net(x)
            loss = self.loss_fun(pred.squeeze(), y)
            optimizer.zero_grad()
            self.manual_backward(loss)
            optimizer.step()
            self.untoggle_optimizer(optimizer)

            preds.append(pred)

        logits = torch.stack(preds, dim=-1)
        mean_logit = logits.mean(dim=-1)
        self.train_loss(mean_logit, y.unsqueeze(-1))

        self.log(
            "train_loss",
            self.train_loss,
            **self.log_kwargs,
        )

        if self.log_bias:
            biases = [(pred - y).mean() for pred in preds]
            self.log(
                "train_bias",
                torch.cat(biases).mean(),
                **self.log_kwargs,
            )

        if self.log_grad_norm and batch_idx > 0:
            grads = [
                [
                    param.grad.detach().flatten()
                    for param in net.parameters()
                    if param.grad is not None
                ]
                for net in self.nets
            ]
            grad_norm = torch.cat(grads).norm()
            self.grad_norm_metric.update(grad_norm)
            self.log(
                "train_grad_norm",
                self.grad_norm_metric.compute().item(),
                **self.log_kwargs,
            )

    def validation_step(self, batch, batch_idx):
        x, y = batch

        preds = [net(x) for net in self.nets]
        if isinstance(preds[0], tuple):
            preds = [pred[0] for pred in preds]

        logits = torch.stack(preds, dim=-1)
        mean_logit = logits.mean(dim=-1)
        self.val_loss(mean_logit, y.unsqueeze(-1))

        self.log(
            "val_loss",
            self.val_loss,
            **self.log_kwargs,
        )

        if self.log_bias:
            biases = [(pred - y).mean() for pred in preds]
            self.log(
                "val_bias",
                biases,
                **self.log_kwargs,
            )

    def configure_optimizers(self):
        optimizers = [
            self.hparams.optimizer(params=net.parameters()) for net in self.nets
        ]
        return optimizers, []
