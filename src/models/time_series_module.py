import torch
import lightning.pytorch as pl
from torchmetrics.regression import MeanSquaredError
from torchmetrics import MeanMetric


class Model(pl.LightningModule):
    """A Pytorch Lightning wrapper around a torch model."""

    def __init__(
        self,
        net: torch.nn.Module,
        optimizer: torch.optim.Optimizer,
        scheduler: torch.optim.lr_scheduler,
        loss_fun: torch.nn.Module,
        log_bias: bool = False,
        log_grad_norm: bool = False,
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

    def forward(self, x):
        return self.net.forward(x)

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
                for param in self.net.parameters()
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
