import rootutils

rootutils.setup_root(__file__, indicator=".project-root", pythonpath=True)

import os
import os.path as osp
import pickle as pkl
from typing import Any, Dict, List, Optional, Tuple

import hydra
import lightning as L
import torch
from captum.attr._utils.lrp_rules import IdentityRule
from lightning import Callback, LightningModule, Trainer
from lightning.pytorch.loggers import Logger
from omegaconf import DictConfig, OmegaConf

from data.components.box_model import BoxModel, plot_time_series
from data.components.forcing import Forcing
from data.time_series_datamodule import TimeSeriesDatamodule
from src.models.time_series_module import Model
from src.utils import (
    RankedLogger,
    extras,
    get_metric_value,
    instantiate_callbacks,
    instantiate_loggers,
    log_hyperparameters,
    task_wrapper,
)
from utils.explainability import plot_attributions
from utils.landscape_utils import LossLandscape, RandomCoordinates, plot_training_path
from utils.plot_utils import compute_bias, plot_gt_pred, save_fig, setup_plt


def ite(i, t, e):
    return t if i else e


def equals(l, r):
    return l == r


OmegaConf.register_new_resolver("ifthenelse", ite)
OmegaConf.register_new_resolver("equals", equals)


setup_plt()

log = RankedLogger(__name__, rank_zero_only=True)


@task_wrapper
def train(cfg: DictConfig) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    """Trains the model. Can additionally evaluate on a testset, using best weights obtained during
    training.

    This method is wrapped in optional @task_wrapper decorator, that controls the behavior during
    failure. Useful for multiruns, saving info about the crash, etc.

    :param cfg: A DictConfig configuration composed by Hydra.
    :return: A tuple with metrics and dict with all instantiated objects.
    """
    # print(hydra.core.hydra_config.HydraConfig.get().runtime)
    # print(hydra.core.hydra_config.HydraConfig.get().runtime.choices.density.nonlinear_density)
    # print(hydra.core.hydra_config.HydraConfig.get())

    # Set seed for random number generators in pytorch, numpy and python.random
    if cfg.get("seed"):
        L.seed_everything(cfg.seed, workers=True)

    working_dir = hydra.core.hydra_config.HydraConfig.get().runtime.output_dir

    log.info(f"Instantiating box model <{cfg.box_model._target_}>")
    box_model: BoxModel = hydra.utils.instantiate(cfg.box_model)

    log.info(f"Instantiating S forcing <{cfg.s_forcing._target_}>")
    s_forcing: Forcing = hydra.utils.instantiate(cfg.s_forcing)
    log.info(f"Instantiating T forcing <{cfg.s_forcing._target_}>")
    t_forcing: Forcing = hydra.utils.instantiate(cfg.t_forcing)

    datamodule = TimeSeriesDatamodule(
        box_model,
        s_forcing,
        t_forcing,
        **cfg.density,
        **cfg.data,
    )

    input_dim = (
        cfg.data.window_size
        if cfg.data.autoregressive
        else sum(map(lambda x: len(x), datamodule.series_dict["features"].values()))
    )
    cfg.model.net.input_dim = input_dim

    log.info(f"Instantiating model <{cfg.model._target_}>")
    model: LightningModule = hydra.utils.instantiate(cfg.model)

    log.info("Instantiating callbacks...")
    callbacks: List[Callback] = instantiate_callbacks(cfg.get("callbacks"))

    log.info("Instantiating loggers...")
    logger: List[Logger] = instantiate_loggers(cfg.get("logger"))

    log.info(f"Instantiating trainer <{cfg.trainer._target_}>")
    trainer: Trainer = hydra.utils.instantiate(
        cfg.trainer,
        callbacks=callbacks,
        logger=logger,
    )

    object_dict = {
        "cfg": cfg,
        "box_model": box_model,
        "s_forcing": s_forcing,
        "t_forcing": t_forcing,
        "datamodule": datamodule,
        "model": model,
        "callbacks": callbacks,
        "logger": logger,
        "trainer": trainer,
    }

    if logger:
        log.info("Logging hyperparameters!")
        log_hyperparameters(object_dict)

    if ckpt_name := cfg.get("ckpt_name"):
        log.info("Loading model from checkpoint.")
        ckpt_path = osp.join(working_dir, "csv", "version_0", "checkpoints", ckpt_name)
        model = Model.load_from_checkpoint(ckpt_path)

    if cfg.get("train"):
        log.info("Starting training!")
        trainer.fit(model=model, datamodule=datamodule)

    train_metrics = trainer.callback_metrics

    if cfg.get("test"):
        log.info("Starting testing!")
        ckpt_path = trainer.checkpoint_callback.best_model_path
        if ckpt_path == "":
            log.warning("Best ckpt not found! Using current weights for testing...")
            ckpt_path = None
        trainer.test(model=model, datamodule=datamodule, ckpt_path=ckpt_path)
        log.info(f"Best ckpt path: {ckpt_path}")

    test_metrics = trainer.callback_metrics
    metric_dict = {**train_metrics, **test_metrics}

    X_train = torch.from_numpy(datamodule.X_train)
    y_train = torch.from_numpy(datamodule.y_train)
    X_test = torch.from_numpy(datamodule.X_test)
    y_test = torch.from_numpy(datamodule.y_test)

    model.eval()

    if plot_cfg := cfg.get("plotting"):
        # Generate bias plot
        if (plot_bias := plot_cfg.get("bias")) and plot_bias.plot:
            log.info("Plotting Bias plot...")
            fig_bias = compute_bias(
                model,
                X_train,
                y_train,
                X_test,
                y_test,
            )
            save_fig(fig_bias, working_dir, "bias", "pdf")

        if (plot_data := plot_cfg.get("data")) and plot_data:
            data_fig = plot_time_series(datamodule.series_dict)
            save_fig(data_fig, working_dir, "data", "pdf")

        # Generate ground truth -- prediction plot
        if (
            groundtruth_prediction := plot_cfg.get("groundtruth_prediction")
        ) and groundtruth_prediction.plot:
            log.info("Plotting [Ground Truth - Prediction] plot...")
            fig_gt_pred = plot_gt_pred(
                model.cpu(),
                X_train,
                y_train,
                X_test,
                y_test,
                plot_cfg.get("show_change_points"),
            )
            save_fig(fig_gt_pred, working_dir, "groundtruth-prediction", "pdf")

        if (
            loss_landscape_cfg := plot_cfg.get("loss_landscape")
        ) and loss_landscape_cfg.plot:
            log.info("Plotting loss landscape...")

            with open(
                osp.join(working_dir, loss_landscape_cfg.relative_path), "rb"
            ) as f:
                training_path = pkl.load(f)

            for i in range(len(training_path)):
                for j in range(len(training_path[i])):
                    training_path[i][j] = training_path[i][j].cpu()

            ll = LossLandscape(model, datamodule.train_dataloader())
            coords = RandomCoordinates(training_path[0], loss_landscape_cfg.dim)

            surface = loss_landscape_cfg.dim == 3

            ll.compile(
                loss_landscape_cfg.range,
                loss_landscape_cfg.points,
                coords,
                model.loss_fun,
                surface,
            )
            fig, ax = ll.plot(loss_landscape_cfg.get("title"), surface=surface)

            if loss_landscape_cfg.plot_training_path:
                fig, ax = plot_training_path(coords, training_path, fig, ax)

            save_fig(fig, working_dir, "loss_landscape", "pdf")

    # Generate attribution plot
    if explain_cfg := cfg.get("explainability"):
        data_cfg = cfg.data

        autoregressive = data_cfg.autoregressive
        feature_names = (
            [rf"\(t - {i}\)" for i in reversed(range(1, input_dim + 1))]
            if autoregressive
            else list(datamodule.series_dict["latex"]["variables"].values())
            + list(datamodule.series_dict["latex"]["forcings"].values())
        )
        data = torch.from_numpy(datamodule.X)
        # Ignore the loss function module and trainining/validation metrics in the Lightning module
        model.loss_fun.rule = IdentityRule()
        model.train_loss.rule = IdentityRule()
        model.val_loss.rule = IdentityRule()
        model.grad_norm_metric.rule = IdentityRule()

        for algorithm_cfg in explain_cfg.values():
            algorithm = hydra.utils.instantiate(algorithm_cfg.algorithm, model)
            ylabel = algorithm_cfg.get("ylabel")
            plot_attributions(
                model,
                data,
                algorithm,
                feature_names,
                ylabel,
                working_dir,
                "pdf",
            )

    return metric_dict, object_dict


@hydra.main(version_base="1.3", config_path="../configs", config_name="train.yaml")
def main(cfg: DictConfig) -> Optional[float]:
    """Main entry point for training.

    :param cfg: DictConfig configuration composed by Hydra.
    :return: Optional[float] with optimized metric value.
    """
    extras(cfg)

    # Train the model
    metric_dict, _ = train(cfg)

    # Safely retrieve metric value for hydra-based hyperparameter optimization
    metric_value = get_metric_value(
        metric_dict=metric_dict, metric_name=cfg.get("optimized_metric")
    )

    # Return optimized metric
    return metric_value


if __name__ == "__main__":
    main()
