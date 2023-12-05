import rootutils

rootutils.setup_root(__file__, indicator=".project-root", pythonpath=True)

from typing import Any, Dict, List, Optional, Tuple

import hydra
import lightning as L
import torch
from lightning import Callback, LightningModule, Trainer
from lightning.pytorch.loggers import Logger
from omegaconf import DictConfig, OmegaConf

from data.components.box_model import BoxModel, plot_time_series
from data.components.forcing import Forcing
from data.time_series_datamodule import TimeSeriesDatamodule
from src.utils import (RankedLogger, extras, get_metric_value,
                       instantiate_callbacks, instantiate_loggers,
                       log_hyperparameters, task_wrapper)
from utils.explainability import plot_attributions
from utils.plot_utils import compute_bias, plot_gt_pred, save_fig, setup_plt


def ite(i, t, e):
    return t if i else e


OmegaConf.register_new_resolver("ifthenelse", ite)


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
    # print(hydra.core.hydra_config.HydraConfig.get().runtime.choices.forcing)
    # print(hydra.core.hydra_config.HydraConfig.get())

    # set seed for random number generators in pytorch, numpy and python.random
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

    if cfg.get("train"):
        log.info("Starting training!")
        trainer.fit(model=model, datamodule=datamodule, ckpt_path=cfg.get("ckpt_path"))

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

    # merge train and test metrics
    metric_dict = {**train_metrics, **test_metrics}

    X_train = torch.from_numpy(datamodule.X_train)
    y_train = torch.from_numpy(datamodule.y_train)
    X_test = torch.from_numpy(datamodule.X_test)
    y_test = torch.from_numpy(datamodule.y_test)

    # Generate bias plot
    if cfg.get("plot_bias"):
        log.info("Plotting Bias plot...")
        fig_bias = compute_bias(
            model,
            X_train,
            y_train,
            X_test,
            y_test,
        )
        save_fig(fig_bias, working_dir, "bias", "pdf")

    if cfg.get("plot_data"):
        data_fig = plot_time_series(datamodule.series_dict)
        save_fig(data_fig, working_dir, "data", "pdf")

    # Generate ground truth -- prediction plot
    if cfg.get("plot_gt_pred"):
        log.info("Plotting [Ground Truth - Prediction] plot...")
        fig_gt_pred = plot_gt_pred(
            model.cpu(),
            X_train,
            y_train,
            X_test,
            y_test,
            cfg.get("show_change_points"),
        )
        save_fig(fig_gt_pred, working_dir, "pred-gt", "pdf")

    # Generate attribution plot
    if cfg.get("attr_algorithm") is not None:
        plot_attributions(
            model,
            torch.from_numpy(datamodule.X),
            cfg.attr_algorithm,
            cfg.feature_names,
            cfg.data.autoregressive,
            cfg.attr_ylabel,
            cfg.model.net.input_dim,
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
    # apply extra utilities
    # (e.g. ask for tags if none are provided in cfg, print cfg tree, etc.)
    extras(cfg)

    # train the model
    metric_dict, _ = train(cfg)

    # safely retrieve metric value for hydra-based hyperparameter optimization
    metric_value = get_metric_value(
        metric_dict=metric_dict, metric_name=cfg.get("optimized_metric")
    )

    # return optimized metric
    return metric_value


if __name__ == "__main__":
    main()
