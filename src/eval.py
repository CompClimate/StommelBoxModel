from typing import Any, Dict, List, Tuple

import hydra
import rootutils
from lightning import LightningDataModule, LightningModule, Trainer
from lightning.pytorch.loggers import Logger
from omegaconf import DictConfig

from data.components.box_model import BoxModel
from data.components.forcing import Forcing
from data.time_series_datamodule import TimeSeriesDatamodule
from src.utils import (RankedLogger, extras, instantiate_loggers,
                       log_hyperparameters, task_wrapper)
from utils.plot_utils import compute_bias, plot_gt_pred, save_fig

rootutils.setup_root(__file__, indicator=".project-root", pythonpath=True)
# ------------------------------------------------------------------------------------ #
# the setup_root above is equivalent to:
# - adding project root dir to PYTHONPATH
#       (so you don't need to force user to install project as a package)
#       (necessary before importing any local modules e.g. `from src import utils`)
# - setting up PROJECT_ROOT environment variable
#       (which is used as a base for paths in "configs/paths/default.yaml")
#       (this way all filepaths are the same no matter where you run the code)
# - loading environment variables from ".env" in root dir
#
# you can remove it if you:
# 1. either install project as a package or move entry files to project root dir
# 2. set `root_dir` to "." in "configs/paths/default.yaml"
#
# more info: https://github.com/ashleve/rootutils
# ------------------------------------------------------------------------------------ #


log = RankedLogger(__name__, rank_zero_only=True)


@task_wrapper
def evaluate(cfg: DictConfig) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    """Evaluates given checkpoint on a datamodule testset.

    This method is wrapped in optional @task_wrapper decorator, that controls the behavior during
    failure. Useful for multiruns, saving info about the crash, etc.

    :param cfg: DictConfig configuration composed by Hydra.
    :return: Tuple[dict, dict] with metrics and dict with all instantiated objects.
    """
    assert cfg.ckpt_path

    working_dir = hydra.core.hydra_config.HydraConfig.get().runtime.output_dir

    log.info(f"Instantiating box model <{cfg.box_model._target_}>")
    box_model: BoxModel = hydra.utils.instantiate(cfg.box_model)

    log.info(f"Instantiating box model <{cfg.forcing._target_}>")
    forcing: Forcing = hydra.utils.instantiate(cfg.forcing)

    datamodule = TimeSeriesDatamodule(box_model, forcing, **cfg.data)

    log.info(f"Instantiating model <{cfg.model._target_}>")
    model: LightningModule = hydra.utils.instantiate(cfg.model)

    log.info("Instantiating loggers...")
    logger: List[Logger] = instantiate_loggers(cfg.get("logger"))

    log.info(f"Instantiating trainer <{cfg.trainer._target_}>")
    trainer: Trainer = hydra.utils.instantiate(cfg.trainer, logger=logger)

    object_dict = {
        "cfg": cfg,
        "box_model": box_model,
        "forcing": forcing,
        "datamodule": datamodule,
        "model": model,
        "logger": logger,
        "trainer": trainer,
    }

    if logger:
        log.info("Logging hyperparameters!")
        log_hyperparameters(object_dict)

    log.info("Starting testing!")
    trainer.test(model=model, datamodule=datamodule, ckpt_path=cfg.ckpt_path)

    # Generate bias plot
    if cfg.get("plot_bias"):
        log.info("Plotting Bias plot...")
        fig_bias = compute_bias(
            model,
            datamodule.X_train,
            datamodule.y_train,
            datamodule.X_test,
            datamodule.y_test,
        )
        save_fig(fig_bias, working_dir, "bias", "pdf")

    # Generate ground truth -- prediction plot
    if cfg.get("plot_gt_pred"):
        log.info("Plotting [Ground Truth - Prediction] plot...")
        fig_gt_pred = plot_gt_pred(
            model.cpu(),
            datamodule.X_train.cpu(),
            datamodule.y_train.cpu(),
            datamodule.X_test.cpu(),
            datamodule.y_test.cpu(),
            cfg.get("show_change_points"),
        )
        save_fig(fig_gt_pred, working_dir, "pred-gt", "pdf")

    metric_dict = trainer.callback_metrics

    return metric_dict, object_dict


@hydra.main(version_base="1.3", config_path="../configs", config_name="eval.yaml")
def main(cfg: DictConfig) -> None:
    """Main entry point for evaluation.

    :param cfg: DictConfig configuration composed by Hydra.
    """
    # apply extra utilities
    # (e.g. ask for tags if none are provided in cfg, print cfg tree, etc.)
    extras(cfg)

    evaluate(cfg)


if __name__ == "__main__":
    main()
