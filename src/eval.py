# from typing import Any, Dict, List, Tuple

# import hydra
# import rootutils
# from lightning import LightningModule, Trainer
# from lightning.pytorch.loggers import Logger
# from omegaconf import DictConfig
# from utils.plot_utils import compute_bias, plot_gt_pred, save_fig

# from data.components.box_model import BoxModel
# from data.components.forcing import Forcing
# from data.time_series_datamodule import TimeSeriesDataModule
# from src.utils import (
#     RankedLogger,
#     extras,
#     instantiate_loggers,
#     log_hyperparameters,
#     task_wrapper,
# )

# rootutils.setup_root(__file__, indicator=".project-root", pythonpath=True)
# log = RankedLogger(__name__, rank_zero_only=True)


# @task_wrapper
# def evaluate(cfg: DictConfig) -> Tuple[Dict[str, Any], Dict[str, Any]]:
#     assert cfg.ckpt_path

#     working_dir = hydra.core.hydra_config.HydraConfig.get().runtime.output_dir

#     log.info(f"Instantiating box model <{cfg.box_model._target_}>")
#     box_model: BoxModel = hydra.utils.instantiate(cfg.box_model)

#     log.info(f"Instantiating box model <{cfg.forcing._target_}>")
#     forcing: Forcing = hydra.utils.instantiate(cfg.forcing)

#     datamodule = TimeSeriesDataModule(box_model, forcing, **cfg.data)

#     log.info(f"Instantiating model <{cfg.model._target_}>")
#     model: LightningModule = hydra.utils.instantiate(cfg.model)

#     log.info("Instantiating loggers...")
#     logger: List[Logger] = instantiate_loggers(cfg.get("logger"))

#     log.info(f"Instantiating trainer <{cfg.trainer._target_}>")
#     trainer: Trainer = hydra.utils.instantiate(cfg.trainer, logger=logger)

#     object_dict = {
#         "cfg": cfg,
#         "box_model": box_model,
#         "forcing": forcing,
#         "datamodule": datamodule,
#         "model": model,
#         "logger": logger,
#         "trainer": trainer,
#     }

#     if logger:
#         log.info("Logging hyperparameters!")
#         log_hyperparameters(object_dict)

#     log.info("Starting testing!")
#     trainer.test(model=model, datamodule=datamodule, ckpt_path=cfg.ckpt_path)

#     # Generate bias plot
#     if cfg.get("plot_bias"):
#         log.info("Plotting Bias plot...")
#         fig_bias = compute_bias(
#             model,
#             datamodule.X_train,
#             datamodule.y_train,
#             datamodule.X_test,
#             datamodule.y_test,
#         )
#         save_fig(fig_bias, working_dir, "bias", "pdf")

#     # Generate ground truth -- prediction plot
#     if cfg.get("plot_gt_pred"):
#         log.info("Plotting [Ground Truth - Prediction] plot...")
#         fig_gt_pred = plot_gt_pred(
#             model.cpu(),
#             datamodule.X_train.cpu(),
#             datamodule.y_train.cpu(),
#             datamodule.X_test.cpu(),
#             datamodule.y_test.cpu(),
#             show_change_points=cfg.get("show_change_points"),
#         )
#         save_fig(fig_gt_pred, working_dir, "pred-gt", "pdf")

#     metric_dict = trainer.callback_metrics

#     return metric_dict, object_dict


# @hydra.main(version_base="1.3", config_path="../configs", config_name="eval.yaml")
# def main(cfg: DictConfig):
#     extras(cfg)
#     evaluate(cfg)


# if __name__ == "__main__":
#     main()


import hydra
import rootutils
from omegaconf import DictConfig

rootutils.setup_root(__file__, indicator=".project-root", pythonpath=True)

from utils import (
    RankedLogger,
    Task,
    execute_task,
    extras,
    register_resolvers,
)

register_resolvers()

log = RankedLogger(__name__, rank_zero_only=True)


@hydra.main(version_base="1.3", config_path="../configs", config_name="eval.yaml")
def main(cfg: DictConfig) -> None:
    extras(cfg)
    execute_task(Task.Eval, cfg)


if __name__ == "__main__":
    main()
