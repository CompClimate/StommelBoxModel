# from typing import Any, Dict, List, Optional, Tuple

# import hydra
# import lightning as L
# import rootutils
# from lightning import Callback, LightningModule, Trainer
# from lightning.pytorch.loggers import Logger
# from omegaconf import DictConfig

# from data.components.box_model import BoxModel
# from data.components.forcing import Forcing
# from data.time_series_datamodule import TimeSeriesDataModule
# from src.utils import (
#     RankedLogger,
#     extras,
#     get_metric_value,
#     instantiate_callbacks,
#     instantiate_loggers,
#     log_hyperparameters,
#     task_wrapper,
# )

# rootutils.setup_root(__file__, indicator=".project-root", pythonpath=True)
# log = RankedLogger(__name__, rank_zero_only=True)


# @task_wrapper
# def train(cfg: DictConfig) -> Tuple[Dict[str, Any], Dict[str, Any]]:
#     if cfg.get("seed"):
#         L.seed_everything(cfg.seed, workers=True)

#     log.info(f"Instantiating box model <{cfg.box_model._target_}>")
#     box_model: BoxModel = hydra.utils.instantiate(cfg.box_model)

#     log.info(f"Instantiating box model <{cfg.forcing._target_}>")
#     forcing: Forcing = hydra.utils.instantiate(cfg.forcing)

#     datamodule = TimeSeriesDataModule(box_model, forcing, **cfg.data)

#     log.info(f"Instantiating model <{cfg.model._target_}>")
#     model: LightningModule = hydra.utils.instantiate(cfg.model)

#     log.info("Instantiating callbacks...")
#     callbacks: List[Callback] = instantiate_callbacks(cfg.get("callbacks"))

#     log.info("Instantiating loggers...")
#     logger: List[Logger] = instantiate_loggers(cfg.get("logger"))

#     log.info(f"Instantiating trainer <{cfg.trainer._target_}>")
#     trainer: Trainer = hydra.utils.instantiate(
#         cfg.trainer, callbacks=callbacks, logger=logger
#     )

#     object_dict = {
#         "cfg": cfg,
#         "box_model": box_model,
#         "forcing": forcing,
#         "datamodule": datamodule,
#         "model": model,
#         "callbacks": callbacks,
#         "logger": logger,
#         "trainer": trainer,
#     }

#     if logger:
#         log.info("Logging hyperparameters!")
#         log_hyperparameters(object_dict)

#     if cfg.get("train"):
#         log.info("Starting training!")
#         trainer.fit(model=model, datamodule=datamodule, ckpt_path=cfg.get("ckpt_path"))

#     train_metrics = trainer.callback_metrics

#     if cfg.get("test"):
#         log.info("Starting testing!")
#         ckpt_path = trainer.checkpoint_callback.best_model_path
#         if ckpt_path == "":
#             log.warning("Best ckpt not found! Using current weights for testing...")
#             ckpt_path = None
#         trainer.test(model=model, datamodule=datamodule, ckpt_path=ckpt_path)
#         log.info(f"Best ckpt path: {ckpt_path}")

#     test_metrics = trainer.callback_metrics
#     metric_dict = {**train_metrics, **test_metrics}

#     return metric_dict, object_dict


# @hydra.main(version_base="1.3", config_path="../configs", config_name="train.yaml")
# def main(cfg: DictConfig) -> Optional[float]:
#     extras(cfg)

#     metric_dict, _ = train(cfg)
#     metric_value = get_metric_value(
#         metric_dict=metric_dict, metric_name=cfg.get("optimized_metric")
#     )

#     return metric_value


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


@hydra.main(version_base="1.3", config_path="../configs", config_name="explain.yaml")
def main(cfg: DictConfig) -> None:
    extras(cfg)
    execute_task(Task.Explain, cfg)


if __name__ == "__main__":
    main()
