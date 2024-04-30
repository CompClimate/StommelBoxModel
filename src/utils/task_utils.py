import itertools
from enum import StrEnum
from pathlib import Path
from typing import Any, Tuple

import hydra
import lightning as L
import numpy as np
import polars as pl
import rootutils
import torch
from omegaconf import DictConfig

from data.kfold_datamodule import LitKFoldDataModule

from .hydra_utils import get_working_dir
from .instantiators import instantiate_essentials
from .logging_utils import log_hyperparameters
from .pylogger import RankedLogger
from .utils import task_wrapper
from .xai_utils import attribute

rootutils.setup_root(__file__, indicator=".project-root", pythonpath=True)
log = RankedLogger(__name__, rank_zero_only=True)


class Task(StrEnum):
    Train = "train"
    Eval = "eval"
    Explain = "explain"
    CrossValidate = "cross_validate"


class explain_mode:
    def __init__(self, model):
        self.model = model

    def __enter__(self):
        if hasattr(self.model, "net"):
            self.model.net.explain_mode = True
        elif hasattr(self.model, "nets"):
            self.model.explain_mode = True
            for i in range(len(self.model.nets)):
                self.model.nets[i].explain_mode = True

    def __exit__(self, exc_type, exc_val, exc_tb):
        if hasattr(self.model, "net"):
            self.model.net.explain_mode = False
        elif hasattr(self.model, "nets"):
            self.model.explain_mode = False
            for i in range(len(self.model.nets)):
                self.model.nets[i].explain_mode = False


@task_wrapper
def execute_task(task: Task, cfg: DictConfig):
    if cfg.get("seed"):
        L.seed_everything(cfg.seed, workers=True)

    print(hydra.core.hydra_config.HydraConfig.get().runtime.choices)

    essential_objects = instantiate_essentials(cfg, log)
    object_dict = {
        "cfg": cfg,
        **essential_objects,
    }

    logger = essential_objects["logger"]
    trainer = essential_objects["trainer"]

    if logger is not None:
        log.info("Logging hyperparameters!")
        log_hyperparameters(object_dict)

    log.info("Starting <{task}>!")

    match task:
        case Task.Train:
            _train(cfg)
        case Task.Eval:
            _evaluate(cfg)
        case Task.Explain:
            _explain(cfg)
        case Task.CrossValidate:
            _cross_validate(cfg)

    metric_dict = trainer.callback_metrics

    return metric_dict, object_dict


def _train(cfg: DictConfig) -> Tuple[dict[str, Any], dict[str, Any]]:
    essential_objects = instantiate_essentials(cfg, log)

    trainer = essential_objects["trainer"]
    model = essential_objects["model"]
    datamodule = essential_objects["datamodule"]

    trainer.fit(model=model, datamodule=datamodule, ckpt_path=cfg.ckpt_path)
    train_metrics = trainer.callback_metrics

    if cfg.get("test"):
        log.info("Starting testing!")
        ckpt_path = trainer.checkpoint_callback.best_model_path

        if ckpt_path in ("", None):
            log.warning("Best ckpt not found! Using current weights for testing...")
            ckpt_path = None

        trainer.test(model=model, datamodule=datamodule, ckpt_path=ckpt_path)
        log.info(f"Best ckpt path: {ckpt_path}")

    test_metrics = trainer.callback_metrics
    metric_dict = {**train_metrics, **test_metrics}

    return metric_dict


def _evaluate(cfg: DictConfig) -> Tuple[dict[str, Any], dict[str, Any]]:
    assert cfg.ckpt_path

    essential_objects = instantiate_essentials(cfg, log)
    trainer = essential_objects["trainer"]
    model = essential_objects["model"]
    datamodule = essential_objects["datamodule"]

    return trainer.test(model=model, datamodule=datamodule, ckpt_path=cfg.ckpt_path)


def _explain(cfg: DictConfig):
    essential_objects = instantiate_essentials(cfg, log)
    model = essential_objects["model"].to(device="cpu")
    datamodule = essential_objects["datamodule"]

    X = []
    for dataloader in [datamodule.train_dataloader(), datamodule.val_dataloader()]:
        for batch in dataloader:
            x, _ = batch
            X.append(x)
    X = torch.cat(X)

    attrs_dict = {}
    with explain_mode(model):
        for algorithm_name, algorithm_cfg in cfg.xai.items():
            log.info(f"Running <{algorithm_cfg.algorithm._target_}>...")

            try:
                algorithm = hydra.utils.instantiate(algorithm_cfg.algorithm, model)
            except BaseException:
                algorithm = hydra.utils.instantiate(algorithm_cfg.algorithm, model, X)

            attrs = attribute(algorithm, X)
            attrs_dict[algorithm_name] = attrs

    attr_values = [attr.values.squeeze() for attr in attrs_dict.values()]
    for i in range(len(attr_values)):
        if isinstance(attr_values[i], torch.Tensor):
            attr_values[i] = attr_values[i].numpy()

    attr_values = np.hstack(attr_values)
    df = pl.from_numpy(attr_values)
    df.columns = list(
        itertools.chain(
            *[[f"{k}_{i + 1}" for i in range(X.shape[1])] for k in attrs_dict]
        )
    )

    for j in reversed(range(X.shape[1])):
        c = pl.Series(f"feature_{j + 1}", X[:, j].numpy())
        df = df.insert_column(0, c)

    save_path = Path(get_working_dir(), cfg.save_fname)
    df.write_csv(save_path)


def _cross_validate(cfg: DictConfig) -> Tuple[dict[str, Any], dict[str, Any]]:
    # assert cfg.ckpt_path

    essential_objects = instantiate_essentials(cfg, log)
    trainer = essential_objects["trainer"]
    model = essential_objects["model"]
    datamodule = essential_objects["datamodule"]

    assert isinstance(
        datamodule, LitKFoldDataModule
    ), "Datamodule needs to conform to a KFold datamodule"

    log.info("Starting Cross Validation!")

    N = datamodule.hparams.num_splits
    test_metrics = []

    for k in range(N):
        datamodule.setup(k=k, stage="test")
        test_output = trainer.test(
            model=model,
            dataloaders=[datamodule.test_dataloader()],
            ckpt_path=cfg.ckpt_path,
        )[0]
        test_output = {k: [v] for k, v in test_output.items()}
        test_metrics.append(pl.from_dict(test_output))

    metric_dict = (sum(test_metrics) / N).to_dict()
    metric_dict = {k: v[0] for k, v in metric_dict.items()}

    return metric_dict


# @task_wrapper
# def train(cfg: DictConfig) -> Tuple[dict[str, Any], dict[str, Any]]:
#     if cfg.get("seed"):
#         L.seed_everything(cfg.seed, workers=True)

#     essential_objects = instantiate_essentials(cfg, log)
#     object_dict = {
#         "cfg": cfg,
#         **essential_objects,
#     }

#     logger = essential_objects["logger"]
#     trainer = essential_objects["trainer"]
#     model = essential_objects["model"]
#     datamodule = essential_objects["datamodule"]

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
#         if ckpt_path in ("", None):
#             log.warning("Best ckpt not found! Using current weights for testing...")
#             ckpt_path = None
#         trainer.test(model=model, datamodule=datamodule, ckpt_path=ckpt_path)
#         log.info(f"Best ckpt path: {ckpt_path}")

#     test_metrics = trainer.callback_metrics
#     metric_dict = {**train_metrics, **test_metrics}

#     return metric_dict, object_dict
