import logging
from typing import List

import hydra
from lightning import Callback, LightningDataModule, LightningModule, Trainer
from lightning.pytorch.loggers import Logger
from models.module import EnsembleModel, Model
from omegaconf import DictConfig

from .utils import pylogger

log = pylogger.RankedLogger(__name__, rank_zero_only=True)


def instantiate_callbacks(callbacks_cfg: DictConfig) -> List[Callback]:
    """Instantiates callbacks from config.

    :param callbacks_cfg: A DictConfig object containing callback configurations.
    :return: A list of instantiated callbacks.
    """
    callbacks: List[Callback] = []

    if not callbacks_cfg:
        log.warning("No callback configs found! Skipping..")
        return callbacks

    if not isinstance(callbacks_cfg, DictConfig):
        raise TypeError("Callbacks config must be a DictConfig!")

    for _, cb_conf in callbacks_cfg.items():
        if isinstance(cb_conf, DictConfig) and "_target_" in cb_conf:
            log.info(f"Instantiating callback <{cb_conf._target_}>")
            callbacks.append(hydra.utils.instantiate(cb_conf))

    return callbacks


def instantiate_loggers(logger_cfg: DictConfig) -> List[Logger]:
    """Instantiates loggers from config.

    :param logger_cfg: A DictConfig object containing logger configurations.
    :return: A list of instantiated loggers.
    """
    logger: List[Logger] = []

    if not logger_cfg:
        log.warning("No logger configs found! Skipping...")
        return logger

    if not isinstance(logger_cfg, DictConfig):
        raise TypeError("Logger config must be a DictConfig!")

    for _, lg_conf in logger_cfg.items():
        if isinstance(lg_conf, DictConfig) and "_target_" in lg_conf:
            log.info(f"Instantiating logger <{lg_conf._target_}>")
            logger.append(hydra.utils.instantiate(lg_conf))

    return logger


def instantiate_essentials(cfg: DictConfig, logger: logging.LoggerAdapter):
    if cfg.get("data", None):
        log.info(f"Instantiating datamodule <{cfg.data._target_}>")
        datamodule: LightningDataModule = hydra.utils.instantiate(cfg.data)
    else:
        log.info(f"Instantiating datamodule <{cfg.cv._target_}>")
        datamodule: LightningDataModule = hydra.utils.instantiate(cfg.cv)

    log.info(f"Instantiating model <{cfg.model._target_}>")
    m = hydra.utils.instantiate(cfg.model)
    # print(f"{type(m).__name__ = }")
    # exit()
    if cfg.ckpt_path:
        if type(m).__name__ == "EnsembleModel":
            model = EnsembleModel.load_from_checkpoint(
                cfg.ckpt_path, loss_fun=m.loss_fun
            )
        else:
            model = Model.load_from_checkpoint(
                cfg.ckpt_path, net=m.net, loss_fun=m.loss_fun
            )
    else:
        model = m

    # model: LightningModule = (
    #     Model.load_from_checkpoint(
    #         cfg.ckpt_path, net=m.net, loss_fun=m.loss_fun
    #     )
    #     if cfg.ckpt_path
    #     else hydra.utils.instantiate(cfg.model)
    # )

    log.info("Instantiating callbacks...")
    callbacks: List[Callback] = instantiate_callbacks(cfg.get("callbacks"))

    log.info("Instantiating loggers...")
    logger: List[Logger] = instantiate_loggers(cfg.get("logger"))

    log.info(f"Instantiating trainer <{cfg.trainer._target_}>")
    trainer: Trainer = hydra.utils.instantiate(
        cfg.trainer, callbacks=callbacks, logger=logger
    )

    objects = {
        "datamodule": datamodule,
        "model": model,
        "callbacks": callbacks,
        "logger": logger,
        "trainer": trainer,
    }
    return objects
