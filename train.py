"""Entry point for training/explanation/evaluation."""

import hydra
import matplotlib.pyplot as plt
import numpy as np
from omegaconf import DictConfig, OmegaConf
from sklearn.model_selection import train_test_split
import torch
from collections import defaultdict
import captum
import captum.attr
import os

import box_model
import deep_model
from utils import save_fig, setup_plt, get_raw_data, plot_attributions, set_input_dim

setup_plt()


def train(cfg, box_model_cfg, data_cfg, model_cfg):
    plot_ext = cfg["plot_ext"]
    input_features = data_cfg["input_features"]
    feature_names = data_cfg["feature_names"]
    time_max = data_cfg["years"] * box_model.YEAR
    forcing = data_cfg.get("forcing", "sinusoidal")
    forcing_kwargs = data_cfg.get("forcing_kwargs", dict())
    plot_results = model_cfg["plot_results"]
    save_path = cfg["save_path"]
    test_size = data_cfg["test_size"]
    compute_bias = model_cfg.get("compute_bias")
    autoregressive = cfg["autoregressive"]
    window_size = data_cfg.get("window_size")

    os.makedirs(save_path, exist_ok=True)

    model = box_model.BoxModel(
        **box_model_cfg,
    )

    Time_DS, Time_DT, F, y, DeltaS, DeltaT = box_model.get_time_series(
        model,
        time_max,
        forcing=forcing,
        forcing_kwargs=forcing_kwargs,
    )

    if plot_results:
        data_fig = box_model.plot_time_series(Time_DS, Time_DT, F, y, DeltaS, DeltaT)
        save_fig(data_fig, save_path, "data", plot_ext)

    X, y = get_raw_data(
        y,
        F,
        DeltaS,
        DeltaT,
        input_features,
        autoregressive,
        window_size,
    )
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=test_size,
        shuffle=False,
    )
    X = torch.from_numpy(X).to(model_cfg["device"])
    X_train = torch.from_numpy(X_train).to(model_cfg["device"])
    y_train = torch.from_numpy(y_train).to(model_cfg["device"])
    X_test = torch.from_numpy(X_test).to(model_cfg["device"])
    y_test = torch.from_numpy(y_test).to(model_cfg["device"])

    set_input_dim(model_cfg, input_features, window_size)
    pl_model = deep_model.train(
        X_train,
        y_train,
        X_test,
        y_test,
        model_cfg,
    )

    pl_model = pl_model.to(model_cfg["device"])
    pl_model.eval()

    # Generate bias plot
    if compute_bias:
        fig_bias = deep_model.compute_bias(pl_model, X_train, y_train, X_test, y_test)
        save_fig(fig_bias, save_path, "bias", plot_ext)

    # Generate ground truth -- prediction plot
    if plot_results:
        fig_gt_pred = deep_model.plot_gt_pred(
            pl_model.cpu(), X_train.cpu(), y_train.cpu(), X_test.cpu(), y_test.cpu(), cfg
        )
        save_fig(fig_gt_pred, save_path, "pred-gt", plot_ext)

    # Generate attribution plot
    if model_cfg["attr_alg"] is not None:
        plot_attributions(cfg, model_cfg, pl_model.to(model_cfg["device"]), X, feature_names)


@hydra.main(version_base=None)
def main(cfg: DictConfig):
    OmegaConf.set_struct(cfg, False)
    cfg = defaultdict(lambda: None, cfg)

    data_cfg = OmegaConf.load(cfg["data_config"])
    box_model_cfg = OmegaConf.load(cfg["box_model_config"])
    model_cfg = OmegaConf.load(cfg["model_config"])

    OmegaConf.set_struct(box_model_cfg, False)
    OmegaConf.set_struct(data_cfg, False)
    OmegaConf.set_struct(model_cfg, False)

    box_model_cfg = defaultdict(lambda: None, box_model_cfg)
    data_cfg = defaultdict(lambda: None, data_cfg)
    model_cfg = defaultdict(lambda: None, model_cfg)

    train(cfg, box_model_cfg, data_cfg, model_cfg)


if __name__ == "__main__":
    main()
