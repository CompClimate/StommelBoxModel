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
from utils import save_fig, setup_plt

setup_plt()


def prepare_features(y, F, DeltaS, DeltaT, input_features, test_size):
    feats = {
        "F": F,
        "DeltaS": DeltaS,
        "DeltaT": DeltaT,
    }

    X = np.hstack([feats[name].reshape(-1, 1) for name in input_features])
    X, y = X.astype(np.float32), y.astype(np.float32)

    X_train_, X_test_, y_train_, y_test_ = train_test_split(
        X,
        y,
        test_size=test_size,
        shuffle=False,
    )

    X_train = torch.from_numpy(X_train_)
    y_train = torch.from_numpy(y_train_)

    X_test = torch.from_numpy(X_test_)
    y_test = torch.from_numpy(y_test_)

    return X_train, y_train, X_test, y_test


def training_task(cfg):
    plot_ext = cfg["plot_ext"]
    input_features = cfg["data"]["input_features"]
    feature_names = cfg["data"]["feature_names"]
    time_max = cfg["years"] * box_model.YEAR
    forcing = cfg["data"].get("forcing", "sinusoidal")
    forcing_kwargs = cfg["data"].get("forcing_kwargs", dict())
    plot_results = cfg["model"]["plot_results"]
    save_path = cfg["save_path"]
    test_size = cfg["data"]["test_size"]
    compute_bias = cfg["model"]["compute_bias"]

    os.makedirs(save_path, exist_ok=True)

    model = box_model.BoxModel(
        **cfg["box_model"],
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

    X_train, y_train, X_test, y_test = prepare_features(
        y, F, DeltaS, DeltaT, input_features, test_size
    )

    cfg["model"]["input_dim"] = len(input_features)
    pl_model = deep_model.train(
        X_train,
        y_train,
        X_test,
        y_test,
        cfg["model"],
    )

    pl_model.eval()

    # Generate bias plot
    if compute_bias:
        fig_bias = deep_model.compute_bias(pl_model, X_train, y_train, X_test, y_test)
        save_fig(fig_bias, save_path, "bias", plot_ext)

    # Generate ground truth -- prediction plot
    if plot_results:
        fig_gt_pred = deep_model.plot_gt_pred(
            pl_model, X_train, y_train, X_test, y_test, cfg
        )
        save_fig(fig_gt_pred, save_path, "pred-gt", plot_ext)

    # Generate attribution plot
    if (alg_cls_str := cfg["model"]["explain_after_train"]) is not None:
        explain_ylabel = cfg["model"]["explain_ylabel"]
        alg_cls = eval(alg_cls_str)
        explain_fig = deep_model.attribute(
            pl_model, alg_cls, X_test, feature_names, explain_ylabel
        )
        save_fig(explain_fig, save_path, alg_cls.__name__, plot_ext)


@hydra.main(version_base=None)
def main(cfg: DictConfig):
    OmegaConf.set_struct(cfg, False)
    cfg = defaultdict(lambda: None, cfg)

    if cfg["task"] == "training":
        training_task(cfg)


if __name__ == "__main__":
    main()
