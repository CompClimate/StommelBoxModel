import os.path as osp
import sys
from enum import StrEnum
from pathlib import Path

sys.path.append(osp.abspath("."))
sys.path.append(osp.join(osp.abspath("."), "src"))

import hydra
import polars as pl
import rootutils
import torch
from omegaconf import DictConfig

from src.utils import RankedLogger, get_working_dir, register_resolvers
from src.utils.instantiators import instantiate_essentials

rootutils.setup_root(__file__, indicator=".project-root", pythonpath=True)

log = RankedLogger(__name__, rank_zero_only=True)

register_resolvers()


class MetricType(StrEnum):
    Bias = "bias"
    Prediction = "prediction"


def get_X_y(dataset, device):
    X, y = (
        [dataset[i][0] for i in range(len(dataset))],
        [dataset[i][1] for i in range(len(dataset))],
    )
    X, y = (
        torch.stack(X).to(device=device),
        torch.stack(y).to(device=device),
    )
    return X, y


@torch.no_grad()
def compute_bias(model, X, y, num_samples: int = 50):
    if hasattr(model, "net") and type(model.net).__name__ == "BNNTorch":
        preds = model(X)
        preds = [model(X) for _ in range(num_samples)]
        preds = torch.stack(preds)
        pred_mean = preds.mean(axis=0).squeeze()
        pred_std = preds.std(axis=0).squeeze()
    else:
        pred_mean, pred_std = model(X)
        pred_mean = pred_mean.squeeze()
        pred_std = pred_std.squeeze()

    bias = pred_mean - y
    return dict(bias=bias)


@torch.no_grad()
def compute_pred(model, X, y, num_samples: int = 50):
    if hasattr(model, "net") and type(model.net).__name__ == "BNNTorch":
        preds = model(X)
        preds = [model(X) for _ in range(num_samples)]
        preds = torch.stack(preds)
        pred_mean = preds.mean(axis=0).squeeze()
        pred_std = preds.std(axis=0).squeeze()
    else:
        pred_mean, pred_std = model(X)
        pred_mean = pred_mean.squeeze()
        pred_std = pred_std.squeeze()

    return dict(pred_mean=pred_mean, pred_std=pred_std, y=y)


@hydra.main(
    version_base="1.3", config_path="../configs", config_name="compute_metric.yaml"
)
def main(cfg: DictConfig):
    working_dir = get_working_dir()
    essential_objects = instantiate_essentials(cfg, log)

    model = essential_objects["model"]
    datamodule = essential_objects["datamodule"]

    model.eval()

    loaders = {
        "train": datamodule.train_dataloader(),
        "val": datamodule.val_dataloader(),
    }

    metric_type = cfg.metric_type
    metrics = {}

    for loader_name, loader in loaders.items():
        X, y = get_X_y(loader.dataset, model.device)
        metric = None

        match metric_type:
            case MetricType.Bias:
                metric = compute_bias(model, X, y)
            case MetricType.Prediction:
                metric = compute_pred(model, X, y)

        for k in metric:
            metric[k] = metric[k].cpu().numpy()
        metrics[loader_name] = metric

    for loader_name in loaders:
        metrics[loader_name] = {
            f"{loader_name}/{k}": v for k, v in metrics[loader_name].items()
        }

    metrics = {k: pl.from_dict(v) for k, v in metrics.items()}
    for k, df in metrics.items():
        output_path = Path(working_dir, f"{k}_{metric_type}.csv")
        df.write_csv(output_path)


if __name__ == "__main__":
    main()
