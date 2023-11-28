from collections import defaultdict

import box_model
import captum
import captum.attr
import hydra
import torch
import torch.nn as nn
from deep_model_torch import Model, torch_model_from_cfg
from omegaconf import DictConfig, OmegaConf

from utils import get_raw_data, plot_attributions, set_input_dim, setup_plt

setup_plt()


@hydra.main(version_base=None)
def main(cfg: DictConfig):
    OmegaConf.set_struct(cfg, False)

    data_cfg = OmegaConf.load(cfg["data_config"])
    box_model_cfg = OmegaConf.load(cfg["box_model_config"])
    model_cfg = OmegaConf.load(cfg["model_config"])

    OmegaConf.set_struct(box_model_cfg, False)
    OmegaConf.set_struct(data_cfg, False)
    OmegaConf.set_struct(model_cfg, False)

    box_model_cfg = defaultdict(lambda: None, box_model_cfg)
    data_cfg = defaultdict(lambda: None, data_cfg)
    model_cfg = defaultdict(lambda: None, model_cfg)

    time_max = data_cfg["years"] * box_model.YEAR
    feature_names = data_cfg["feature_names"]
    forcing = data_cfg.get("forcing", "sinusoidal")
    forcing_kwargs = data_cfg.get("forcing_kwargs", dict())
    input_features = data_cfg["input_features"]
    autoregressive = cfg["autoregressive"]
    window_size = data_cfg.get("window_size")
    device = model_cfg["device"]

    model = box_model.BoxModel(
        **box_model_cfg,
    )

    _, _, F, y, DeltaS, DeltaT = box_model.get_time_series(
        model,
        time_max,
        forcing=forcing,
        forcing_kwargs=forcing_kwargs,
    )
    X, y = get_raw_data(
        y,
        F,
        DeltaS,
        DeltaT,
        input_features,
        autoregressive,
        window_size,
    )
    X, y = torch.from_numpy(X), torch.from_numpy(y)

    set_input_dim(model_cfg, input_features, window_size)
    torch_model = torch_model_from_cfg(model_cfg)

    model = Model.load_from_checkpoint(
        cfg["checkpoint"],
        torch_model=torch_model,
        loss_fun=nn.MSELoss(),
        lr=model_cfg["lr"],
    )
    model.to(model_cfg["device"])
    model.eval()

    plot_attributions(cfg, model_cfg, model, X.to(model_cfg["device"]), feature_names)


if __name__ == "__main__":
    main()
