import hydra
from omegaconf import DictConfig, OmegaConf
from collections import defaultdict
import matplotlib.pyplot as plt
import os

import box_model
from utils import save_fig, setup_plt

setup_plt()


@hydra.main(version_base=None)
def main(cfg: DictConfig):
    data_cfg = OmegaConf.load(cfg["data_config"])
    box_model_cfg = OmegaConf.load(cfg["box_model_config"])

    OmegaConf.set_struct(box_model_cfg, False)
    OmegaConf.set_struct(data_cfg, False)

    box_model_cfg = defaultdict(lambda: None, box_model_cfg)
    data_cfg = defaultdict(lambda: None, data_cfg)

    time_max = data_cfg["years"] * box_model.YEAR
    forcing = data_cfg.get("forcing", "sinusoidal")
    forcing_kwargs = data_cfg.get("forcing_kwargs", dict())
    save_path = cfg["save_path"]
    data_plot_name = cfg["data_plot_name"]
    data_plot_ext = cfg["plot_ext"]

    model = box_model.BoxModel(
        **box_model_cfg,
    )

    Time_DS, Time_DT, F, y, DeltaS, DeltaT = box_model.get_time_series(
        model,
        time_max,
        forcing=forcing,
        forcing_kwargs=forcing_kwargs,
    )

    fig = box_model.plot_time_series(Time_DS, Time_DT, F, y, DeltaS, DeltaT)

    if save_path is None:
        plt.show()
    else:
        os.makedirs(save_path, exist_ok=True)
        save_fig(fig, save_path, data_plot_name, data_plot_ext)


if __name__ == "__main__":
    main()
