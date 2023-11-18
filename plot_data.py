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
    OmegaConf.set_struct(cfg, False)
    cfg = defaultdict(lambda: None, cfg)

    time_max = cfg["years"] * box_model.YEAR
    forcing = cfg["data"].get("forcing", "sinusoidal")
    forcing_kwargs = cfg["data"].get("forcing_kwargs", dict())
    save_path = cfg["save_path"]
    data_plot_name = cfg["data_plot_name"]
    data_plot_ext = cfg["plot_ext"]

    model = box_model.BoxModel(
        **cfg["box_model"],
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
