import os.path as osp
import sys

sys.path.append(osp.abspath("."))
sys.path.append(osp.join(osp.abspath("."), "src"))

import dill
import hydra
import rootutils
from omegaconf import DictConfig

from src.data.components import box_model as box
from src.utils import RankedLogger

rootutils.setup_root(__file__, indicator=".project-root", pythonpath=True)

log = RankedLogger(__name__, rank_zero_only=True)


def generate_data(
    box_model,
    s_forcing,
    t_forcing,
    solution_cfg: DictConfig,
    init_time_series_cfg: DictConfig,
    density: DictConfig,
):
    series_dict = box.get_time_series(
        model=box_model,
        s_forcing=s_forcing,
        t_forcing=t_forcing,
        nonlinear_density=density.nonlinear_density,
        solution_cfg=solution_cfg,
        init_time_series_cfg=init_time_series_cfg,
    )
    return series_dict


@hydra.main(
    version_base="1.3", config_path="../configs", config_name="generate_data.yaml"
)
def main(cfg: DictConfig):
    box_model = hydra.utils.instantiate(cfg.box_model)
    s_forcing = hydra.utils.instantiate(cfg.s_forcing)
    t_forcing = cfg.get("t_forcing")

    if t_forcing is not None:
        t_forcing = hydra.utils.instantiate(cfg.t_forcing)

    series_dict = generate_data(
        box_model, s_forcing, t_forcing, cfg.solution, cfg.init_time_series, cfg.density
    )

    choices = hydra.core.hydra_config.HydraConfig.get().runtime.choices
    s_forcing_name = choices.s_forcing
    t_forcing_name = choices.get("t_forcing", "none")
    density_name = choices.density

    save_name = cfg.save_name
    if save_name is None:
        save_name = f"s_forcing={s_forcing_name},t_forcing={t_forcing_name},density={density_name}.pkl"

    save_path = osp.join(cfg.save_dir, save_name)
    save_path = osp.abspath(save_path)

    if osp.exists(save_path) and not cfg.overwrite_existing:
        log.info("Output path exists but overwrite flag not set, aborting.")
    else:
        with open(save_path, "wb") as f:
            dill.dump(series_dict, f)


if __name__ == "__main__":
    main()
