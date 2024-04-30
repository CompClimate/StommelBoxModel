import pickle

import hydra
import lightning as L
import rootutils
from omegaconf import DictConfig
from utils.landscape_utils import LossLandscape, RandomCoordinates
from utils.plot_utils import save_fig, setup_plt

from data.components.box_model import BoxModel
from data.components.forcing import Forcing
from data.time_series_datamodule import TimeSeriesDataModule
from src.models.module import Model
from src.utils import RankedLogger, extras, get_working_dir

rootutils.setup_root(__file__, indicator=".project-root", pythonpath=True)


setup_plt()

log = RankedLogger(__name__, rank_zero_only=True)


def run(cfg: DictConfig):
    if cfg.get("seed"):
        L.seed_everything(cfg.seed, workers=True)

    working_dir = get_working_dir()

    log.info(f"Instantiating box model <{cfg.box_model._target_}>")
    box_model: BoxModel = hydra.utils.instantiate(cfg.box_model)

    log.info(f"Instantiating box model <{cfg.forcing._target_}>")
    forcing: Forcing = hydra.utils.instantiate(cfg.forcing)

    datamodule = TimeSeriesDataModule(box_model, forcing, **cfg.data)

    log.info("Loading model from checkpoint.")
    model = Model.load_from_checkpoint(cfg.checkpoint_path)

    with open(cfg.training_path, "rb") as f:
        training_path = pickle.load(f)

    for i in range(len(training_path)):
        for j in range(len(training_path[i])):
            training_path[i][j] = training_path[i][j].cpu()

    ll = LossLandscape(model, datamodule.train_dataloader())
    coords = RandomCoordinates(training_path[0], cfg.dim)

    surface = cfg.dim == 3

    ll.compile(cfg.range, cfg.points, coords, model.loss_fun, surface)
    fig, ax = ll.plot("Loss Landscape", surface=surface)
    save_fig(fig, working_dir, "loss_landscape", "pdf")


@hydra.main(
    version_base="1.3", config_path="../configs", config_name="plot_landscape.yaml"
)
def main(cfg: DictConfig):
    extras(cfg)
    run(cfg)


if __name__ == "__main__":
    main()
