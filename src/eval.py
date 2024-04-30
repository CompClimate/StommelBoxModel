import hydra
import rootutils
from omegaconf import DictConfig

rootutils.setup_root(__file__, indicator=".project-root", pythonpath=True)

from utils import (
    RankedLogger,
    Task,
    execute_task,
    extras,
    register_resolvers,
)

register_resolvers()

log = RankedLogger(__name__, rank_zero_only=True)


@hydra.main(version_base="1.3", config_path="../configs", config_name="eval.yaml")
def main(cfg: DictConfig) -> None:
    extras(cfg)
    execute_task(Task.Eval, cfg)


if __name__ == "__main__":
    main()
