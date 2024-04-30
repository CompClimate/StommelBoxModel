#     if plot_cfg := cfg.get("plotting"):
#         if (plot_data := plot_cfg.get("data")) and plot_data:
#             fig_forcings, fig_variables, fig_amoc = plot_time_series(
#                 datamodule.series_dict
#             )
#             save_fig(fig_forcings, working_dir, "data_forcings", "pdf")
#             save_fig(fig_variables, working_dir, "data_variables", "pdf")
#             save_fig(fig_amoc, working_dir, "data_amoc", "pdf")

#         # Generate ground truth -- prediction plot
#         if (
#             groundtruth_prediction := plot_cfg.get("groundtruth_prediction")
#         ) and groundtruth_prediction.plot:
#             log.info("Plotting [Ground Truth - Prediction] plot...")
#             fig_gt_pred = plot_gt_pred(
#                 model.cpu(),
#                 X_train,
#                 y_train,
#                 X_test,
#                 y_test,
#                 show_change_points=plot_cfg.get("show_change_points"),
#             )
#             save_fig(
#                 fig_gt_pred,
#                 working_dir,
#                 "groundtruth-prediction",
#                 "pdf",
#             )

#     # Generate attribution plot
#     if explain_cfg := cfg.get("explainability"):
#         log.info("Running explainability...")
#         data_cfg = cfg.data

#         autoregressive = data_cfg.autoregressive
#         feature_names = (
#             [rf"\(\tau - {i}\)" for i in reversed(range(1, input_dim + 1))]
#             if autoregressive
#             else list(datamodule.series_dict["latex"]["variables"].values())
#             + list(datamodule.series_dict["latex"]["forcings"].values())
#         )
#         data = torch.from_numpy(datamodule.X)

#         for algorithm_cfg in explain_cfg.values():
#             log.info(f"Running <{algorithm_cfg.algorithm._target_}>...")

#             try:
#                 algorithm = hydra.utils.instantiate(algorithm_cfg.algorithm, model)
#             except BaseException:
#                 algorithm = hydra.utils.instantiate(
#                     algorithm_cfg.algorithm, model, torch.from_numpy(datamodule.X_train)
#                 )

#             ylabel = algorithm_cfg.get("ylabel")
#             plot_attributions(
#                 model,
#                 data,
#                 algorithm,
#                 feature_names,
#                 ylabel,
#                 working_dir,
#                 "pdf",
#             )


from typing import Optional

import hydra
import rootutils
from omegaconf import DictConfig

rootutils.setup_root(__file__, indicator=".project-root", pythonpath=True)

from utils import (
    RankedLogger,
    Task,
    execute_task,
    extras,
    get_metric_value,
    register_resolvers,
)

register_resolvers()

log = RankedLogger(__name__, rank_zero_only=True)


@hydra.main(version_base="1.3", config_path="../configs", config_name="train.yaml")
def main(cfg: DictConfig) -> Optional[float]:
    extras(cfg)

    metric_dict, _ = execute_task(Task.Train, cfg)
    metric_value = get_metric_value(
        metric_dict=metric_dict, metric_name=cfg.get("optimized_metric")
    )

    return metric_value


if __name__ == "__main__":
    main()
