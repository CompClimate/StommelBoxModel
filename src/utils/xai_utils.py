# import os.path as osp
# from enum import Enum

# import captum
# import captum.attr
# import matplotlib.pyplot as plt
# import shap
# import torch
# from captum.attr._utils.visualization import visualize_timeseries_attr
# from shap.plots import colors
# from utils.plot_utils import heatmap


# class AttributionMethod(Enum):
#     shap_heatmap = 1
#     captum_heatmap = 2


# class explain_mode:
#     def __init__(self, model):
#         self.model = model

#     def __enter__(self):
#         if hasattr(self.model, "net"):
#             self.model.net.explain_mode = True
#         elif hasattr(self.model, "nets"):
#             self.model.explain_mode = True
#             for i in range(len(self.model.nets)):
#                 self.model.nets[i].explain_mode = True

#     def __exit__(self, exc_type, exc_val, exc_tb):
#         if hasattr(self.model, "net"):
#             self.model.net.explain_mode = False
#         elif hasattr(self.model, "nets"):
#             self.model.explain_mode = False
#             for i in range(len(self.model.nets)):
#                 self.model.nets[i].explain_mode = False


# def explain(model, X, n_features):
#     """Computes Shapley values for a model and given input using integrated gradients.

#     It is assumed that the input is in the form of a time series with
#     input series length `n_features`.
#     """
#     features = [rf"\(\tau - {i}\)" for i in reversed(range(1, n_features + 1))]

#     e = shap.GradientExplainer(model, X)

#     e.features = features
#     shap_values = e(X)

#     return shap_values


# def explain_captum(algorithm, X, feature_names, num_samples=20, **kwargs):
#     attrs_ls = [
#         algorithm.attribute(X, **kwargs).cpu().detach() for _ in range(num_samples)
#     ]
#     attrs = torch.stack(attrs_ls, dim=-1)
#     # Average attribution over num_samples runs
#     attrs_mean = attrs.mean(dim=-1)
#     # Average standard deviation across all features
#     attrs_std = attrs.std(dim=-1).mean(dim=-1)

#     if len(attrs_mean.size()) == 1:
#         attrs_mean = attrs_mean.unsqueeze(1)

#     return (
#         shap.Explanation(values=attrs_mean, data=X, feature_names=feature_names),
#         attrs_std,
#     )


# def save_fig(fig, save_path, name, ext):
#     fig.savefig(osp.join(save_path, f"{name}.{ext}"))


# def attribute(pl_model, algorithm, X, feature_names, ylabel, method="shap_heatmap"):
#     if hasattr(algorithm, "shap_values"):
#         attrs = algorithm.shap_values(X, check_additivity=False)
#         attrs = shap.Explanation(values=attrs, data=X, feature_names=feature_names)
#     else:
#         with explain_mode(pl_model):
#             attrs, attrs_std = explain_captum(
#                 algorithm,
#                 X,
#                 feature_names,
#             )

#     fig, ax = plt.subplots()

#     method = AttributionMethod[method]
#     if method == AttributionMethod.captum_heatmap:
#         fig, axes = visualize_timeseries_attr(
#             attrs.values,
#             X.numpy(),
#             method="overlay_individual",
#             channel_labels=[ylabel] * attrs.values.shape[1],
#             cmap=colors.red_white_blue,
#             show_colorbar=True,
#             fig_size=(6, 12),
#         )
#         for ax, feature_name in zip(axes, feature_names):
#             ax.set_title(feature_name)
#     elif method == AttributionMethod.shap_heatmap:
#         fig, ax = heatmap(
#             attrs,
#             fig=fig,
#             ax=ax,
#             ylabel=ylabel,
#             show=False,
#         )
#     else:
#         raise ValueError("Invalid attribution visualization method")

#     fig.tight_layout()

#     return fig


# def plot_attributions(
#     model,
#     X,
#     algorithm,
#     feature_names,
#     ylabel,
#     save_path,
#     plot_ext,
# ):
#     explain_fig = attribute(model, algorithm, X, feature_names, ylabel)
#     save_fig(explain_fig, save_path, algorithm.__class__.__name__, plot_ext)


import shap
import torch


def attribute(algorithm, X, **kwargs):
    if hasattr(algorithm, "shap_values"):
        attrs = algorithm.shap_values(X, check_additivity=False)
        attrs = shap.Explanation(values=attrs, data=X)
    else:
        attrs = explain_captum(
            algorithm,
            X,
            **kwargs,
        )
    return attrs


def explain_captum(algorithm, X, num_samples=20, **kwargs):
    attrs_ls = [
        algorithm.attribute(X, **kwargs).cpu().detach() for _ in range(num_samples)
    ]
    attrs = torch.stack(attrs_ls, dim=-1)
    # Average attribution over num_samples runs
    attrs_mean = attrs.mean(dim=-1)
    # Average standard deviation across all features
    # attrs_std = attrs.std(dim=-1).mean(dim=-1)

    if len(attrs_mean.size()) == 1:
        attrs_mean = attrs_mean.unsqueeze(1)

    return shap.Explanation(values=attrs_mean, data=X)
