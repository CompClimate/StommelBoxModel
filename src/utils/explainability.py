import os.path as osp
from enum import Enum

import captum
import captum.attr
from captum.attr._utils.visualization import visualize_timeseries_attr
import matplotlib.pyplot as plt
import shap

from utils.plot_utils import heatmap


class AttributionMethod(Enum):
    shap_heatmap = 1
    captum_heatmap = 2


class explain_mode:
    def __init__(self, model):
        self.model = model

    def __enter__(self):
        self.model.net.explain_mode = True

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.model.net.explain_mode = False


def explain(model, X, n_features):
    """Computes Shapley values for a model and given input using integrated gradients.

    It is assumed that the input is in the form of a time series with
    input series length `n_features`.
    """
    features = [rf"\(t - {i}\)" for i in reversed(range(1, n_features + 1))]

    e = shap.GradientExplainer(model, X)

    e.features = features
    shap_values = e(X)

    return shap_values


def explain_captum(algorithm, X, feature_names, **kwargs):
    attrs = algorithm.attribute(X, **kwargs).cpu().detach()

    if len(attrs.size()) == 1:
        attrs = attrs.unsqueeze(1)

    return shap.Explanation(values=attrs, data=X, feature_names=feature_names)


def save_fig(fig, save_path, name, ext):
    fig.savefig(osp.join(save_path, f"{name}.{ext}"))


def attribute(pl_model, algorithm, X, feature_names, ylabel, method="captum_heatmap"):
    with explain_mode(pl_model):
        attrs = explain_captum(
            algorithm,
            X,
            feature_names,
        )

    fig, ax = plt.subplots()

    method = AttributionMethod[method]
    if method == AttributionMethod.captum_heatmap:
        fig, axes = visualize_timeseries_attr(
            attrs.values,
            X,
            method="overlay_individual",
            channel_labels=[ylabel] * attrs.values.shape[1],
        )
        for ax, feature_name in zip(axes, feature_names):
            ax.set_title(feature_name)
    elif method == AttributionMethod.shap_heatmap:
        heatmap(attrs, fig=fig, ax=ax, ylabel=ylabel, show=False)
    else:
        raise ValueError("Invalid attribution visualization method")

    fig.tight_layout()

    return fig


def plot_attributions(
    model,
    X,
    algorithm,
    feature_names,
    ylabel,
    save_path,
    plot_ext,
):
    explain_fig = attribute(model, algorithm, X, feature_names, ylabel)
    save_fig(explain_fig, save_path, algorithm.__class__.__name__, plot_ext)
