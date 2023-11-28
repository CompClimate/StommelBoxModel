import shap
import matplotlib.pyplot as plt
from utils.plot_utils import heatmap
import os.path as osp
import captum
import captum.attr


class explain_mode:
    def __init__(self, model):
        self.model = model

    def __enter__(self):
        self.model.net.explain_mode = True

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.model.net.explain_mode = False


def explain(model, X, n_features):
    """
    Computes Shapley values for a model and given input using integrated gradients.
    It is assumed that the input is in the form of a time series with
    input series length `n_features`.
    """
    features = [f"\(t - {i}\)" for i in reversed(range(1, n_features + 1))]

    e = shap.GradientExplainer(model, X)

    e.features = features
    shap_values = e(X)

    return shap_values


def explain_captum(pl_model, attr_algorithm_cls, X, feature_names, **kwargs):
    attr_alg = attr_algorithm_cls(pl_model)
    attrs = attr_alg.attribute(X, **kwargs).cpu().detach()
    if len(attrs.size()) == 1:
        attrs = attrs.unsqueeze(1)
    return shap.Explanation(values=attrs, data=X, feature_names=feature_names)


def save_fig(fig, save_path, name, ext):
    fig.savefig(osp.join(save_path, f"{name}.{ext}"))


def attribute(pl_model, alg_cls, X, feature_names, ylabel):
    with explain_mode(pl_model):
        attrs = explain_captum(
            pl_model,
            alg_cls,
            X,
            feature_names,
        )

    fig, ax = plt.subplots()
    heatmap(attrs, fig=fig, ax=ax, ylabel=ylabel, show=False)

    return fig


def plot_attributions(
    model,
    X,
    attr_algorithm,
    feature_names,
    autoregressive,
    ylabel,
    input_dim,
    save_path,
    plot_ext,
):
    alg_cls = eval(attr_algorithm)
    if autoregressive and feature_names is None:
        feature_names = [f"\(t - {i}\)" for i in reversed(range(1, input_dim + 1))]
    explain_fig = attribute(model, alg_cls, X, feature_names, ylabel)
    save_fig(explain_fig, save_path, alg_cls.__name__, plot_ext)
