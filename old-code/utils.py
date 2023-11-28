import shap
from shap import Explanation
from shap.utils import OpChain
from shap.plots import colors
from shap.plots._utils import convert_ordering
import numpy as np
import matplotlib.pyplot as pl
from tqdm import tqdm
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
import captum
import captum.attr

import os.path as osp


class explain_mode:
    def __init__(self, model):
        self.model = model

    def __enter__(self):
        self.model.explain_mode = True

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.model.explain_mode = False


def setup_plt():
    plt.rcParams.update(
        {
            "text.usetex": True,
            "text.latex.preamble": r"""
    \usepackage{amsmath}
    \usepackage{amssymb}
    """,
        }
    )


def combine_forcing(time_0, time_max, ls_F_type, ls_F_length, step=150):
    area = 2500000000000000
    S0 = 35.0
    year = 365 * 24 * 3600

    def Fs_constant(time, time_max, flux=2):
        return flux * area * S0 / year

    def Fs_linear(time, time_max, FW_min=-0.1, FW_max=5):
        # Linear interpolation between minimum F and maximum F
        flux = FW_min + (FW_max - FW_min) * time / time_max
        return flux * area * S0 / year

    def Fs_sinusoidal(time, time_max, FW_min=-0.1, FW_max=5):
        # Sinusoidal interpolation between minimum F and maximum F
        half_range = (FW_max - FW_min) / 2
        flux = FW_min + half_range + np.sin(13 * time / time_max) * half_range
        return flux * area * S0 / year

    assert len(ls_F_type) == len(ls_F_length)
    assert time_max - time_0 == sum(ls_F_length)

    combined = []

    last_t = time_0
    last_F = None
    for i, (F_type, F_length) in enumerate(zip(ls_F_type, ls_F_length)):
        if F_type == "c":
            func = Fs_constant
        elif F_type == "l":
            func = Fs_linear
        elif F_type == "s":
            func = Fs_sinusoidal

        diff = None
        for k in tqdm(range(0, F_length, step)):
            F = func(last_t + k, time_max)

            if i > 0:
                if k == 0:
                    diff = last_F - F
                F += diff

            combined.append(F)

        last_F = combined[-1]
        last_t += F_length

    return combined


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


def explain_captum(model, attr_algorithm_cls, X, feature_names, **kwargs):
    attr_alg = attr_algorithm_cls(model)
    attrs = attr_alg.attribute(X, **kwargs).cpu().detach()
    if len(attrs.size()) == 1:
        attrs = attrs.unsqueeze(1)
    return shap.Explanation(values=attrs, data=X, feature_names=feature_names)


def heatmap(
    attr_values,
    fig=None,
    ax=None,
    instance_order=Explanation.hclust(),
    feature_values=Explanation.abs.mean(0),
    feature_order=None,
    max_display=10,
    cmap=colors.red_white_blue,
    show=True,
    plot_width=8,
    xlabel="\(t\)",
    ylabel="Attribution Value",
):
    # sort the SHAP values matrix by rows and columns
    values = attr_values.values
    if issubclass(type(feature_values), OpChain):
        feature_values = feature_values.apply(Explanation(values))
    if issubclass(type(feature_values), Explanation):
        feature_values = feature_values.values
    if feature_order is None:
        # feature_order = np.argsort(-feature_values)
        feature_order = np.arange(len(feature_values))
        feature_order = feature_order[::-1].copy()
    elif issubclass(type(feature_order), OpChain):
        feature_order = feature_order.apply(Explanation(values))
    elif not hasattr(feature_order, "__len__"):
        raise Exception("Unsupported feature_order: %s!" % str(feature_order))
    instance_order = convert_ordering(instance_order, attr_values)

    feature_names = np.array(attr_values.feature_names)[feature_order]
    values = attr_values.values[instance_order][:, feature_order]
    feature_values = feature_values[feature_order]

    # if we have more features than `max_display`, then group all the excess features
    # into a single feature
    if values.shape[1] > max_display:
        new_values = np.zeros((values.shape[0], max_display))
        new_values[:, :-1] = values[:, : max_display - 1]
        new_values[:, -1] = values[:, max_display - 1 :].sum(1)
        new_feature_values = np.zeros(max_display)
        new_feature_values[:-1] = feature_values[: max_display - 1]
        new_feature_values[-1] = feature_values[max_display - 1 :].sum()
        feature_names = [
            *feature_names[: max_display - 1],
            f"Sum of {values.shape[1] - max_display + 1} other features",
        ]
        values = new_values
        feature_values = new_feature_values

    # define the plot size based on how many features we are plotting
    row_height = 0.5
    f = fig if fig is not None else pl.gcf()
    f.set_size_inches(plot_width, values.shape[1] * row_height + 2.5)
    ax = ax if ax is not None else pl.gca()

    # plot the matrix of SHAP values as a heat map
    vmin, vmax = np.nanpercentile(values.flatten(), [1, 99])
    ax.imshow(
        values.T,
        aspect=0.7 * values.shape[0] / values.shape[1],
        interpolation="nearest",
        vmin=min(vmin, -vmax),
        vmax=max(-vmin, vmax),
        cmap=cmap,
    )

    # adjust the axes ticks and spines for the heat map + f(x) line chart
    ax.xaxis.set_ticks_position("bottom")
    ax.yaxis.set_ticks_position("left")
    ax.spines[["left", "right"]].set_visible(True)
    ax.spines[["left", "right"]].set_bounds(values.shape[1] - row_height, -row_height)
    ax.spines[["top", "bottom"]].set_visible(False)
    ax.tick_params(axis="both", direction="out")
    ax.set_ylim(values.shape[1] - row_height, -3)

    heatmap_yticks_pos = np.arange(values.shape[1])
    heatmap_yticks_labels = feature_names
    ax.yaxis.set_ticks(
        [-1.5, *heatmap_yticks_pos],
        [r"\(\small{\mathbb{E}[\text{Attr}]}\)", *heatmap_yticks_labels],
        fontsize=13,
    )
    # remove the y-tick line for the f(x) label
    ax.yaxis.get_ticklines()[0].set_visible(False)

    ax.set_xlim(-0.5, values.shape[0] - 0.5)
    ax.set_xlabel(xlabel)

    # plot the f(x) line chart above the heat map
    ax.axhline(-1.5, color="#aaaaaa", linestyle="--", linewidth=0.5)
    fx = values.T.mean(0)
    ax.plot(
        -fx / np.abs(fx).max() - 1.5,
        color="#000000",
        linewidth=1,
    )

    # plot the bar plot on the right spine of the heat map
    bar_container = ax.barh(
        heatmap_yticks_pos,
        (feature_values / np.abs(feature_values).max()) * values.shape[0] / 20,
        height=0.7,
        align="center",
        color="#000000",
        left=values.shape[0] * 1.0 - 0.5,
        # color=[colors.red_rgb if shap_values[feature_inds[i]] > 0 else colors.blue_rgb for i in range(len(y_pos))]
    )
    for b in bar_container:
        b.set_clip_on(False)

    # draw the color bar
    import matplotlib.cm as cm

    m = cm.ScalarMappable(cmap=cmap)
    m.set_array([min(vmin, -vmax), max(-vmin, vmax)])
    cb = pl.colorbar(
        m,
        ticks=[min(vmin, -vmax), max(-vmin, vmax)],
        ax=ax,
        aspect=80,
        fraction=0.01,
        pad=0.10,  # padding between the cb and the main axes
    )
    cb.set_label(ylabel, size=12, labelpad=-10)
    cb.ax.tick_params(labelsize=11, length=0)
    cb.set_alpha(1)
    cb.outline.set_visible(False)
    # bbox = cb.ax.get_window_extent().transformed(pl.gcf().dpi_scale_trans.inverted())
    # cb.ax.set_aspect((bbox.height - 0.9) * 15)
    # cb.draw_all()

    if show:
        pl.show()


def save_fig(fig, save_path, name, ext):
    fig.savefig(osp.join(save_path, f"{name}.{ext}"))


def sliding_windows(data, seq_length):
    """
    Transforms a 1d time series into sliding windows of length `seq_length`.
    """
    x = []
    y = []

    for i in range(len(data) - seq_length - 1):
        _x = data[i : (i + seq_length)]
        _y = data[i + seq_length]
        x.append(_x)
        y.append(_y)

    return np.array(x), np.array(y)


def prepare_data(train, test, seq_len, scale=False):
    """From a train/test split, optionally scales the data and returns sliding windows."""
    if scale:
        sc = MinMaxScaler()
        train = sc.fit_transform(train)
        test = sc.fit_transform(test)

    X_train, y_train = sliding_windows(train, seq_len)
    X_test, y_test = sliding_windows(test, seq_len)

    X_train, X_test = X_train.astype(np.float32), X_test.astype(np.float32)
    y_train, y_test = y_train.astype(np.float32), y_test.astype(np.float32)

    return (X_train, y_train), (X_test, y_test)


def get_raw_data(
    y,
    F,
    DeltaS,
    DeltaT,
    input_features,
    autoregressive=False,
    window_size=None,
):
    if autoregressive:
        X, y = sliding_windows(y, window_size)
    else:
        feats = {
            "F": F,
            "DeltaS": DeltaS,
            "DeltaT": DeltaT,
        }

        X = np.hstack([feats[name].reshape(-1, 1) for name in input_features])

    X, y = X.astype(np.float32), y.astype(np.float32)
    return X, y


def attribute(pl_model, alg_cls, X, feature_names, ylabel):
    with explain_mode(pl_model.model):
        attrs = explain_captum(
            pl_model.model,
            alg_cls,
            X,
            feature_names,
        )

    fig, ax = plt.subplots()
    heatmap(attrs, fig=fig, ax=ax, ylabel=ylabel, show=False)

    return fig


def plot_attributions(cfg, model_cfg, model, X, feature_names):
    explain_ylabel = model_cfg["explain_ylabel"]
    alg_cls = eval(cfg["attr_algorithm"])
    if cfg["autoregressive"] and feature_names is None:
        feature_names = [
            f"\(t - {i}\)" for i in reversed(range(1, model_cfg["input_dim"] + 1))
        ]
    explain_fig = attribute(model, alg_cls, X, feature_names, explain_ylabel)
    save_fig(explain_fig, cfg["save_path"], alg_cls.__name__, cfg["plot_ext"])


def set_input_dim(cfg, input_features, window_size):
    cfg["input_dim"] = len(input_features or ["dummy"] * window_size)
