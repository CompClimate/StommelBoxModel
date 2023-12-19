import os.path as osp

import matplotlib.pyplot as plt
import numpy as np
import ruptures as rpt
import torch
from shap import Explanation
from shap.plots import colors
from shap.plots._utils import convert_ordering
from shap.utils import OpChain


def setup_plt():
    plt.rcParams.update(
        {
            "text.usetex": True,
            "text.latex.preamble": r"""
    \usepackage{amsmath}
    \usepackage{amssymb}
    \usepackage{textcomp,mathcomp}
    """,
            "font.size": 15,
            "xtick.labelsize": 10,
            "ytick.labelsize": 10,
        }
    )


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
    xlabel=r"\(t\)",
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
    f = fig if fig is not None else plt.gcf()
    f.set_size_inches(plot_width, values.shape[1] * row_height + 2.5)
    ax = ax if ax is not None else plt.gca()

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
    cb = plt.colorbar(
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
        plt.show()


def compute_bias(pl_model, X_train, y_train, X_test, y_test):
    with torch.no_grad():
        device = pl_model.device
        pl_model.cpu()

        train_pred_mean, train_pred_std = pl_model(X_train)
        test_pred_mean, test_pred_std = pl_model(X_test)

        train_pred_mean, test_pred_mean = (
            train_pred_mean.squeeze(),
            test_pred_mean.squeeze(),
        )
        train_pred_std, test_pred_std = (
            train_pred_std.squeeze(),
            test_pred_std.squeeze(),
        )

        train_bias = train_pred_mean - y_train
        test_bias = test_pred_mean - y_test

    fig, ax = plt.subplots(figsize=(10, 8))

    xs_time_train = list(range(1, len(y_train) + 1))
    xs_time_test = list(range(len(y_train), len(y_train) + len(y_test)))

    ax.plot(xs_time_train, train_bias, label="Training Set")
    ax.plot(xs_time_test, test_bias, label="Test Set")

    ax.fill_between(
        xs_time_train,
        train_bias - train_pred_std,
        train_bias + train_pred_std,
        alpha=0.3,
    )
    ax.fill_between(
        xs_time_test,
        test_bias - test_pred_std,
        test_bias + test_pred_std,
        alpha=0.3,
    )

    ax.set_xlabel(r"\(t\)")
    ax.set_ylabel("Bias")
    ax.legend()

    pl_model.to(device)

    return fig


def plot_gt_pred(pl_model, X_train, y_train, X_test, y_test, show_change_points=False):
    with torch.no_grad():
        train_pred_mean, train_pred_std = pl_model(X_train)
        train_pred_mean = train_pred_mean.view(-1)
        train_pred_std = train_pred_std.view(-1)

        test_pred_mean, test_pred_std = pl_model(X_test)
        test_pred_mean = test_pred_mean.view(-1)
        test_pred_std = test_pred_std.view(-1)

    if show_change_points:
        algo = rpt.Pelt(model="rbf").fit(y_test)
        result = algo.predict(pen=10)

    fig, ax = plt.subplots()

    xs_time_train = list(range(1, len(train_pred_mean) + 1))
    xs_time_test = list(
        range(len(train_pred_mean), len(train_pred_mean) + len(test_pred_mean))
    )

    ax.plot(
        xs_time_train + xs_time_test,
        torch.hstack((y_train, y_test)),
        label="Ground Truth",
        color="tab:blue",
    )
    ax.plot(
        xs_time_train + xs_time_test,
        torch.hstack((train_pred_mean, test_pred_mean)),
        label="Prediction",
        color="tab:orange",
    )
    ax.fill_between(
        xs_time_train + xs_time_test,
        torch.hstack(
            (train_pred_mean - train_pred_std, test_pred_mean - test_pred_std)
        ),
        torch.hstack(
            (train_pred_mean + train_pred_std, test_pred_mean + test_pred_std)
        ),
        alpha=0.3,
    )

    ax.axvline(xs_time_train[-1], ls="--")
    ax.text(
        xs_time_train[-1] - 15,
        0.97,
        "Train-Test Split",
        color="k",
        ha="right",
        va="top",
        rotation=0,
        transform=ax.get_xaxis_transform(),
    )

    # ax.plot(
    #     xs_time_train,
    #     y_train,
    #     label="Ground Truth: Training Set",
    #     color="tab:blue",
    # )
    # ax.plot(
    #     xs_time_train,
    #     train_pred_mean,
    #     label="Prediction: Training Set",
    #     color="tab:green",
    # )
    # ax.plot(xs_time_test, y_test, label="Ground Truth: Test Set", color="tab:orange")
    # ax.plot(
    #     xs_time_test,
    #     test_pred_mean,
    #     label="Prediction: Test Set",
    #     color="tab:red",
    # )
    # ax.fill_between(
    #     xs_time_train,
    #     train_pred_mean - train_pred_std,
    #     train_pred_mean + train_pred_std,
    #     alpha=0.3,
    # )
    # ax.fill_between(
    #     xs_time_test,
    #     test_pred_mean - test_pred_std,
    #     test_pred_mean + test_pred_std,
    #     alpha=0.3,
    # )

    # if show_change_points:
    # ax.vlines(result, y_test.min(), y_test.max(), ls="--")

    ax.set_xlabel(r"\(t\)")
    ax.set_ylabel(r"\(q\) (Sv)")
    ax.legend()

    return fig


def save_fig(fig, save_path, name, ext):
    fig.savefig(osp.join(save_path, f"{name}.{ext}"))
