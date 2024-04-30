import os.path as osp

import matplotlib.cm as cm
import matplotlib.pyplot as plt
import numpy as np
import ruptures as rpt
import torch
from shap import Explanation
from shap.plots import colors
from shap.plots._utils import convert_ordering
from shap.utils import OpChain


def setup_plt():
    # font_size = 30
    # tick_labelsize = 30
    # axes_labelsize = 30
    font_size = 30
    tick_labelsize = "large"
    axes_labelsize = "large"
    # legend_fontsize = 14
    plt.rcParams.update(
        {
            "text.usetex": True,
            "text.latex.preamble": r"""
    \usepackage{amsmath}
    \usepackage{amssymb}
    \usepackage{textcomp,mathcomp}
    """,
            "font.size": font_size,
            "axes.labelsize": axes_labelsize,
            "xtick.labelsize": tick_labelsize,
            "ytick.labelsize": tick_labelsize,
            # "legend.fontsize": legend_fontsize,
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
    xlabel=r"\(\tau\)",
    ylabel="Attribution Value",
):
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

    row_height = 1.0 if (len(feature_names) not in [2, 3, 4]) else 1.5

    if fig is None or ax is None:
        fig, ax = plt.subplots()

    if len(feature_names) not in [2, 3, 4]:
        fig.set_size_inches(plot_width, values.shape[1] * row_height)
    else:
        fig.set_size_inches(plot_width, values.shape[1] * row_height + 2.5)

    vmin, vmax = np.nanpercentile(values.flatten(), [1, 99])
    ax.imshow(
        values.T,
        aspect=0.7 * values.shape[0] / values.shape[1]
        if len(feature_names) not in [2, 3, 4]
        else "auto",
        # else 0.1,
        interpolation="nearest",
        vmin=min(vmin, -vmax),
        vmax=max(-vmin, vmax),
        cmap=cmap,
    )

    # Adjust the axes ticks and spines for the heat map + f(x) line chart
    ax.xaxis.set_ticks_position("bottom")
    ax.yaxis.set_ticks_position("left")
    ax.spines[["left", "right"]].set_visible(True)
    ax.spines[["left", "right"]].set_bounds(values.shape[1] - row_height, -row_height)
    ax.spines[["top", "bottom"]].set_visible(False)
    ax.tick_params(axis="both", direction="out")

    # if len(feature_names) != 2:
    # ax.set_ylim(values.shape[1] - row_height + 0.5, -3.5)
    # else:
    # ax.set_ylim(values.shape[1] - row_height, -3)
    # ax.set_ylim(-3, -3.5)

    heatmap_yticks_labels = feature_names
    heatmap_yticks_pos = np.arange(values.shape[1])
    # if len(heatmap_yticks_labels) != 2:
    # heatmap_yticks_pos -= 0.5
    # else:
    # heatmap_yticks_pos += 1

    if len(heatmap_yticks_labels) == 10:
        ax.tick_params(axis="y", labelrotation=90)
        heatmap_yticks_labels = [""] * 10
        heatmap_yticks_labels[0] = r"\(\tau - 10, \dots, \tau - 1\)"

    shift = -1.5
    ax.yaxis.set_ticks(
        [shift, *heatmap_yticks_pos],
        [
            r"\(\mu\)",
            *heatmap_yticks_labels,
        ],
    )
    ax.yaxis.get_ticklines()[0].set_visible(False)

    ax.set_xlim(-0.5, values.shape[0] - 0.5)
    ax.set_xlabel(xlabel)

    ax.axhline(shift, color="#aaaaaa", linestyle="--", linewidth=0.5)
    fx = values.T.mean(0)
    fx_adjusted = -fx / np.abs(fx).max() + shift
    ax.plot(
        fx_adjusted,
        color="#000000",
        linewidth=1,
    )

    # bar_container = ax.barh(
    #     heatmap_yticks_pos,
    #     (feature_values / np.abs(feature_values).max()) * values.shape[0] / 20,
    #     height=0.7,
    #     align="center",
    #     color="#000000",
    #     left=values.shape[0] * 1.0 - 0.5,
    #     # color=[colors.red_rgb if shap_values[feature_inds[i]] > 0 else colors.blue_rgb for i in range(len(y_pos))]
    # )
    # for b in bar_container:
    #     b.set_clip_on(False)

    m = cm.ScalarMappable(cmap=cmap)
    m.set_array([min(vmin, -vmax), max(-vmin, vmax)])
    cb = plt.colorbar(
        m,
        ticks=[min(vmin, -vmax), max(-vmin, vmax)],
        ax=ax,
        aspect=80,
        fraction=0.01,
        pad=0.10,
    )
    cb.set_label(
        ylabel,
        labelpad=-20,
    )
    cb.ax.tick_params(length=0)
    cb.set_alpha(1)
    cb.outline.set_visible(False)

    fig.tight_layout()

    if show:
        plt.show()

    return fig, ax


def compute_bias(pl_model, X_train, y_train, X_test, y_test, num_samples=50):
    with torch.no_grad():
        device = pl_model.device
        pl_model.cpu()

        if hasattr(pl_model, "net") and type(pl_model.net).__name__ == "BNNTorch":
            preds_train = [pl_model(X_train) for _ in range(num_samples)]
            preds_train = torch.stack(preds_train)
            train_pred_mean = preds_train.mean(axis=0).squeeze()
            train_pred_std = preds_train.std(axis=0).squeeze()

            preds_test = [pl_model(X_test) for _ in range(num_samples)]
            preds_test = torch.stack(preds_test)
            test_pred_mean = preds_test.mean(axis=0).squeeze()
            test_pred_std = preds_test.std(axis=0).squeeze()
        else:
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

    fig, ax = plt.subplots()

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

    ax.set_xlabel(r"\(\tau\)")
    ax.xaxis.label.set_size(33)
    ax.set_ylabel("Bias")
    ax.yaxis.label.set_size(33)

    fig.tight_layout()

    pl_model.to(device)

    return fig


def plot_gt_pred(
    pl_model,
    X_train,
    y_train,
    X_test,
    y_test,
    num_samples=50,
    std_multiplier=2,
    show_change_points=False,
):
    with torch.no_grad():
        if hasattr(pl_model, "net") and type(pl_model.net).__name__ == "BNNTorch":
            preds_train = [pl_model(X_train) for _ in range(num_samples)]
            preds_train = torch.stack(preds_train)
            train_pred_mean = preds_train.mean(axis=0).squeeze()
            train_pred_std = preds_train.std(axis=0).squeeze()

            preds_test = [pl_model(X_test) for _ in range(num_samples)]
            preds_test = torch.stack(preds_test)
            test_pred_mean = preds_test.mean(axis=0).squeeze()
            test_pred_std = preds_test.std(axis=0).squeeze()
        else:
            train_pred_mean, train_pred_std = pl_model(X_train)
            train_pred_mean = train_pred_mean.view(-1)
            train_pred_std = train_pred_std.view(-1)

            test_pred_mean, test_pred_std = pl_model(X_test)
            test_pred_mean = test_pred_mean.view(-1)
            test_pred_std = test_pred_std.view(-1)

    if show_change_points:
        algo = rpt.Pelt(model="rbf").fit(y_test)
        result = algo.predict(pen=10)

    fig, ax = plt.subplots(layout="constrained")
    legend_kwargs = {
        "loc": "lower left",
        "bbox_to_anchor": (-0.08, 1.02, 1.1, 0.3),
        "mode": "expand",
        "ncol": 3,
        "fontsize": 16,
    }

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

    ax.axvline(xs_time_train[-1], ls="--", color="black", label="Train-Test Split")

    ax.set_xlabel(r"\(\tau\)")
    ax.set_ylabel(r"\(q\) (Sv)")
    # ax.legend(**legend_kwargs)

    # fig_legend = plt.figure(figsize=(8.3, 0.8))
    # plt.figlegend(*ax.get_legend_handles_labels(), ncol=3)
    # fig.tight_layout()

    return fig


def save_fig(fig, save_path, name, ext, **kwargs):
    fig.savefig(osp.join(save_path, f"{name}.{ext}"), **kwargs)
