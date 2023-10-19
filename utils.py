import shap
from shap import Explanation
from shap.utils import OpChain
from shap.plots import colors
from shap.plots._utils import convert_ordering
import numpy as np
import matplotlib.pyplot as pl


class explain_mode:
    def __init__(self, model):
        self.model = model

    def __enter__(self):
        self.model.explain_mode = True

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.model.explain_mode = False


def explain(model, X, n_features):
    """
    Computes Shapley values for a model and given input using integrated gradients.
    It is assumed that the input is in the form of a time series with
    input series length `n_features`.
    """
    features = [f'\(t - {i}\)' for i in reversed(range(1, n_features + 1))]
    
    e = shap.GradientExplainer(model, X)

    e.features = features
    shap_values = e(X)
    
    return shap_values


def explain_captum(model, attr_algorithm_cls, X, n_features, **kwargs):
    # ig = attr.IntegratedGradients(model)
    # ig_nt = attr.NoiseTunnel(ig)
    # dl = attr.DeepLift(model)
    # gs = attr.GradientShap(model)
    # fa = attr.FeatureAblation(model)
    # lrp = attr.LRP(model)

    # ig_attr_test = ig.attribute(X_test, target=0, n_steps=50)
    # ig_nt_attr_test = ig_nt.attribute(X_test)
    # fa_attr_test = fa.attribute(X_test)

    attr_alg = attr_algorithm_cls(model)
    attrs = attr_alg.attribute(X, **kwargs)

    features = [f'\(t - {i}\)' for i in reversed(range(1, n_features + 1))]
    return shap.Explanation(values=attrs, data=X, feature_names=features)


def heatmap(attr_values, instance_order=Explanation.hclust(), feature_values=Explanation.abs.mean(0),
            feature_order=None, max_display=10, cmap=colors.red_white_blue, show=True,
            plot_width=8, ylabel='Attribution Value'):
    # sort the SHAP values matrix by rows and columns
    values = attr_values.values
    if issubclass(type(feature_values), OpChain):
        feature_values = feature_values.apply(Explanation(values))
    if issubclass(type(feature_values), Explanation):
        feature_values = feature_values.values
    if feature_order is None:
        feature_order = np.argsort(-feature_values)
    elif issubclass(type(feature_order), OpChain):
        feature_order = feature_order.apply(Explanation(values))
    elif not hasattr(feature_order, "__len__"):
        raise Exception("Unsupported feature_order: %s!" % str(feature_order))
    xlabel = "Instances"
    instance_order = convert_ordering(instance_order, attr_values)
    # if issubclass(type(instance_order), OpChain):
    #     #xlabel += " " + instance_order.summary_string("SHAP values")
    #     instance_order = instance_order.apply(Explanation(values))
    # elif not hasattr(instance_order, "__len__"):
    #     raise Exception("Unsupported instance_order: %s!" % str(instance_order))
    # else:
    #     instance_order_ops = None

    feature_names = np.array(attr_values.feature_names)[feature_order]
    values = attr_values.values[instance_order][:,feature_order]
    feature_values = feature_values[feature_order]

    # if we have more features than `max_display`, then group all the excess features
    # into a single feature
    if values.shape[1] > max_display:
        new_values = np.zeros((values.shape[0], max_display))
        new_values[:, :-1] = values[:, :max_display-1]
        new_values[:, -1] = values[:, max_display-1:].sum(1)
        new_feature_values = np.zeros(max_display)
        new_feature_values[:-1] = feature_values[:max_display-1]
        new_feature_values[-1] = feature_values[max_display-1:].sum()
        feature_names = [
            *feature_names[:max_display-1],
            f"Sum of {values.shape[1] - max_display + 1} other features",
        ]
        values = new_values
        feature_values = new_feature_values

    # define the plot size based on how many features we are plotting
    row_height = 0.5
    pl.gcf().set_size_inches(plot_width, values.shape[1] * row_height + 2.5)
    ax = pl.gca()

    # plot the matrix of SHAP values as a heat map
    vmin, vmax = np.nanpercentile(values.flatten(), [1, 99])
    ax.imshow(
        values.T,
        aspect=0.7 * values.shape[0] / values.shape[1],
        interpolation="nearest",
        vmin=min(vmin,-vmax),
        vmax=max(-vmin,vmax),
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
        [r"\(\small{\underset{k}{\mathbb{E}}[\text{Attr}(t - k)]}\)", *heatmap_yticks_labels],
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
