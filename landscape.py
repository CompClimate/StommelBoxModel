import torch
import numpy as np
from sklearn.decomposition import PCA
from torch.utils.data import TensorDataset, DataLoader
import pytorch_lightning as pl
import matplotlib
import matplotlib.pyplot as plt
import itertools
import csv
import copy
import random


def set_weights(model, weights):
    # Weights should be a sequence of tensors
    with torch.no_grad():
        for param, new_param in zip(model.parameters(), weights):
            param.copy_(new_param)


class RandomCoordinates:
    """
    Implements the random coordinates approach for Loss Landscape Visualization.
    """

    def __init__(self, origin, dim=2):
        self.origin_ = origin
        self.dim = dim
        self.v0_ = normalize_weights(
            [np.random.normal(size=w.shape) for w in origin], origin
        )
        self.v1_ = normalize_weights(
            [np.random.normal(size=w.shape) for w in origin], origin
        )

    def __call__(self, a, b=None):
        if self.dim == 3:
            return [
                a * w0 + b * w1 + wc
                for w0, w1, wc in zip(self.v0_, self.v1_, self.origin_)
            ]
        elif self.dim == 2:
            return [a * w0 + wc for w0, wc in zip(self.v0_, self.origin_)]


@torch.no_grad()
def normalize_weights(weights, origin):
    """
    Performs an element-wise normalization of the weight matrices.
    """
    return [
        w * np.linalg.norm(wc) / np.linalg.norm(w) for w, wc in zip(weights, origin)
    ]


def vectorize_weights_(weights):
    """
    A utility to transform the raw model parameters into a numpy array.
    """
    vec = [w.flatten() for w in weights]
    vec = [
        t.cpu().clone().detach().numpy()
        if isinstance(t, torch.Tensor)
        else torch.tensor(t, device="cpu").numpy()
        for t in vec
    ]
    vec = np.hstack(vec)
    return vec


def vectorize_weight_list_(weight_list):
    vec_list = []
    for weights in weight_list:
        if not weights:
            continue
        vec_list.append(vectorize_weights_(weights))
    weight_matrix = np.column_stack(vec_list)
    return weight_matrix


def shape_weight_matrix_like_(weight_matrix, example):
    weight_vecs = np.hsplit(weight_matrix, weight_matrix.shape[1])
    sizes = [torch.numel(v) for v in example]
    shapes = [v.shape for v in example]
    weight_list = []
    for net_weights in weight_vecs:
        vs = np.split(net_weights, np.cumsum(sizes))[:-1]
        vs = [v.reshape(s) for v, s in zip(vs, shapes)]
        weight_list.append(vs)
    return weight_list


def get_path_components_(training_path, n_components=2):
    # Vectorize network weights
    weight_matrix = vectorize_weight_list_(training_path)
    # Create components
    pca = PCA(n_components=n_components, whiten=True)
    components = pca.fit_transform(weight_matrix)
    # Reshape to fit network
    example = training_path[1]
    weight_list = shape_weight_matrix_like_(components, example)
    return pca, weight_list


class PCACoordinates:
    """
    Implements the PCA coordinates approach for Loss Landscape Visualization.
    """

    def __init__(self, training_path, dim=3):
        origin = training_path[-1]
        self.dim = dim
        self.pca_, self.components = get_path_components_(
            training_path, n_components=dim - 1
        )
        self.set_origin([t.to("cpu").detach().numpy() for t in origin])

    def __call__(self, a, b=None):
        if self.dim == 3:
            return [
                a * w0 + b * w1 + wc
                for w0, w1, wc in zip(self.v0_, self.v1_, self.origin_)
            ]
        elif self.dim == 2:
            return [a * w0 + wc for w0, wc in zip(self.v0_, self.origin_)]

    def set_origin(self, origin, renorm=True):
        self.origin_ = origin
        if renorm:
            self.v0_ = normalize_weights(self.components[0], origin)
            if self.dim == 3:
                self.v1_ = normalize_weights(self.components[1], origin)


class LossSurface:
    """
    Represents the loss surface of a model on a data set using a loss function.
    """

    def __init__(self, model, inputs, outputs):
        self.model_ = model
        self.inputs_ = inputs
        self.outputs_ = outputs

    def compile(
        self,
        range_,
        points,
        coords,
        loss_fun,
        surf=True,
        no_lg=False,
        device="cuda:0",
    ):
        r"""
        Computes the loss surface as a set of triples :math:`(a, b, l)`.

        Args:
            `range`: The domain over which to compute the loss surface.
                     This is expected to be a scalar multiplier such that :math:`\text{range} \cdot \left[-1, 1\right]`
                     is the final domain.
            `points`: The amount of triples to sample.
            `coords`: A ``RandomCoordinates`` or ``PCACoordiantes`` object.
            `loss_fun`: The loss function to sample from. This is expected to be a callable object (preferably a PyTorch composed function).
            `surf`: Whether to compile a 3d surface or a 2d line line.
            `log_lg`: Whether to record the loss-gradient-norm pair for each compiled point.

        :return: A ``matplotlib`` Axes object.
        """

        test_batch = TensorDataset(self.inputs_, self.outputs_)
        testbatch_loader = DataLoader(test_batch, shuffle=True)

        if surf:
            a_grid = torch.linspace(-1.0, 1.0, steps=points) ** 3 * range_
            b_grid = torch.linspace(-1.0, 1.0, steps=points) ** 3 * range_
            loss_grid = np.empty([len(a_grid), len(b_grid)])
            gradient_grid = np.empty([len(a_grid), len(b_grid)])

            self.model_ = self.model_.to(device)

            for i, a in enumerate(a_grid):
                for j, b in enumerate(b_grid):
                    c = coords(a, b)
                    self.model_.set_weights(c, device=device)

                    loss = 0.0
                    for x, y in testbatch_loader:
                        x, y = x.to(device), y.to(device)
                        yhat = self.model_(x)
                        loss += loss_fun(yhat, y).item()
                    loss /= len(testbatch_loader)

                    if not no_lg:
                        delta_f_D = self.model_.delta(
                            self.inputs_.to(device), self.outputs_.to(device), loss_fun
                        )
                        raw_jacobian = torch.autograd.functional.jacobian(
                            delta_f_D,
                            tuple([_.view(-1) for _ in self.model_.parameters()]),
                        )
                        cat_jacobian = torch.cat(raw_jacobian)
                        cat_jacobian_norm = torch.linalg.norm(cat_jacobian)
                        gradient_grid[j, i] = cat_jacobian_norm

                    loss_grid[j, i] = loss

            self.model_.set_weights(
                list(map(lambda ary: torch.from_numpy(ary), coords.origin_))
            )
            self.a_grid_ = a_grid
            self.b_grid_ = b_grid
            self.loss_grid_ = loss_grid
            self.gradient_grid_ = gradient_grid
        else:
            a_line = torch.linspace(-1.0, 1.0, steps=points) ** 3 * range_
            loss_line = np.empty(len(a_line))
            gradient_line = np.empty(len(a_line))

            self.model_ = self.model_.to(device)

            for i, a in enumerate(a_line):
                print(f"Point {i}/{len(a_line)}")
                c = coords(a)
                set_weights(self.model_, c)

                loss = 0.0
                for x, y in testbatch_loader:
                    x, y = x.to(device), y.to(device)
                    yhat, _ = self.model_(x)
                    yhat = yhat.unsqueeze(dim=-1)
                    loss += loss_fun(yhat, y).item()
                loss /= len(testbatch_loader)

                # delta_f_D = self.model_.delta(self.inputs_.to(gpu2), self.outputs_.to(gpu2), loss_fun)

                # print(torch.cuda.memory_summary(device=gpu1))
                # print(torch.cuda.memory_summary(device=gpu2))

                # raw_jacobian = jacobian(
                #     delta_f_D, tuple([_.view(-1) for _ in self.model_.parameters()])
                # )
                # cat_jacobian = torch.cat(raw_jacobian)
                # cat_jacobian_norm = torch.linalg.norm(cat_jacobian)

                loss_line[i] = loss
                # gradient_line[i] = cat_jacobian_norm

            ls_params = list(map(lambda ary: ary.data, coords.origin_))
            set_weights(self.model_, ls_params)

            self.a_line_ = a_line
            self.loss_line_ = loss_line
            self.gradient_line_ = gradient_line

    def plot(self, title, levels=20, ax=None, cmap="magma", surf=False, **kwargs):
        """
        Plots the surface as a matplotlib contour plot.

        :return: A ``matplotlib`` Axes object.
        """
        if surf:
            xs = self.a_grid_
            ys = self.b_grid_
            zs = self.loss_grid_
        else:
            xs = self.a_line_
            ys = self.loss_line_

        fig = None

        if ax is None:
            fig, ax = plt.subplots(**kwargs)
            ax.set_title(title)
            ax.set_aspect("equal")

        if surf:
            # Set Levels
            min_loss = zs.min()
            max_loss = zs.max()
            levels = np.exp(np.linspace(np.log(min_loss), np.log(max_loss), num=levels))
            # Create Contour Plot
            CS = ax.contour(
                xs,
                ys,
                zs,
                levels=levels,
                cmap=cmap,
                linewidths=0.75,
                norm=matplotlib.colors.LogNorm(vmin=min_loss, vmax=max_loss * 2.0),
            )
            ax.clabel(CS, inline=True, fontsize=8, fmt="%1.2f")
        else:
            plot = ax.plot(xs, ys)
            ax.set_xlabel("a")
            ax.set_ylabel("loss")

        return fig, ax

    def to_csv(self, fname, surf=True):
        """
        Saves the loss surface to a ``.csv`` file.
        """
        with open(fname, "w+") as f:
            writer = csv.writer(f)
            if surf:
                writer.writerow(["xcoordinates", "ycoordinates", "train_loss"])
                xys = list(
                    itertools.product(self.a_grid_.numpy(), self.b_grid_.numpy())
                )
                xs = [tup[0] for tup in xys]
                ys = [tup[1] for tup in xys]
                writer.writerows(zip(xs, ys, self.loss_grid_.flatten()))
            else:
                writer.writerow(["xcoordinates", "train_loss"])
                writer.writerows(zip(self.a_grid_.numpy(), self.loss_line_))


def weights_to_coordinates(coords, training_path, surf=True):
    """
    Projects the training path onto the first two principal components using the pseudoinverse.
    """
    components = [coords.v0_]
    if surf:
        components.append(coords.v1_)

    comp_matrix = vectorize_weight_list_(components)
    # the pseudoinverse
    comp_matrix_i = np.linalg.pinv(comp_matrix)
    # the origin vector
    w_c = vectorize_weights_(training_path[-1])
    coord_path = np.array(
        [
            comp_matrix_i @ (vectorize_weights_(weights) - w_c)
            for weights in training_path
        ]
    )
    return coord_path


def plot_training_path(
    coords,
    training_path,
    fig=None,
    ax=None,
    end=None,
    surf=True,
    loss_history=[],
    **kwargs,
):
    """
    Args:
        `loss_history`: A list of loss values, each belonging to one item in ``training_path``.
    """
    path = weights_to_coordinates(coords, training_path, surf=surf)

    if ax is None:
        fig, ax = plt.subplots(**kwargs)

    colors = range(path.shape[0])
    end = path.shape[0] if end is None else end
    norm = plt.Normalize(0, end)

    if surf:
        xs = (path[:, 0],)
        ys = path[:, 1]
    else:
        xs = path[:, 0]
        ys = loss_history

    ax.scatter(xs, ys, s=4, c=colors, cmap="cividis", norm=norm)

    return fig, ax


def columns_to_csv_(csv_filepath, colnames, cols):
    """
    Writes a sequence of sequences of equal length to a csv.

    Args:
        `colnames`: The column names.
        `cols`: The values of the column entries.
    """
    with open(csv_filepath, "w+") as f:
        writer = csv.writer(f)
        writer.writerow(colnames)
        writer.writerows(list(zip(*cols)))


class TrainingPathCallback(pl.Callback):
    def __init__(self, net):
        super().__init__()
        self.net = net

        initial_point = list(self.net.parameters())
        ps = []
        for parameter in initial_point:
            ps.append(parameter.data.clone())
        self.training_path = [ps]

    def on_train_epoch_end(self, trainer, pl_module):
        # The problem in directly passing `net.parameters()` consists in the return value of
        # `net.parameters()`; it is a list of tensor references in order not to pass around huge
        # tensors. If we add this to the training path, we end up receiving a final training path
        # of zeros only since w_c will be equal to all other elements in the path. Subtracting
        # w_c from any element then leads to the 0-tensor and we receive a single point at (0, 0).
        # We thus clone each of the referenced tensors such that we actually collect the weights.
        params = list(self.net.parameters())
        ps = []

        for param in params:
            ps.append(param.data.clone())

        self.training_path.append(ps)
