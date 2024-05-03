import itertools
import os.path as osp
from typing import Iterable

import dill
import fire
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib import gridspec

colors = plt.rcParams["axes.prop_cycle"]()


class Plotter:
    def __init__(self, pkl_files: Iterable[str], **rcParams):
        plt.rcParams.update(**rcParams)
        plt.rcParams.update(
            {
                "text.latex.preamble": r"""
    \usepackage{amsmath}
    \usepackage{amssymb}
    \usepackage{textcomp,mathcomp}
    """
            }
        )

        series_dicts = []
        for pkl_file in pkl_files:
            with open(pkl_file, "rb") as f:
                series_dicts.append(dill.load(f))

        variables = [
            list(series_dict["features"]["variables"].values())
            for series_dict in series_dicts
        ]
        var_names = [
            list(series_dict["features"]["variables"].keys())
            for series_dict in series_dicts
        ]

        var_labels = []
        for i, ls_var in enumerate(var_names):
            for v in ls_var:
                var_label = series_dicts[i]["latex"]["variables"][v]
                var_labels.append(var_label)

        var_unit_labels = []
        for i, ls_var in enumerate(var_names):
            for v in ls_var:
                var_unit_label = series_dicts[i]["units"][v]
                var_unit_labels.append(var_unit_label)

        variables = list(itertools.chain(*variables))
        forcings = [
            list(series_dict["features"]["forcings"].values())
            for series_dict in series_dicts
        ]
        forcing_names = [
            list(series_dict["features"]["forcings"].keys())
            for series_dict in series_dicts
        ]

        forcing_labels = []
        for i, ls_forcing in enumerate(forcing_names):
            for v in ls_forcing:
                forcing_label = series_dicts[i]["latex"]["forcings"][v]
                forcing_labels.append(forcing_label)

        forcing_unit_labels = []
        for i, ls_forcing in enumerate(forcing_names):
            for v in ls_forcing:
                forcing_unit_label = series_dicts[i]["units"][v]
                forcing_unit_labels.append(forcing_unit_label)

        forcings = list(itertools.chain(*forcings))
        qs = [series_dict["q"] for series_dict in series_dicts]

        N_vars = len(variables)
        N_forcings = len(forcings)
        N_qs = len(qs)

        gs_vars = gridspec.GridSpec(
            nrows=max(N_vars // 2, 1), ncols=max(N_vars // 2, 1)
        )
        gs_forcings = gridspec.GridSpec(nrows=N_forcings, ncols=1)
        gs_qs = gridspec.GridSpec(nrows=1, ncols=N_qs)

        self.fig_vars = plt.figure()
        self.fig_forcings = plt.figure()
        self.fig_qs = plt.figure()

        for i in range(N_vars):
            ax = self.fig_vars.add_subplot(gs_vars[i])
            y = variables[i]
            sns.lineplot(x=range(1, len(y) + 1), y=y, ax=ax)
            ax.set_xlabel(r"\(\tau\)")
            ax.set_ylabel(var_labels[i] + f" ({var_unit_labels[i]})")

        for i in range(N_forcings):
            ax = self.fig_forcings.add_subplot(gs_forcings[i])
            y = forcings[i]
            sns.lineplot(x=range(1, len(y) + 1), y=y, ax=ax)
            ax.set_xlabel(r"\(\tau\)")
            ax.set_ylabel(forcing_labels[i] + f" ({forcing_unit_labels[i]})")

        for i in range(N_qs):
            ax = self.fig_qs.add_subplot(gs_qs[i])
            y = qs[i]
            sns.lineplot(x=range(1, len(y) + 1), y=y, ax=ax)
            ax.set_xlabel(r"\(\tau\)")
            ax.set_ylabel(r"\(q\) (Sv)")

    def __str__(self):
        return ""

    def width(self, w):
        self.fig_vars.set_figwidth(w)
        self.fig_forcings.set_figwidth(w)
        self.fig_qs.set_figwidth(w)
        return self

    def height(self, h):
        self.fig_vars.set_figheight(h)
        self.fig_forcings.set_figheight(h)
        self.fig_qs.set_figheight(h)
        return self

    def tight_layout(self):
        self.fig_vars.tight_layout()
        self.fig_forcings.tight_layout()
        self.fig_qs.tight_layout()
        return self

    def save(self, directory: str, name: str, ext: str, **kwargs):
        p = osp.join(directory, name)
        self.fig_vars.savefig(p + "_vars" + ext, **kwargs)
        self.fig_forcings.savefig(p + "_forcings" + ext, **kwargs)
        self.fig_qs.savefig(p + "_qs" + ext, **kwargs)
        return self


if __name__ == "__main__":
    fire.Fire(Plotter)
