from abc import ABC, abstractmethod
from enum import StrEnum
from typing import Callable, Iterable, Optional

import matplotlib.pyplot as plt
import polars as pl
import seaborn as sns
import shap
from seaborn.axisgrid import _BaseGrid


class Op(StrEnum):
    Mean = "mean"
    Min = "min"
    Max = "max"
    Median = "median"
    Std = "std"
    Var = "var"


class Reducer(ABC):
    def __str__(self):
        return ""

    @abstractmethod
    def reduce(self, operation: Op, columns: Optional[list[str]] = None):
        pass


class Plotter(ABC):
    def __init__(self, **rcParams):
        self.rcParams(**rcParams)
        self.fig, self.ax = plt.subplots()

    def __str__(self):
        return ""

    def _plot_vars(self, fun: Callable, vars: Iterable[str], **kwargs):
        for var in vars:
            df_var = self.df.filter(pl.col("tag") == var)
            fun(data=df_var, label=var, ax=self.ax, **kwargs)

    def _plot(
        self,
        plot_fun_name: str,
        vars: Iterable[str],
        include_ax: bool = True,
        pd: bool = False,
        use_shap: bool = False,
        **kwargs,
    ):
        fun = (
            eval(f"shap.plots.{plot_fun_name}")
            if use_shap
            else eval(f"sns.{plot_fun_name}")
        )

        if len(vars) == 0:
            if include_ax:
                kwargs["ax"] = self.ax

            df = self.df
            if pd:
                df = df.to_pandas()

            ret_value = fun(data=df, **kwargs)
            if isinstance(ret_value, _BaseGrid):
                self.fig = ret_value
        else:
            self._plot_vars(fun, vars, use_shap=use_shap, **kwargs)

        return self

    def rcParams(self, **rcParams):
        plt.rcParams.update(rcParams)
        return self

    def shap_heatmap(self, column_pattern, **kwargs):
        cols = [
            col for col in self.df.columns if col.lower().find(column_pattern) != -1
        ]
        return self._plot("heatmap", cols, use_shap=True, **kwargs)

    def lineplot(self, vars=[], **kwargs):
        return self._plot("lineplot", vars, **kwargs)

    def scatterplot(self, vars=[], **kwargs):
        return self._plot("scatterplot", vars, **kwargs)

    def kdeplot(self, vars=[], **kwargs):
        return self._plot("kdeplot", vars, **kwargs)

    def pairplot(self, **kwargs):
        return self._plot("pairplot", vars=[], **kwargs)

    def bands(self, col_name, std, shift_x: bool = False, **kwargs):
        if kwargs.get("x") is None:
            kwargs["x"] = (
                range(len(self.df))
                if not shift_x
                else range(
                    len(self.df), len(self.df) + len(self.df.get_column(col_name))
                )
            )

        col = self.df.get_column(col_name).to_numpy()
        val = self.df.get_column(std).to_numpy()
        self.ax.fill_between(y1=col - val, y2=col + val, alpha=0.3, **kwargs)
        return self

    def show(self):
        plt.show()
        return self

    def savefig(self, fname, **kwargs):
        self.fig.savefig(fname, **kwargs)
        return self

    def set(self, xlabel: str, ylabel: str):
        self.ax.set(xlabel=xlabel, ylabel=ylabel)
        return self

    def nolegend(self):
        self.ax.get_legend().set_visible(False)
        return self

    def legend(self):
        self.ax.legend()
        return self

    def tight_layout(self):
        self.fig.tight_layout()
        return self
