from glob import glob
from typing import Iterable

import fire
import matplotlib.pyplot as plt
import numpy as np
import polars as pl
import seaborn as sns
import seaborn.objects as so


class SOPlotter:
    def __init__(self, npy_filenames, column_names: Iterable[str] = None, **rcParams):
        plt.rcParams.update(**rcParams)

        filenames = (
            glob(npy_filenames, recursive=True)
            if isinstance(npy_filenames, str)
            else npy_filenames
        )

        arr = []
        for fname in filenames:
            arr.append(np.load(fname).reshape(-1, 1))
        arr = np.hstack(arr)

        self.df = pl.from_numpy(arr)
        if column_names is not None:
            self.df.columns = column_names

    def __str__(self):
        return ""

    def plot(self, **kwargs):
        self.p = so.Plot(self.df, **kwargs)
        return self

    def add(self, class_name: str, **kwargs):
        instance = eval(f"so.{class_name}")
        instance = instance(**kwargs)

        t = kwargs.get("transforms")
        transforms = (
            [eval(f"so.{transform_class_name}")() for transform_class_name in t]
            if t
            else []
        )

        self.p = self.p.add(instance, *transforms)
        return self

    def facet(self, **kwargs):
        self.p = self.p.facet(**kwargs)
        return self

    def pair(self, **kwargs):
        self.p = self.p.pair(**kwargs)
        return self

    def theme(self, **kwargs):
        self.p = self.p.theme(kwargs)
        return self

    def ax_style(self, style):
        self.p = self.p.theme(sns.axes_style(style))
        return self

    def save(self, fname, **kwargs):
        self.p.save(fname, **kwargs)
        return self


if __name__ == "__main__":
    fire.Fire(SOPlotter)
