from glob import glob
from typing import Iterable

import fire
import matplotlib as mpl
import numpy as np
import polars as pl
from shared import Plotter


class NpyPlotter(Plotter):
    def __init__(
        self,
        npy_filenames: Iterable[str],
        column_names: Iterable[str] = None,
        **rcParams,
    ):
        super().__init__(**rcParams)

        # font_names = mpl.font_manager.get_font_names()
        # print(font_names)

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


if __name__ == "__main__":
    fire.Fire(NpyPlotter)
