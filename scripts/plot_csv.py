from typing import Iterable

import fire
import numpy as np
import pandas as pd
import polars as pl
import shap
from shared import Plotter

# Plot SHAP values
# python scripts/plot_csv.py \
# --csv_filenames='["logs/explain/runs/F5_hifreq_pi_de_shap_deeplift/explanations.csv"]' \
# --text.usetex=True - \
# shap_heatmap '["shap_1","shap_2","shap_3","shap_4","shap_5","shap_6"]' \
# --feature_names='["\(S_1\)","\(S_2\)","\(T_1\)","\(T_2\)","\(F_s\)","\(F_t\)"]' - \
# tight_layout \
# savefig F5_hifreq_feature_1.png --dpi=800


class renamer:
    def __init__(self):
        self.d = dict()

    def __call__(self, x):
        if x not in self.d:
            self.d[x] = 0
            return x
        else:
            self.d[x] += 1
            return "%s_%d" % (x, self.d[x])


class CSVPlotter(Plotter):
    def __init__(self, csv_filenames: Iterable[str], **rcParams):
        super().__init__(**rcParams)

        dfs = [pd.read_csv(fname) for fname in csv_filenames]
        self.df = pd.concat(dfs, axis=1)
        self.df = self.df.rename(columns=renamer())
        self.df = pl.from_pandas(self.df)

    def _plot_vars(
        self, fun, vars, use_shap: bool = False, shift_x: bool = False, **kwargs
    ):
        if kwargs.get("x") is None:
            kwargs["x"] = (
                range(len(self.df))
                if not shift_x
                else range(
                    len(self.df), len(self.df) + len(self.df.get_column(kwargs["y"]))
                )
            )

        if use_shap:
            values = self.df.select(vars).to_numpy()
            feature_names = kwargs.pop("feature_names", vars)
            fun(
                shap.Explanation(
                    values=values,
                    feature_names=feature_names,
                ),
                feature_order=np.arange(len(vars)),
            )
        else:
            fun(data=self.df, label=vars[0], ax=self.ax, **kwargs)


if __name__ == "__main__":
    fire.Fire(CSVPlotter)
