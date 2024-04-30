import fire
import polars as pl
from shared import Plotter
from tbparse import SummaryReader


class TbPlotter(Plotter):
    def __init__(self, tb_filename: str, **rcParams):
        super().__init__(**rcParams)

        reader = SummaryReader(log_path=tb_filename)
        # step | tag | value
        self.df = pl.from_pandas(reader.scalars)


if __name__ == "__main__":
    fire.Fire(TbPlotter)
