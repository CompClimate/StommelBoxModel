from typing import Optional

import fire
import pandas as pd
from shared import Op, Reducer


class CSVReducer(Reducer):
    def __init__(self, csv_files: list[str]):
        self.dfs = [pd.read_csv(f) for f in csv_files]
        self.df = pd.concat(self.dfs).groupby(level=0)
        self.out_df = self.dfs[0]

    def reduce(self, operation: Op, columns: Optional[list[str]] = None):
        assert operation in Op._member_map_.values(), f"Unknown operation {operation}"

        self.out_df = self.dfs[0]
        if columns is not None:
            for column in columns:
                self.out_df[column] = eval(f"self.df['{column}'].{operation}()")
        else:
            self.out_df = eval(f"self.df.{operation}()")

        return self

    def to_csv(self, outfile):
        self.out_df.to_csv(outfile)
        return self


if __name__ == "__main__":
    fire.Fire(CSVReducer)
