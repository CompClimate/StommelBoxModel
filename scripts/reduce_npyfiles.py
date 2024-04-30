from glob import glob
from itertools import chain
from typing import Optional

import fire
import matplotlib.pyplot as plt
import numpy as np
from braceexpand import braceexpand
from shared import Op, Reducer


class NpyReducer(Reducer):
    def __init__(self, npy_filenames):
        filenames = (
            list(
                chain.from_iterable(
                    [
                        glob(fname, recursive=True)
                        for fname in list(braceexpand(npy_filenames))
                    ]
                )
            )
            if isinstance(npy_filenames, str)
            else npy_filenames
        )

        arr = []
        for fname in filenames:
            arr.append(np.load(fname))
        self.arr = np.stack(arr, axis=-1)
        self.arr_reduced = None

    def reduce(self, operation: Op, columns: Optional[list[str]] = None):
        print(f"{self.arr.shape = }")
        self.arr_reduced = eval(f"self.arr.{operation}(axis=-1)")
        print(f"{self.arr_reduced.shape = }")
        return self

    def save(self, filename):
        np.save(filename, self.arr_reduced)
        return self


if __name__ == "__main__":
    fire.Fire(NpyReducer)
