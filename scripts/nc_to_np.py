import os
import os.path as osp
import pprint

import fire
import numpy as np
import xarray as xr


class Converter:
    def __init__(self, input_file: str, **kwargs):
        self.input_file = input_file
        self.xdf = xr.open_dataset(
            input_file,
            engine="netcdf4",
            decode_times=True,
            mask_and_scale=True,
            **kwargs,
        )

    def __str__(self):
        return pprint.pformat(self.xdf)

    def describe(self):
        pprint.pprint(self)
        return self

    def convert_multiple(
        self, output_dir: str, variable_name: str, input_files: list[str]
    ):
        os.makedirs(output_dir, exist_ok=True)
        print(f"{input_files = }")

        for input_file in input_files:
            self.convert(input_file, variable_name, output_dir, False)

        return self

    def convert(self, variable_name: str, output_dir: str, concat_exp_ver: False):
        # Extract the variable and convert it to a numpy array.
        arr = self.xdf[variable_name].to_numpy()

        if concat_exp_ver:
            # Concatenate along the `expver` dimension.
            arr = arr.reshape(arr.shape[0], -1, arr.shape[2])

        output_filename = osp.basename(self.input_file)
        output_filename = osp.splitext(output_filename)[0]
        output_file = osp.join(output_dir, output_filename)

        np.save(output_file, arr)


if __name__ == "__main__":
    fire.Fire(Converter)
