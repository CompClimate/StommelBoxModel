from glob import glob
from typing import Optional

import fire
import tensorboard_reducer as tbr

from . import Op


class TBReducer:
    def __init__(self, glob_pattern: str):
        input_event_dirs = sorted(glob(glob_pattern, recursive=True))
        self.events_dict = tbr.load_tb_events(input_event_dirs)
        self.reduced_events = None

        # Number of recorded tags
        self.n_scalars = len(self.events_dict)
        self.n_steps, self.n_events = list(self.events_dict.values())[0].shape

        print(
            f"Loaded {self.n_events} TensorBoard runs with {self.n_scalars} scalars and {self.n_steps} steps each"
        )
        print(", ".join(self.events_dict))

    def __str__(self):
        return ""

    def reduce(self, reduce_ops: list[str] = ["mean"]):
        assert set(reduce_ops).issubset(Op._member_map_.values()), "Unknown operation."

        self.reduced_events = tbr.reduce_events(self.events_dict, reduce_ops)
        return self

    def save(self, output_dir: str, overwrite: Optional[bool] = False):
        tbr.write_tb_events(self.reduced_events, output_dir, overwrite)
        return self

    def to_csv(self, csv_out_path: str, overwrite: Optional[bool] = False):
        print(f"Writing results to '{csv_out_path}'")
        tbr.write_data_file(self.reduced_events, csv_out_path, overwrite)
        return self


if __name__ == "__main__":
    fire.Fire(TBReducer)
