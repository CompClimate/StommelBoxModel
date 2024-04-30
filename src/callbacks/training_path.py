import os
import os.path as osp
import pickle

import lightning.pytorch as pl


class TrainingPath(pl.Callback):
    PKL_FILE_NAME = "training_path.pkl"

    def __init__(self, save_dir, name):
        super().__init__()
        self.save_dir = save_dir
        self.name = name
        self.training_path = []

    def _append_params(self, pl_module):
        params = list(pl_module.parameters())
        ps = []

        for param in params:
            ps.append(param.data.clone())

        self.training_path.append(ps)

    def on_train_start(self, trainer, pl_module):
        self._append_params(pl_module)

    def on_train_epoch_end(self, trainer, pl_module):
        # The problem in directly passing `net.parameters()` consists in the return value of
        # `net.parameters()`; it is a list of tensor references in order not to pass around huge
        # tensors. If we add this to the training path, we end up receiving a final training path
        # of zeros only since w_c will be equal to all other elements in the path. Subtracting
        # w_c from any element then leads to the 0-tensor and we receive a single point at (0, 0).
        # We thus clone each of the referenced tensors such that we actually
        # collect the weights.
        self._append_params(pl_module)

    def on_train_end(self, trainer, pl_module):
        path = osp.join(self.save_dir, self.name)
        os.makedirs(path, exist_ok=True)

        with open(osp.join(path, self.PKL_FILE_NAME), "wb") as f:
            pickle.dump(self.training_path, f)

    def load_state_dict(self, state_dict):
        self.training_path = state_dict["training_path"]

    def state_dict(self):
        return {"training_path": self.training_path}
