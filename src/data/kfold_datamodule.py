from lightning.pytorch import LightningDataModule
from sklearn.model_selection import KFold
from torch.utils.data import DataLoader, Dataset


class LitKFoldDataModule(LightningDataModule):
    def __init__(
        self,
        dataset: Dataset,
        split_seed: int = 12345,
        num_splits: int = 10,
        batch_size: int = 32,
        num_workers: int = 0,
        pin_memory: bool = False,
    ):
        super().__init__()

        self.save_hyperparameters(logger=False, ignore=["dataset"])

        # num_splits = 10 means our dataset will be split to 10 parts
        # so we train on 90% of the data and validate on 10%
        # assert 1 <= k <= num_splits, "Incorrect fold number"

        self.dataset = dataset

    def setup(self, k: int = 1, stage: str = None):
        kf = KFold(
            n_splits=self.hparams.num_splits,
            shuffle=True,
            random_state=self.hparams.split_seed,
        )
        all_splits = [k for k in kf.split(self.dataset)]
        train_indexes, val_indexes = all_splits[k]
        train_indexes, val_indexes = train_indexes.tolist(), val_indexes.tolist()

        self.train_set, self.val_set = (
            self.dataset[train_indexes],
            self.dataset[val_indexes],
        )

    def train_dataloader(self):
        return DataLoader(
            self.train_set,
            shuffle=True,
            batch_size=self.hparams.batch_size,
            num_workers=self.hparams.num_workers,
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_set,
            shuffle=False,
            batch_size=self.hparams.batch_size,
            num_workers=self.hparams.num_workers,
        )

    def test_dataloader(self):
        return DataLoader(
            self.val_set,
            shuffle=False,
            batch_size=self.hparams.batch_size,
            num_workers=self.hparams.num_workers,
        )

    def predict_dataloader(self):
        return DataLoader(
            self.val_set,
            shuffle=False,
            batch_size=self.hparams.batch_size,
            num_workers=self.hparams.num_workers,
        )
