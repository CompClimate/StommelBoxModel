from typing import Any, List, Optional

import torch
from lightning import LightningDataModule
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, TensorDataset

import utils.data_utils as data_utils
from data.components import box_model as box
from data.components import forcing


class TimeSeriesDatamodule(LightningDataModule):
    def __init__(
        self,
        box_model: box.BoxModel,
        s_forcing: forcing.Forcing,
        t_forcing: forcing.Forcing,
        autoregressive: bool,
        test_size: float,
        nonlinear_density: bool = False,
        window_size: Optional[int] = None,
        batch_size: int = 16,
    ) -> None:
        super().__init__()

        self.save_hyperparameters(logger=False)

        series_dict = box.get_time_series(
            box_model,
            s_forcing=s_forcing,
            t_forcing=t_forcing,
            nonlinear_density=nonlinear_density,
        )
        self.series_dict = series_dict

        X, y = data_utils.get_raw_data(
            series_dict,
            autoregressive,
            window_size,
        )
        self.X, self.y = X, y

        X_train, X_test, y_train, y_test = train_test_split(
            X,
            y,
            test_size=test_size,
            shuffle=False,
        )
        self.X_train, self.X_test, self.y_train, self.y_test = (
            X_train,
            X_test,
            y_train,
            y_test,
        )

        X = torch.from_numpy(X)
        X_train = torch.from_numpy(X_train)
        y_train = torch.from_numpy(y_train)
        X_test = torch.from_numpy(X_test)
        y_test = torch.from_numpy(y_test)

        self.data_train = TensorDataset(X_train, y_train)
        self.data_val = TensorDataset(X_test, y_test)

        self.batch_size_per_device = batch_size

    def setup(self, stage: Optional[str] = None) -> None:
        # Divide batch size by the number of devices.
        if self.trainer is not None:
            if self.hparams.batch_size % self.trainer.world_size != 0:
                raise RuntimeError(
                    f"Batch size ({self.hparams.batch_size}) is not divisible by the number of devices ({self.trainer.world_size})."
                )
            self.batch_size_per_device = (
                self.hparams.batch_size // self.trainer.world_size
            )

    def train_dataloader(self) -> DataLoader[Any]:
        """Create and return the train dataloader.

        :return: The train dataloader.
        """
        return DataLoader(
            dataset=self.data_train,
            batch_size=self.batch_size_per_device,
            # num_workers=self.hparams.num_workers,
            # pin_memory=self.hparams.pin_memory,
            shuffle=True,
        )

    def val_dataloader(self) -> DataLoader[Any]:
        """Create and return the test dataloader.

        :return: The test dataloader.
        """
        return DataLoader(
            dataset=self.data_val,
            batch_size=self.batch_size_per_device,
            # num_workers=self.hparams.num_workers,
            # pin_memory=self.hparams.pin_memory,
            shuffle=False,
        )

    def test_dataloader(self) -> DataLoader[Any]:
        return DataLoader(
            dataset=self.data_val,
            batch_size=self.batch_size_per_device,
            # num_workers=self.hparams.num_workers,
            # pin_memory=self.hparams.pin_memory,
            shuffle=False,
        )
