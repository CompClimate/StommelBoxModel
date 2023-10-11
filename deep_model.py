import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torchmetrics.regression import MeanSquaredError
from sklearn.preprocessing import MinMaxScaler
import lightning.pytorch as pl


def sliding_windows(data, seq_length):
    x = []
    y = []

    for i in range(len(data) - seq_length - 1):
        _x = data[i:(i + seq_length)]
        _y = data[i + seq_length]
        x.append(_x)
        y.append(_y)

    return np.array(x), np.array(y)


def prepare_data(train, test, seq_len=10):
    # sc = MinMaxScaler()

    # train = sc.fit_transform(train)
    # test = sc.fit_transform(test)

    X_train, y_train = sliding_windows(train, seq_len)
    X_test, y_test = sliding_windows(test, seq_len)

    X_train, X_test = X_train.astype(np.float32), X_test.astype(np.float32)
    y_train, y_test = y_train.astype(np.float32), y_test.astype(np.float32)

    return (X_train, y_train), (X_test, y_test)


class RNNModel(nn.Module):
    def __init__(self,
                 num_classes,
                 input_size,
                 hidden_size,
                 num_layers,
                 seq_len,
                 rnn_dropout=0.3,
                 bidirectional=False):
        super().__init__()
        
        self.num_classes = num_classes
        self.num_layers = num_layers
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.seq_len = seq_len
        
        self.rnn = nn.LSTM(input_size=input_size,
                            hidden_size=hidden_size,
                            num_layers=num_layers,
                            batch_first=True,
                            dropout=rnn_dropout,
                            bidirectional=bidirectional)
        self.fc = nn.Linear(hidden_size * 2 if bidirectional else hidden_size,
                            num_classes)

    def forward(self, x):
        x, _ = self.rnn(x)
        x = x[:, -1, :]
        out = self.fc(x)
        return out


class Model(pl.LightningModule):
    def __init__(self,
                 torch_model,
                 loss_fun,
                 lr,
                 lr_scheduler=None):
        super().__init__()
        self.model = torch_model
        self.loss_fun = loss_fun
        self.lr = lr
        self.lr_scheduler = lr_scheduler
        self.train_loss = MeanSquaredError()
        self.val_loss = MeanSquaredError()

    def forward(self, x):
        return self.model.forward(x)
    
    def training_step(self, batch, batch_idx):
        x, y = batch
        pred = self(x)
        loss = self.loss_fun(pred, y)

        self.train_loss(pred, y)
        self.log(
            'train_loss',
            self.train_loss,
            on_step=False,
            on_epoch=True,
            prog_bar=True,
            sync_dist=True,
        )
        
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        pred = self(x)
        loss = self.loss_fun(pred, y)

        self.val_loss(pred, y)
        self.log(
            'val_loss',
            self.val_loss,
            on_step=False,
            on_epoch=True,
            prog_bar=True,
            sync_dist=True,
        )

        return loss

    def configure_optimizers(self):
        optimizer = optim.Adam(
            self.model.parameters(),
            lr=self.lr,
        )
        lr_scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer,
                                                            T_max=10)
        return dict(
            optimizer=optimizer,
            lr_scheduler=lr_scheduler,
        )
