import math
from numbers import Number

import lightning.pytorch as pl
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import MinMaxScaler
from torch.distributions import Normal
from torch.distributions.utils import _standard_normal
from torch.utils.data import DataLoader, TensorDataset
from torchmetrics.regression import MeanSquaredError


def sliding_windows(data, seq_length):
    """
    Transforms a 1d time series into sliding windows of length `seq_length`.
    """
    x = []
    y = []

    for i in range(len(data) - seq_length - 1):
        _x = data[i:(i + seq_length)]
        _y = data[i + seq_length]
        x.append(_x)
        y.append(_y)

    return np.array(x), np.array(y)


def prepare_data(train, test, seq_len, scale=False):
    """From a train/test split, optionally scaled the data and returns sliding windows."""
    if scale:
        sc = MinMaxScaler()
        train = sc.fit_transform(train)
        test = sc.fit_transform(test)

    X_train, y_train = sliding_windows(train, seq_len)
    X_test, y_test = sliding_windows(test, seq_len)

    X_train, X_test = X_train.astype(np.float32), X_test.astype(np.float32)
    y_train, y_test = y_train.astype(np.float32), y_test.astype(np.float32)

    return (X_train, y_train), (X_test, y_test)


class RNNModel(nn.Module):
    """Implements a reccurent model."""
    def __init__(self,
                 num_classes,
                 input_size,
                 hidden_size,
                 num_layers,
                 seq_len,
                 rnn_dropout=0.3,
                 bidirectional=False,
                 quantify_uncertainty=0.0):
        super().__init__()
        
        self.num_classes = num_classes
        self.num_layers = num_layers
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.rnn_dropout = rnn_dropout
        self.bidirectional = bidirectional
        self.seq_len = seq_len
        self.quantify_uncertainty = quantify_uncertainty
        
        self.rnn = nn.LSTM(input_size=input_size,
                            hidden_size=hidden_size,
                            num_layers=num_layers,
                            batch_first=True,
                            dropout=rnn_dropout,
                            bidirectional=bidirectional)
        if quantify_uncertainty > 0.0:
            self.init_second_rnn_()
        self.fc = nn.Linear(hidden_size * 2 if bidirectional else hidden_size,
                            num_classes)

    def init_second_rnn_(self):
        self.rnn2 = nn.LSTM(input_size=self.hidden_size,
                            hidden_size=self.hidden_size,
                            num_layers=self.num_layers,
                            batch_first=True,
                            dropout=self.rnn_dropout,
                            bidirectional=self.bidirectional)

    def forward(self, x):
        x, _ = self.rnn(x)

        if self.quantify_uncertainty > 0.0:
            x = F.dropout(x, self.quantify_uncertainty, training=True)
            x, _ = self.rnn2(x)
            x = F.dropout(x, self.quantify_uncertainty, training=True)

        x = x[:, -1, :]
        out = self.fc(x)
        return out

    @torch.no_grad()
    def quantify(self, x, p, k):
        if not hasattr(self, 'rnn2'):
            self.init_second_rnn_()
        
        quant_old = self.quantify_uncertainty
        self.quantify_uncertainty = p

        preds = torch.hstack([self(x) for _ in range(k)])

        self.quantify_uncertainty = quant_old

        return preds.mean(dim=-1), 6 * preds.std(dim=-1)


class MLPModel(nn.Module):
    """Implements a simple MLP."""
    def __init__(self,
                 input_dim,
                 hidden_dim,
                 n_hidden):
        super().__init__()
        self.input_block = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
        )
        self.hidden_layers = [
            nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim),
                nn.ReLU(),
            )
        ] * n_hidden
        self.hidden_layers = nn.Sequential(*self.hidden_layers)
        self.output_block = nn.Linear(hidden_dim, 1)

    def forward(self, x):
        x = self.input_block(x)
        x = self.hidden_layers(x)
        x = self.output_block(x)
        return x


class BayesMLPModel(nn.Module):
    """Implements a simple Bayesian neural network."""
    def __init__(self,
                 input_dim,
                 hidden_dim,
                 n_hidden,
                 num_MC=1):
        super().__init__()
        self.mc_exp = MC_ExpansionLayer(num_MC=num_MC, input_dim=2)
        self.input_block = nn.Sequential(
            BayesLinear(input_dim, hidden_dim),
            nn.ReLU(),
        )
        self.hidden_layers = [
            nn.Sequential(
                BayesLinear(hidden_dim, hidden_dim),
                nn.ReLU(),
            )
        ] * n_hidden
        self.hidden_layers = nn.Sequential(*self.hidden_layers)
        self.output_block = BayesLinear(hidden_dim, 1)

    def forward(self, x):
        x = self.mc_exp(x)
        x = self.input_block(x)
        x = self.hidden_layers(x)
        x = self.output_block(x)
        mu, std = x.mean(dim=0).squeeze(), x.std(dim=0).squeeze()
        return mu, std


class TorchEnsemble(nn.Module):
    """Implements a deep ensemble given a list of models."""
    def __init__(self, models):
        super().__init__()
        self.models = models
        for i, model in enumerate(self.models):
            self.register_module(f'model_{i}', model)

    def forward(self, x):
        preds = torch.hstack([model(x) for model in self.models])
        return preds.mean(dim=-1), preds.std(dim=-1)


class Model(pl.LightningModule):
    """A Pytorch Lightning wrapper around a torch model."""
    def __init__(self,
                 torch_model,
                 loss_fun,
                 lr,
                 alpha=1e-3,
                 lr_scheduler=None):
        super().__init__()
        self.model = torch_model
        self.loss_fun = loss_fun
        self.lr = lr
        self.alpha = alpha
        self.lr_scheduler = lr_scheduler
        self.train_loss = MeanSquaredError()
        self.val_loss = MeanSquaredError()

    def forward(self, x):
        return self.model.forward(x)
    
    def training_step(self, batch, batch_idx):
        x, y = batch
        pred = self(x)

        if isinstance(pred, tuple):
            pred, _ = pred
        pred = pred.squeeze()
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

        if isinstance(pred, tuple):
            pred, _ = pred
        pred = pred.squeeze()
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
            weight_decay=self.alpha,
        )
        lr_scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer,
                                                            T_max=10)
        # return dict(
        #     optimizer=optimizer,
        #     lr_scheduler=lr_scheduler,
        # )
        return optimizer


class GaussianPrior:
    """Implements a Gaussian prior."""
    def __init__(self, scale):
        assert scale > 0.
        self.scale = scale

    def dist(self):
        return Normal(0, self.scale)


class LaplacePrior(torch.nn.Module):
    """Implements a Laplacian prior."""
    def __init__(self, module, clamp=False):
        super().__init__()

        self.scale = torch.ones_like(module.weight.loc.data)
        module.weight.loc.register_hook(self._save_grad)

        self.clamp = clamp
        self.step = 0
        self.beta = 0.99

    def _save_grad(self, grad):
        self.step += 1
        bias_correction = 1 - self.beta ** self.step
        self.scale.mul_(self.beta).add_(alpha=1-self.beta, other=(1 / grad.data**2).div_(bias_correction+1e-8))

    def dist(self):
        if self.clamp:
            return Normal(0, torch.clamp(self.scale**0.5, min=1.0))
        else:
            return Normal(0, self.scale ** 0.5 + 1e-8)


class VariationalNormal(torch.nn.Module, torch.distributions.Distribution):
    def __init__(self, loc, scale):
        torch.nn.Module.__init__(self)

        assert loc.shape == scale.shape

        self.loc = torch.nn.Parameter(loc)
        self.logscale = torch.nn.Parameter(torch.log(torch.exp(scale)-1))

        torch.distributions.Distribution.__init__(self,batch_shape=self.loc.shape)

    def dist(self):
        return Normal(self.loc, F.softplus(self.logscale))

    def rsample(self, sample_shape):
        shape = self._extended_shape(sample_shape)
        self.eps = _standard_normal(shape, dtype=self.loc.dtype, device=self.loc.device)
        samples = self.loc + self.eps * F.softplus(self.logscale)
        return samples
     

class MC_ExpansionLayer(torch.nn.Module):
    def __init__(self, num_MC=1, input_dim=2):
        '''
        :param num_MC: if input.dim()==input_dim, expand first dimension by num_MC
        :param input_dim: number of input dimensions, if input.dim()=input_dim then add and expand 0th dimension
        '''
        super().__init__()

        self.num_MC = num_MC
        self.input_dim = input_dim

    def forward(self, x):
        if x.dim() == self.input_dim:
            out = x.unsqueeze(0).repeat(self.num_MC, *(x.dim() * (1,)))
        elif x.dim() == self.input_dim + 1:
            out = x
        else:
            raise ValueError(f"Input.dim()={x.dim()}, but should be either {self.input_dim} and expanded or {self.input_dim+1}")
        return out


class BayesLinear(torch.nn.Module):
    """A Bayesian linear layer."""
    def __init__(self, in_features, out_features, num_MC=None, prior=1., bias=True):
        super().__init__()

        self.dim_input = in_features
        self.dim_output = out_features
        self.num_MC = num_MC

        self.mu_init_std = torch.sqrt(torch.scalar_tensor(2 / (in_features + out_features)))
        self.logsigma_init_std = 0.001

        self.weight = VariationalNormal(torch.FloatTensor(in_features, out_features).normal_(0., self.mu_init_std),
                                        torch.FloatTensor(in_features, out_features).fill_(self.logsigma_init_std))

        if bias:
            self.bias = VariationalNormal(torch.FloatTensor(out_features).normal_(0., self.mu_init_std),
                                          torch.FloatTensor(out_features).fill_(self.logsigma_init_std))
        else:
            self.bias = None

        if prior == 'laplace':
            self.prior = LaplacePrior(module=self)
        elif prior == 'laplace_clamp':
            self.prior = LaplacePrior(module=self, clamp=True)
        elif isinstance(float(prior), Number):
            self.prior = GaussianPrior(scale=prior)
        else:
            exit('Wrong Prior ... should be in [1.0, "laplace"]')

        self.reset_parameters(scale_offset=0)

    def reset_parameters(self, scale_offset=0):
        torch.nn.init.kaiming_uniform_(self.weight.loc.data, a=math.sqrt(5))
        self.weight.logscale.data.fill_(torch.log(torch.exp((self.mu_init_std)/self.weight.loc.shape[1] )-1)+scale_offset)

        if self.bias is not None:
            fan_in, _ = torch.nn.init._calculate_fan_in_and_fan_out(self.weight.loc.data)
            bound = 1 / math.sqrt(fan_in)
            torch.nn.init.uniform_(self.bias.loc.data, -bound, bound)
            self.bias.logscale.data.fill_(torch.log(torch.exp(self.mu_init_std)-1)+scale_offset)

    def forward(self, x: torch.Tensor, prior=None, stochastic=True):
        '''
        :param x: 3 dimensional tensor of shape [N_MC, M_BatchSize, d_Input]
        :return:
        '''

        assert x.dim() == 3, f"Input tensor not of shape [N_MC, BatchSize, Features] but is {x.shape=}"

        num_MC = x.shape[0]
        bs = x.shape[1]

        forward = ['reparam', 'local_reparam'][0]

        if forward == 'reparam':
            self.sampled_w = self.weight.rsample((num_MC,))

            if self.bias is not None:
                self.sampled_b = self.bias.rsample((num_MC,))
                out = torch.baddbmm(self.sampled_b.unsqueeze(1), x, self.sampled_w)
            else:
                out = torch.bmm(x, self.sampled_w)
        elif forward == 'local_reparam':
            w_sigma = F.softplus(self.weight_logscale)
            mean = torch.matmul(x, self.weight_loc) + self.bias_loc
            std = torch.sqrt(torch.matmul(x.pow(2), F.softplus(self.weight_logscale).pow(2)) + F.softplus(self.bias_logscale).pow(2))
            epsilon = torch.FloatTensor(x.shape[0], x.shape[1], self.dim_output).normal_(0., self.epsilon_sigma)
            out = mean + epsilon * std

        self.kl_div = torch.distributions.kl_divergence(self.weight.dist(), self.prior.dist()).sum()
        self.entropy = self.weight.dist().entropy().sum()

        return out

    def __repr__(self):
        return f'BayesLinear(in_features={self.dim_input}, out_features={self.dim_output}'


def sklearn_mlp(cfg):
    net = MLPRegressor(
        **cfg,
        # hidden_layer_sizes=[64, 64],
        # max_iter=3000,
        # alpha=1e-3,
        # learning_rate_init=1e-4,
        # batch_size=32,
    )
    return net


def sklearn_train(X_train, y_train, X_test, y_test):
    """Trains a scikit-learn model given a configuration."""
    n_models = 10

    if n_models > 0:
        nets = [sklearn_mlp() for _ in range(n_models)]

        for i in range(n_models):
            nets[i] = nets[i].fit(X_train, y_train)
        
        preds = [nets[i].predict(X_train).reshape(-1, 1) for i in range(n_models)]
        preds = np.hstack(preds)
        train_preds_mean = preds.mean(axis=1)
        train_preds_std = 2 * preds.std(axis=1)

        preds = [nets[i].predict(X_test).reshape(-1, 1) for i in range(n_models)]
        preds = np.hstack(preds)

        test_preds_mean = preds.mean(axis=1)
        test_preds_std = 2 * preds.std(axis=1)
    else:
        net = sklearn_mlp()
        net = net.fit(X_train, y_train)
        train_pred = net.predict(X_train)
        test_pred = net.predict(X_test)
    
    fig, ax = plt.subplots(ncols=2)

    label = 'Prediction'
    if n_models > 0:
        label += f' (\(\mu\); \(k = {n_models}\))'

    ax[0].plot(y_train, linewidth=1, label='Ground Truth')
    ax[0].plot(train_preds_mean if n_models > 0 else train_pred,
               linewidth=1,
               label=label)
    if n_models > 0:
        ax[0].fill_between(np.arange(train_preds_mean.shape[0]),
                           train_preds_mean - train_preds_std,
                           train_preds_mean + train_preds_std,
                           alpha=0.3,
                           color='g',
                           label='\(2 \sigma\)')
    ax[0].set_title('Training Set')
    ax[0].legend()

    ax[1].plot(y_test, linewidth=1, label='Ground Truth')
    ax[1].plot(test_preds_mean if n_models > 0 else test_pred,
               linewidth=1,
               label=label)
    if n_models > 0:
        ax[1].fill_between(np.arange(test_preds_mean.shape[0]),
                           test_preds_mean - test_preds_std,
                           test_preds_mean + test_preds_std,
                           alpha=0.3,
                           color='g',
                           label='\(2 \sigma\)')
    ax[1].set_title('Test Set')
    ax[1].legend()

    plt.show()


def pytorch_train(
        X_train, y_train,
        X_test, y_test,
        cfg,
    ):
    """Trains a torch model given a training configuration."""
    model_type = cfg['model_type']

    if model_type == 'ensemble':
        torch_model = TorchEnsemble([
            MLPModel(
                cfg['seq_len'],
                cfg['hidden_size'],
                cfg['num_layers'],
            ) for _ in range(cfg['num_models'])
        ])
    elif model_type == 'bayes_mlp':
        torch_model = BayesMLPModel(
            cfg['seq_len'],
            cfg['hidden_size'],
            cfg['num_layers'],
            num_MC=cfg['num_models'],
        )
    elif model_type == 'mlp':
        torch_model = MLPModel(
            cfg['seq_len'],
            cfg['hidden_size'],
            cfg['num_layers'],
        )

    criterion = nn.MSELoss()

    X_train = torch.from_numpy(X_train)
    X_test = torch.from_numpy(X_test)
    y_train = torch.from_numpy(y_train)
    y_test = torch.from_numpy(y_test)

    train_set = TensorDataset(X_train, y_train)
    test_set = TensorDataset(X_test, y_test)

    train_loader = DataLoader(
        train_set, batch_size=cfg['batch_size'],
    )
    test_loader = DataLoader(
        test_set, batch_size=cfg['batch_size'],
    )

    trainer = pl.Trainer(
        max_epochs=cfg['num_epochs'],
    )

    pl_model = Model(
        torch_model,
        criterion,
        cfg['lr'],
        alpha=cfg['l2_alpha'],
    )
    trainer.fit(pl_model,
                train_dataloaders=train_loader,
                val_dataloaders=test_loader)

    if cfg['return_plot']:
        pl_model.eval()

        with torch.no_grad():
            train_pred_mean, train_pred_std = pl_model(X_train)
            train_pred_mean = train_pred_mean.view(-1)
            train_pred_std = train_pred_std.view(-1)

            test_pred_mean, test_pred_std = pl_model(X_test)
            test_pred_mean = test_pred_mean.view(-1)
            test_pred_std = test_pred_std.view(-1)

        fig, ax = plt.subplots(ncols=2)

        ax[0].plot(y_train, label='Ground Truth')
        ax[0].plot(train_pred_mean, label='Prediction')
        ax[0].fill_between(
            list(range(1, len(train_pred_mean) + 1)),
            train_pred_mean - train_pred_std,
            train_pred_mean + train_pred_std,
            alpha=0.3,
        )
        ax[0].set_title('Training Set')
        ax[0].legend()

        ax[1].plot(y_test, label='Ground Truth')
        ax[1].plot(test_pred_mean, label='Prediction')
        ax[1].fill_between(
            list(range(1, len(test_pred_mean) + 1)),
            test_pred_mean - test_pred_std,
            test_pred_mean + test_pred_std,
            alpha=0.3,
        )
        ax[1].set_title('Test Set')
        ax[1].legend()

        return fig, ax
