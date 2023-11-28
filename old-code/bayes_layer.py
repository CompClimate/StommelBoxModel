import math
from numbers import Number

import torch
import torch.nn as nn
import torch.nn.functional as F
from captum.attr._utils.lrp_rules import EpsilonRule
from torch.distributions import Normal
from torch.distributions.utils import _standard_normal

EPS = torch.finfo(torch.float32).eps
# EPS = 1e-10


class GaussianPrior:
    """Implements a Gaussian prior."""

    def __init__(self, scale):
        assert scale > 0.0
        self.scale = scale

    def dist(self):
        return Normal(0, self.scale)


class LaplacePrior(nn.Module):
    """Implements a Laplacian prior."""

    def __init__(self, module, clamp=False):
        super().__init__()

        self.rule = EpsilonRule()

        self.scale = torch.ones_like(module.weight.loc.data)
        module.weight.loc.register_hook(self._save_grad)

        self.clamp = clamp
        self.step = 0
        self.beta = 0.99

    def _save_grad(self, grad):
        self.step += 1
        bias_correction = 1 - self.beta**self.step
        self.scale.mul_(self.beta).add_(
            alpha=1 - self.beta, other=(1 / grad.data**2).div_(bias_correction + 1e-8)
        )

    def dist(self):
        if self.clamp:
            return Normal(0, torch.clamp(self.scale**0.5, min=1.0))
        else:
            return Normal(0, self.scale**0.5 + 1e-8)


class VariationalNormal(nn.Module, torch.distributions.Distribution):
    def __init__(self, loc, scale):
        nn.Module.__init__(self)

        assert loc.shape == scale.shape

        self.rule = EpsilonRule()

        self.loc = nn.Parameter(loc)
        self.logscale = nn.Parameter(torch.log(torch.exp(scale) - 1))

        torch.distributions.Distribution.__init__(self, batch_shape=self.loc.shape)

    def dist(self):
        return Normal(self.loc.clamp(min=EPS), F.softplus(self.logscale).clamp(min=EPS))

    def rsample(self, sample_shape):
        shape = self._extended_shape(sample_shape)
        self.eps = _standard_normal(shape, dtype=self.loc.dtype, device=self.loc.device)
        samples = self.loc + self.eps * F.softplus(self.logscale)
        return samples


class MC_ExpansionLayer(nn.Module):
    def __init__(self, num_MC=1, input_dim=2):
        """
        :param num_MC: if input.dim()==input_dim, expand first dimension by num_MC
        :param input_dim: number of input dimensions, if input.dim()=input_dim then add and expand 0th dimension
        """
        super().__init__()

        self.num_MC = num_MC
        self.input_dim = input_dim

        self.rule = EpsilonRule()

    def forward(self, x):
        if x.dim() == self.input_dim:
            out = x.unsqueeze(0).repeat(self.num_MC, *(x.dim() * (1,)))
        elif x.dim() == self.input_dim + 1:
            out = x
        else:
            raise ValueError(
                f"Input.dim()={x.dim()}, but should be either {self.input_dim} and expanded or {self.input_dim+1}"
            )
        return out


class BayesLinear(nn.Module):
    """A Bayesian linear layer."""

    def __init__(self, in_features, out_features, num_MC=None, prior=1.0, bias=True):
        super().__init__()

        self.dim_input = in_features
        self.dim_output = out_features
        self.num_MC = num_MC

        self.mu_init_std = torch.sqrt(torch.scalar_tensor(2 / (in_features + out_features)))
        self.logsigma_init_std = 0.001

        self.weight = VariationalNormal(
            torch.FloatTensor(in_features, out_features).normal_(0.0, self.mu_init_std),
            torch.FloatTensor(in_features, out_features).fill_(self.logsigma_init_std),
        )

        if bias:
            self.bias = VariationalNormal(
                torch.FloatTensor(out_features).normal_(0.0, self.mu_init_std),
                torch.FloatTensor(out_features).fill_(self.logsigma_init_std),
            )
        else:
            self.bias = None

        if prior == "laplace":
            self.prior = LaplacePrior(module=self)
        elif prior == "laplace_clamp":
            self.prior = LaplacePrior(module=self, clamp=True)
        elif isinstance(float(prior), Number):
            self.prior = GaussianPrior(scale=prior)
        else:
            exit('Wrong Prior ... should be in [1.0, "laplace"]')

        self.reset_parameters(scale_offset=0)

    def reset_parameters(self, scale_offset=0):
        nn.init.kaiming_uniform_(self.weight.loc.data, a=math.sqrt(5))
        self.weight.logscale.data.fill_(
            torch.log(torch.exp((self.mu_init_std) / self.weight.loc.shape[1]) - 1) + scale_offset
        )

        if self.bias is not None:
            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight.loc.data)
            bound = 1 / math.sqrt(fan_in)
            nn.init.uniform_(self.bias.loc.data, -bound, bound)
            self.bias.logscale.data.fill_(
                torch.log(torch.exp(self.mu_init_std) - 1) + scale_offset
            )

    def forward(self, x: torch.Tensor, prior=None, stochastic=True):
        """
        :param x: 3 dimensional tensor of shape [N_MC, M_BatchSize, d_Input]
        :return:
        """

        assert (
            x.dim() == 3
        ), f"Input tensor not of shape [N_MC, BatchSize, Features] but is {x.shape=}"

        num_MC = x.shape[0]
        bs = x.shape[1]

        forward = ["reparam", "local_reparam"][0]

        if forward == "reparam":
            self.sampled_w = self.weight.rsample((num_MC,))

            if self.bias is not None:
                self.sampled_b = self.bias.rsample((num_MC,))
                out = torch.baddbmm(self.sampled_b.unsqueeze(1), x, self.sampled_w)
            else:
                out = torch.bmm(x, self.sampled_w)
        elif forward == "local_reparam":
            w_sigma = F.softplus(self.weight_logscale)
            mean = torch.matmul(x, self.weight_loc) + self.bias_loc
            std = torch.sqrt(
                torch.matmul(x.pow(2), F.softplus(self.weight_logscale).pow(2))
                + F.softplus(self.bias_logscale).pow(2)
            )
            epsilon = torch.FloatTensor(x.shape[0], x.shape[1], self.dim_output).normal_(
                0.0, self.epsilon_sigma
            )
            out = mean + epsilon * std

        self.kl_div = torch.distributions.kl_divergence(
            self.weight.dist(), self.prior.dist()
        ).sum()
        self.entropy = self.weight.dist().entropy().sum()

        return out

    def __repr__(self):
        return f"BayesLinear(in_features={self.dim_input}, out_features={self.dim_output}"
