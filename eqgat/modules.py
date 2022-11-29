import torch.nn as nn
import torch
from torch import Tensor
import math
from typing import Tuple
from torch_geometric.nn.inits import reset
import torch.nn.functional as F

from torch_scatter import scatter_mean

from typing import Callable, Union, Optional

from torch.nn.init import kaiming_uniform_
from torch.nn.init import zeros_


class DenseLayer(nn.Linear):
    def __init__(
        self,
        in_features: int,
        out_features: int,
        bias: bool = True,
        activation: Union[Callable, nn.Module] = None,
        weight_init: Callable = kaiming_uniform_,
        bias_init: Callable = zeros_,
    ):
        self.weight_init = weight_init
        self.bias_init = bias_init
        super(DenseLayer, self).__init__(in_features, out_features, bias)

        self.activation = activation
        if self.activation is None:
            self.activation = nn.Identity()

    def reset_parameters(self):
        # https://github.com/pytorch/pytorch/blob/master/torch/nn/modules/linear.py#L106
        self.weight_init(self.weight, a=math.sqrt(5))
        if self.bias is not None:
            self.bias_init(self.bias)

    def forward(self, input: torch.Tensor):
        y = F.linear(input, self.weight, self.bias)
        y = self.activation(y)
        return y


class CosineCutoff(nn.Module):
    def __init__(self, cutoff):
        super(CosineCutoff, self).__init__()
        self.cutoff = cutoff

    @staticmethod
    def cosine_cutoff(r: Tensor, rcut: float) -> Tensor:
        out = 0.5 * (torch.cos((math.pi * r) / rcut) + 1.0)
        out = out * (r < rcut).float()
        return out

    def forward(self, r):
        return self.cosine_cutoff(r, self.cutoff)

    def __repr__(self):
        return f"{self.__class__.__name__}(cutoff={self.cutoff})"


class PolynomialCutoff(nn.Module):
    def __init__(self, cutoff, p: int = 6):
        super(PolynomialCutoff, self).__init__()
        self.cutoff = cutoff
        self.p = p

    @staticmethod
    def polynomial_cutoff(
        r: Tensor,
        rcut: float,
        p: float = 6.0
    ) -> Tensor:
        """
        Polynomial cutoff, as proposed in DimeNet: https://arxiv.org/abs/2003.03123
        """
        if not p >= 2.0:
            print(f"Exponent p={p} has to be >= 2.")
            print("Exiting code.")
            exit()

        rscaled = r / rcut

        out = 1.0
        out = out - (((p + 1.0) * (p + 2.0) / 2.0) * torch.pow(rscaled, p))
        out = out + (p * (p + 2.0) * torch.pow(rscaled, p + 1.0))
        out = out - ((p * (p + 1.0) / 2) * torch.pow(rscaled, p + 2.0))

        return out * (rscaled < 1.0).float()

    def forward(self, r):
        return self.polynomial_cutoff(r=r, rcut=self.cutoff, p=self.p)

    def __repr__(self):
        return f"{self.__class__.__name__}(cutoff={self.cutoff}, p={self.p})"


class ChebyshevExpansion(nn.Module):
    def __init__(self, max_value: float, K: int = 20):
        super(ChebyshevExpansion, self).__init__()
        self.K = K
        self.max_value = max_value
        self.shift_scale = lambda x: 2 * x / max_value - 1.0

    @staticmethod
    def chebyshev_recursion(x, n):
        if x.ndim == 1:
            x = x.unsqueeze(-1)
        if not n > 2:
            print(f"Naural exponent n={n} has to be > 2.")
            print("Exiting code.")
            exit()

        t_n_1 = torch.ones_like(x)
        t_n = x
        ts = [t_n_1, t_n]
        for i in range(n - 2):
            t_n_new = 2 * x * t_n - t_n_1
            t_n_1 = t_n
            t_n = t_n_new
            ts.append(t_n_new)
        return torch.cat(ts, dim=-1)

    def forward(self, x: Tensor) -> Tensor:
        x = self.shift_scale(x)
        x = self.chebyshev_recursion(x, n=self.K)
        return x

    def __repr__(self):
        return f"{self.__class__.__name__}(K={self.K}, max_value={self.max_value})"


def gaussian_basis_expansion(inputs: Tensor, offsets: Tensor, widths: Tensor):
    coeff = -0.5 / torch.pow(widths, 2)
    diff = inputs[..., None] - offsets
    y = torch.exp(coeff * torch.pow(diff, 2))
    return y


class GaussianExpansion(nn.Module):

    def __init__(
        self, max_value: float, K: int, start: float = 0.0, trainable: bool = False
    ):
        super(GaussianExpansion, self).__init__()
        self.K = K

        offset = torch.linspace(start, max_value, K)
        widths = torch.FloatTensor(
            torch.abs(offset[1] - offset[0]) * torch.ones_like(offset)
        )
        if trainable:
            self.widths = nn.Parameter(widths)
            self.offsets = nn.Parameter(offset)
        else:
            self.register_buffer("widths", widths)
            self.register_buffer("offsets", offset)

    def forward(self, inputs: Tensor):
        return gaussian_basis_expansion(inputs, self.offsets, self.widths)


class BesselExpansion(nn.Module):
    def __init__(
        self, max_value: float, K: int = 20
    ):
        super(BesselExpansion, self).__init__()
        self.max_value = max_value
        self.K = K
        frequency = math.pi * torch.arange(start=1, end=K + 1)
        self.register_buffer("frequency", frequency)
        self.reset_parameters()

    def reset_parameters(self):
        self.frequency.data = math.pi * torch.arange(start=1, end=self.K + 1)

    def forward(self, x: Tensor) -> Tensor:
        """
        Bessel RBF, as proposed in DimeNet: https://arxiv.org/abs/2003.03123
        """
        ax = x.unsqueeze(-1) / self.max_value
        ax = ax * self.frequency
        sinax = torch.sin(ax)
        norm = torch.where(
            x == 0.0, torch.tensor(1.0, dtype=x.dtype, device=x.device), x
        )
        out = sinax / norm[..., None]
        out *= math.sqrt(2 / self.max_value)
        return out


class GatedEquivBlock(nn.Module):
    def __init__(
        self,
        in_dims: Tuple[int, int],
        out_dims: Tuple[int, Optional[int]],
        hs_dim: Optional[int] = None,
        hv_dim: Optional[int] = None,
        norm_eps: float = 1e-6,
        use_mlp: bool = False
    ):
        super(GatedEquivBlock, self).__init__()
        self.si, self.vi = in_dims
        self.so, self.vo = out_dims
        self.vo = 0 if self.vo is None else self.vo

        self.hs_dim = hs_dim or max(self.si, self.so)
        self.hv_dim = hv_dim or max(self.vi, self.vo)
        self.norm_eps = norm_eps

        self.use_mlp = use_mlp

        self.Wv0 = DenseLayer(self.vi, self.hv_dim + self.vo, bias=False)

        if not use_mlp:
            self.Ws = DenseLayer(self.hv_dim + self.si, self.vo + self.so, bias=True)
        else:
            self.Ws = nn.Sequential(
                DenseLayer(self.hv_dim + self.si, self.si, bias=True, activation=nn.SiLU()),
                DenseLayer(self.si, self.vo + self.so, bias=True)
            )
            self.Wv1 = DenseLayer(self.vo, self.vo, bias=False)

        self.reset_parameters()

    def reset_parameters(self):
        reset(self.Ws)
        reset(self.Wv0)
        if self.use_mlp:
            reset(self.Wv1)


    def forward(self, x: Tuple[Tensor, Tensor]) -> Tuple[Tensor, Tensor]:

        s, v = x
        vv = self.Wv0(v)

        if self.vo > 0:
            vnorm, v = vv.split([self.hv_dim, self.vo], dim=-1)
        else:
            vnorm = vv

        vnorm = torch.clamp(torch.pow(vnorm, 2).sum(dim=1), min=self.norm_eps).sqrt()
        s = torch.cat([s, vnorm], dim=-1)
        s = self.Ws(s)
        if self.vo > 0:
            gate, s = s.split([self.vo, self.so], dim=-1)
            v = gate.unsqueeze(1) * v
            if self.use_mlp:
                v = self.Wv1(v)

        return s, v


class GatedEquivBlockTP(nn.Module):
    def __init__(
        self,
        in_dims: Tuple[int, int],
        out_dims: Tuple[int, Optional[int]],
        hs_dim: Optional[int] = None,
        hv_dim: Optional[int] = None,
        norm_eps: float = 1e-6,
        use_mlp: bool = False
    ):
        super(GatedEquivBlockTP, self).__init__()
        self.si, self.vi = in_dims
        self.so, self.vo = out_dims
        self.vo = 0 if self.vo is None else self.vo

        self.hs_dim = hs_dim or max(self.si, self.so)
        self.hv_dim = hv_dim or max(self.vi, self.vo)
        self.norm_eps = norm_eps

        self.use_mlp = use_mlp

        self.scalar_mixing = nn.Parameter(torch.zeros(size=(1, 1, self.si, self.vi)))
        self.vector_mixing = nn.Parameter(torch.zeros(size=(1, 1, self.si, self.vi)))

        if not use_mlp:
            self.Ws = DenseLayer(self.si, self.vo + self.so, bias=True)
        else:
            self.Ws = nn.Sequential(
                DenseLayer(self.si, self.si, bias=True, activation=nn.SiLU()),
                DenseLayer(self.si, self.vo + self.so, bias=True)
            )
            self.Wv = DenseLayer(self.vo, self.vo, bias=False)

        self.reset_parameters()

    def reset_parameters(self):
        reset(self.Ws)
        if self.use_mlp:
            reset(self.Wv)
        kaiming_uniform_(self.scalar_mixing)
        kaiming_uniform_(self.vector_mixing)

    def forward(self, x: Tuple[Tensor, Tensor]) -> Tuple[Tensor, Tensor]:

        s, v = x

        # tensor product of l=0 and l=1
        sv = torch.einsum('bk, bdm -> bdkm', s, v)

        # path for scalar features
        # mean aggregate along vector-channels
        s = torch.sum(sv * self.scalar_mixing, dim=-1)
        # make SO(3) invariant
        s = torch.pow(s, 2).sum(dim=1)
        s = torch.clamp(s, min=self.norm_eps).sqrt()
        # feed into invariant MLP / linear net
        s = self.Ws(s)

        if self.vo > 0:
            # path for vector features
            # mean aggregate along scalar-channels
            v = torch.sum(self.vector_mixing * sv, dim=-2)
            gate, s = s.split([self.vo, self.so], dim=-1)
            v = gate.unsqueeze(1) * v
            if self.use_mlp:
                v = self.Wv(v)

        return s, v


class BatchNorm(nn.Module):
    def __init__(self, dims: Tuple[int, Optional[int]], eps: float = 1e-6, affine: bool = True):
        super().__init__()

        self.dims = dims
        self.sdim, self.vdim = dims
        self.eps = eps
        self.affine = affine
        if affine:
            self.weight = nn.Parameter(torch.Tensor(self.sdim))
            self.bias = nn.Parameter(torch.Tensor(self.sdim))
        else:
            self.register_parameter('weight', None)
            self.register_parameter('bias', None)

        self.reset_parameters()

    def reset_parameters(self):
        if self.affine:
            self.weight.data.fill_(1.0)
            self.bias.data.fill_(0.0)

    def forward(self, x: Tuple[Tensor, Optional[Tensor]], batch: Tensor) -> Tuple[Tensor, Optional[Tensor]]:
        s, v = x
        batch_size = int(batch.max()) + 1

        smean = scatter_mean(s, batch, dim=0, dim_size=batch_size)

        if s.device == "cpu":
            smean = smean.index_select(0, batch)
        else:
            smean = torch.gather(smean, dim=0, index=batch.view(-1, 1))

        s = s - smean

        var = scatter_mean(s * s, batch, dim=0, dim_size=batch_size)
        var = torch.clamp(var, min=self.eps)
        prec = torch.pow(torch.sqrt(var), -1)
        if prec.device == "cpu":
            prec = prec.index_select(0, batch)
        else:
            prec = torch.gather(prec, dim=0, index=batch.view(-1, 1))

        sout = s * prec

        if self.weight is not None and self.bias is not None:
            sout = sout * self.weight + self.bias

        if v is not None:
            vmean = torch.pow(v, 2).sum(-1, keepdim=True)
            vmean = scatter_mean(vmean, batch, dim=0, dim_size=batch_size)
            vmean = torch.clamp(vmean, min=self.eps)
            if vmean.device == "cpu":
                vmean = vmean.index_select(0, batch)
            else:
                vmean = torch.gather(vmean, dim=0, index=batch.view(-1, 1, 1))

            vout = v / vmean
        else:
            vout = None

        out = sout, vout

        return out

    def __repr__(self):
        return (f'{self.__class__.__name__}(dims={self.dims}, '
                f'affine={self.affine})')



class LayerNorm(nn.Module):
    def __init__(self, dims: Tuple[int, Optional[int]], eps: float = 1e-6, affine: bool = True):
        super().__init__()

        self.dims = dims
        self.sdim, self.vdim = dims
        self.eps = eps
        self.affine = affine
        if affine:
            self.weight = nn.Parameter(torch.Tensor(self.sdim))
            self.bias = nn.Parameter(torch.Tensor(self.sdim))
        else:
            self.register_parameter('weight', None)
            self.register_parameter('bias', None)

        self.reset_parameters()

    def reset_parameters(self):
        if self.affine:
            self.weight.data.fill_(1.0)
            self.bias.data.fill_(0.0)

    def forward(self, x: Tuple[Tensor, Optional[Tensor]], batch: Tensor) -> Tuple[Tensor, Optional[Tensor]]:
        s, v = x
        batch_size = int(batch.max()) + 1
        smean = s.mean(dim=-1, keepdim=True)
        smean = scatter_mean(smean, batch, dim=0, dim_size=batch_size)

        if s.device == "cpu":
            smean = smean.index_select(0, batch)
        else:
            smean = torch.gather(smean, dim=0, index=batch.view(-1, 1))

        s = s - smean

        var = (s * s).mean(dim=-1, keepdim=True)
        var = scatter_mean(var, batch, dim=0, dim_size=batch_size)
        var = torch.clamp(var, min=self.eps)
        prec = 1 / var

        if prec.device == "cpu":
            prec = prec.index_select(0, batch)
        else:
            prec = torch.gather(prec, dim=0, index=batch.view(-1, 1))

        sout = s * prec

        if self.weight is not None and self.bias is not None:
            sout = sout * self.weight + self.bias

        if v is not None:
            vmean = torch.pow(v, 2).sum(-1, keepdim=True).mean(dim=-1, keepdim=True)
            vmean = scatter_mean(vmean, batch, dim=0, dim_size=batch_size)
            vmean = torch.clamp(vmean, min=self.eps)
            if vmean.device == "cpu":
                vmean = vmean.index_select(0, batch)
            else:
                vmean = torch.gather(vmean, dim=0, index=batch.view(-1, 1, 1))

            vout = v / vmean
        else:
            vout = None

        out = sout, vout

        return out

    def __repr__(self):
        return (f'{self.__class__.__name__}(dims={self.dims}, '
                f'affine={self.affine})')


class ShiftedSoftPlus(nn.Module):
    def __init__(self):
        super(ShiftedSoftPlus, self).__init__()

    def forward(self, x: Tensor):
        return F.softplus(x) - math.log(2.0)



def visualize_basis_expansions():
    import matplotlib.pyplot as plt

    dist = torch.linspace(0, 5.0, 1000)

    K = 32
    gauss_rbf = GaussianExpansion(max_value=5.0, K=K)
    cheb_rbf = ChebyshevExpansion(max_value=5.0, K=K)
    bessel_rbf = BesselExpansion(max_value=5.0, K=K)

    gauss = gauss_rbf(dist)
    cheb = cheb_rbf(dist)
    bessel = bessel_rbf(dist)

    show = "gaussian"

    if show == "gaussian":
        plt.plot(dist, gauss)
    elif show == "cheb":
        plt.plot(dist, cheb)
    elif show == "bessel":
        plt.plot(dist, bessel)
    plt.xlabel("Distance")
    plt.ylabel("RBF Values")
    plt.title(f"{show} RBF")
    plt.show()