from typing import Optional, Tuple, Callable

import torch
from torch import Tensor, nn
from torch_geometric.nn import MessagePassing
from torch_geometric.nn.inits import reset
from torch_scatter import scatter
from torch_geometric.utils.num_nodes import maybe_num_nodes
from torch_scatter.composite import scatter_softmax
import torch.nn.functional as F

from eqgat.modules import (BesselExpansion, ChebyshevExpansion, GaussianExpansion,\
    DenseLayer, GatedEquivBlock, PolynomialCutoff, GatedEquivBlockTP)


def degree_normalization(edge_index,
                         num_nodes: Optional[int] = None,
                         flow: str = "source_to_target") -> Tensor:

    if flow not in ["source_to_target", "target_to_source"]:
        print(f"Wrong selected flow {flow}.")
        print("Only 'source_to_target', or 'target_to_source' is possible")
        print("Exiting code")
        exit()

    num_nodes = maybe_num_nodes(edge_index, num_nodes)
    edge_weight = torch.ones((edge_index.size(1),), dtype=torch.get_default_dtype(),
                             device=edge_index.device)

    row, col = edge_index[0], edge_index[1]
    idx = col if flow == "source_to_target" else row
    deg = scatter(edge_weight, idx, dim=0, dim_size=num_nodes, reduce="add")
    deg = torch.clamp(deg, min=1.0)
    deg_inv_sqrt = deg.pow_(-0.5)
    norm = deg_inv_sqrt[row] * edge_weight * deg_inv_sqrt[col]
    return norm


def sqrt_normalization(edge_index,
                       num_nodes: Optional[int] = None,
                       flow: str = "source_to_target") -> Tensor:

    if flow not in ["source_to_target", "target_to_source"]:
        print(f"Wrong selected flow {flow}.")
        print("Only 'source_to_target', or 'target_to_source' is possible")
        print("Exiting code")
        exit()

    num_nodes = maybe_num_nodes(edge_index, num_nodes)
    edge_weight = torch.ones((edge_index.size(1),), dtype=torch.get_default_dtype(),
                             device=edge_index.device)

    row, col = edge_index[0], edge_index[1]
    idx = col if flow == "source_to_target" else row
    deg = scatter(edge_weight, idx, dim=0, dim_size=num_nodes, reduce="add")
    deg = torch.clamp(deg, min=1.0)
    deg_inv_sqrt = deg.pow_(-0.5)
    norm = edge_weight * deg_inv_sqrt[col]
    return norm


def scatter_normalization(
    x: Tensor,
    index: Tensor,
    dim: int = 0,
    act: Callable = nn.Softplus(),
    eps: float = 1e-6,
    dim_size: Optional[int] = None,
):
    xa = act(x)
    aggr_logits = scatter(src=xa, index=index, dim=dim, reduce="add", dim_size=dim_size)
    aggr_logits = aggr_logits[index]
    xa = xa / (aggr_logits + eps)
    return xa


class EQGATConv(MessagePassing):
    def __init__(
        self,
        in_dims: Tuple[int, Optional[int]],
        out_dims: Tuple[int, Optional[int]],
        num_radial: int,
        cutoff: float,
        eps: float = 1e-6,
        has_v_in: bool = True,
        basis: str = "bessel",
        use_mlp_update: bool = True,
        vector_aggr: str = "mean",
    ):
        super(EQGATConv, self).__init__(
            node_dim=0, aggr=None, flow="source_to_target"
        )

        self.vector_aggr = vector_aggr
        self.in_dims = in_dims
        self.si, self.vi = in_dims
        self.out_dims = out_dims
        self.so, self.vo = out_dims
        self.has_v_in = has_v_in
        if has_v_in:
            self.vector_net = DenseLayer(self.vi, self.vi, bias=False)
            self.v_mul = 3
        else:
            self.v_mul = 1
            self.vector_net = nn.Identity()

        self.cutoff = cutoff
        self.num_radial = num_radial

        if basis == "chebyshev":
            rbf = ChebyshevExpansion
        elif basis == "bessel":
            rbf = BesselExpansion
        elif basis == "gaussian":
            rbf = GaussianExpansion
        else:
            raise ValueError

        self.distance_expansion = rbf(
            max_value=cutoff,
            K=num_radial,
        )
        self.cutoff_fnc = PolynomialCutoff(cutoff, p=6)
        self.edge_net = nn.Sequential(DenseLayer(2 * self.si + self.num_radial,
                                                 self.si,
                                                 bias=True, activation=nn.SiLU()),
                                      DenseLayer(self.si, self.v_mul * self.vi + self.si,
                                                 bias=True)
                                      )
        self.scalar_net = DenseLayer(self.si, self.si, bias=True)
        self.update_net = GatedEquivBlock(in_dims=(self.si, self.vi),
                                          hs_dim=self.si, hv_dim=self.vi,
                                          out_dims=(self.si, self.vi),
                                          norm_eps=eps,
                                          use_mlp=use_mlp_update)
        self.reset_parameters()

    def reset_parameters(self):
        reset(self.edge_net)
        reset(self.scalar_net)
        reset(self.update_net)

    def forward(
        self,
        x: Tuple[Tensor, Tensor],
        edge_index: Tensor,
        edge_attr: Tuple[Tensor, Tensor],
    ):

        s, v = x
        d, r = edge_attr

        ms, mv = self.propagate(
            sa=s,
            sb=self.scalar_net(s),
            va=v,
            vb=self.vector_net(v),
            edge_attr=(d, r),
            edge_index=edge_index,
            dim_size=s.size(0),
        )

        s = ms + s
        v = mv + v

        ms, mv = self.update_net(x=(s, v))

        s = ms + s
        v = mv + v

        return s, v

    def aggregate(
        self,
            inputs: Tuple[Tensor, Tensor],
            index: Tensor,
            dim_size: Optional[int] = None
    ) -> Tuple[Tensor, Tensor]:
        s = scatter(inputs[0], index=index, dim=0, reduce="add", dim_size=dim_size)
        v = scatter(inputs[1], index=index, dim=0, reduce=self.vector_aggr, dim_size=dim_size)
        return s, v

    def message(
        self,
        sa_i: Tensor,
        sa_j: Tensor,
        sb_j: Tensor,
        va_i: Tensor,
        va_j: Tensor,
        vb_j: Tensor,
        index: Tensor,
        edge_attr: Tuple[Tensor, Tensor],
        dim_size: Optional[int]
    ) -> Tuple[Tensor, Tensor]:

        d, r = edge_attr

        de = self.distance_expansion(d)
        dc = self.cutoff_fnc(d)
        de = dc.view(-1, 1) * de

        aij = torch.cat([sa_i, sa_j, de], dim=-1)
        aij = self.edge_net(aij)

        if self.has_v_in:
            aij, vij0 = aij.split([self.si, 3*self.vi], dim=-1)
            vij0 = vij0.unsqueeze(1)
            vij0, vij1, vij2 = vij0.chunk(3, dim=-1)
        else:
            aij, vij0 = aij.split([self.si, self.vi], dim=-1)
            vij0 = vij0.unsqueeze(1)

        # feature attention
        aij = scatter_softmax(aij, index=index, dim=0, dim_size=dim_size)
        ns_j = aij * sb_j
        nv0_j = r.unsqueeze(-1) * vij0

        if self.has_v_in:
            nv1_j = vij1 * vb_j
            v_ij_cross = torch.linalg.cross(va_i, vb_j, dim=1)
            nv2_j = vij2 * v_ij_cross
            nv_j = nv0_j + nv1_j + nv2_j
        else:
            nv_j = nv0_j

        return ns_j, nv_j


class EQGATNoFeatAttnConv(MessagePassing):
    def __init__(
        self,
        in_dims: Tuple[int, Optional[int]],
        out_dims: Tuple[int, Optional[int]],
        num_radial: int,
        cutoff: float,
        eps: float = 1e-6,
        has_v_in: bool = True,
        basis: str = "bessel",
        use_mlp_update: bool = True,
        vector_aggr: str = "mean",
    ):
        super(EQGATNoFeatAttnConv, self).__init__(
            node_dim=0, aggr=None, flow="source_to_target"
        )

        self.vector_aggr = vector_aggr
        self.in_dims = in_dims
        self.si, self.vi = in_dims
        self.out_dims = out_dims
        self.so, self.vo = out_dims
        self.has_v_in = has_v_in
        if has_v_in:
            self.vector_net = DenseLayer(self.vi, self.vi, bias=False)
            self.v_mul = 3
        else:
            self.v_mul = 1
            self.vector_net = nn.Identity()

        self.cutoff = cutoff
        self.num_radial = num_radial

        if basis == "chebyshev":
            rbf = ChebyshevExpansion
        elif basis == "bessel":
            rbf = BesselExpansion
        elif basis == "gaussian":
            rbf = GaussianExpansion
        else:
            raise ValueError

        self.distance_expansion = rbf(
            max_value=cutoff,
            K=num_radial,
        )
        self.cutoff_fnc = PolynomialCutoff(cutoff, p=6)
        self.edge_net = nn.Sequential(DenseLayer(2 * self.si + self.num_radial,
                                                 self.si,
                                                 bias=True, activation=nn.SiLU()),
                                      DenseLayer(self.si, self.v_mul * self.vi + 1,
                                                 bias=True)
                                      )
        self.scalar_net = DenseLayer(self.si, self.si, bias=True)
        self.update_net = GatedEquivBlock(in_dims=(self.si, self.vi),
                                          hs_dim=self.si, hv_dim=self.vi,
                                          out_dims=(self.si, self.vi),
                                          norm_eps=eps,
                                          use_mlp=use_mlp_update)
        self.reset_parameters()

    def reset_parameters(self):
        reset(self.edge_net)
        reset(self.scalar_net)
        reset(self.update_net)

    def forward(
        self,
        x: Tuple[Tensor, Tensor],
        edge_index: Tensor,
        edge_attr: Tuple[Tensor, Tensor],
    ):

        s, v = x
        d, r = edge_attr

        ms, mv = self.propagate(
            sa=s,
            sb=self.scalar_net(s),
            va=v,
            vb=self.vector_net(v),
            edge_attr=(d, r),
            edge_index=edge_index,
            dim_size=s.size(0),
        )

        s = ms + s
        v = mv + v

        ms, mv = self.update_net(x=(s, v))

        s = ms + s
        v = mv + v

        return s, v

    def aggregate(
        self,
            inputs: Tuple[Tensor, Tensor],
            index: Tensor,
            dim_size: Optional[int] = None
    ) -> Tuple[Tensor, Tensor]:
        s = scatter(inputs[0], index=index, dim=0, reduce="add", dim_size=dim_size)
        v = scatter(inputs[1], index=index, dim=0, reduce=self.vector_aggr, dim_size=dim_size)
        return s, v

    def message(
        self,
        sa_i: Tensor,
        sa_j: Tensor,
        sb_j: Tensor,
        va_i: Tensor,
        va_j: Tensor,
        vb_j: Tensor,
        index: Tensor,
        edge_attr: Tuple[Tensor, Tensor],
        dim_size: Optional[int]
    ) -> Tuple[Tensor, Tensor]:

        d, r = edge_attr

        de = self.distance_expansion(d)
        dc = self.cutoff_fnc(d)
        de = dc.view(-1, 1) * de

        aij = torch.cat([sa_i, sa_j, de], dim=-1)
        aij = self.edge_net(aij)

        if self.has_v_in:
            aij, vij0 = aij.split([1, 3*self.vi], dim=-1)
            vij0 = vij0.unsqueeze(1)
            vij0, vij1, vij2 = vij0.chunk(3, dim=-1)
        else:
            aij, vij0 = aij.split([1, self.vi], dim=-1)
            vij0 = vij0.unsqueeze(1)

        # feature attention
        aij = scatter_softmax(aij, index=index, dim=0, dim_size=dim_size)
        ns_j = aij * sb_j
        nv0_j = r.unsqueeze(-1) * vij0

        if self.has_v_in:
            nv1_j = vij1 * vb_j
            v_ij_cross = torch.linalg.cross(va_i, vb_j, dim=1)
            nv2_j = vij2 * v_ij_cross
            nv_j = nv0_j + nv1_j + nv2_j
        else:
            nv_j = nv0_j

        return ns_j, nv_j


class EQGATConvNoCross(MessagePassing):
    def __init__(
        self,
        in_dims: Tuple[int, Optional[int]],
        out_dims: Tuple[int, Optional[int]],
        num_radial: int,
        cutoff: float,
        eps: float = 1e-6,
        has_v_in: bool = True,
        basis: str = "bessel",
        use_mlp_update: bool = True,
        vector_aggr: str = "mean",
    ):
        super(EQGATConvNoCross, self).__init__(
            node_dim=0, aggr=None, flow="source_to_target"
        )

        self.vector_aggr = vector_aggr
        self.in_dims = in_dims
        self.si, self.vi = in_dims
        self.out_dims = out_dims
        self.so, self.vo = out_dims
        self.has_v_in = has_v_in
        if has_v_in:
            self.vector_net = DenseLayer(self.vi, self.vi, bias=False)
            self.v_mul = 2
        else:
            self.v_mul = 1
            self.vector_net = nn.Identity()

        self.cutoff = cutoff
        self.num_radial = num_radial

        if basis == "chebyshev":
            rbf = ChebyshevExpansion
        elif basis == "bessel":
            rbf = BesselExpansion
        elif basis == "gaussian":
            rbf = GaussianExpansion
        else:
            raise ValueError

        self.distance_expansion = rbf(
            max_value=cutoff,
            K=num_radial,
        )
        self.cutoff_fnc = PolynomialCutoff(cutoff, p=6)
        self.edge_net = nn.Sequential(DenseLayer(2 * self.si + self.num_radial,
                                                 self.si,
                                                 bias=True, activation=nn.SiLU()),
                                      DenseLayer(self.si, self.v_mul * self.vi + self.si,
                                                 bias=True)
                                      )
        self.scalar_net = DenseLayer(self.si, self.si, bias=True)
        self.update_net = GatedEquivBlock(in_dims=(self.si, self.vi),
                                          hs_dim=self.si, hv_dim=self.vi,
                                          out_dims=(self.si, self.vi),
                                          norm_eps=eps,
                                          use_mlp=use_mlp_update)
        self.reset_parameters()

    def reset_parameters(self):
        reset(self.edge_net)
        reset(self.scalar_net)
        reset(self.update_net)

    def forward(
        self,
        x: Tuple[Tensor, Tensor],
        edge_index: Tensor,
        edge_attr: Tuple[Tensor, Tensor],
    ):

        s, v = x
        d, r = edge_attr

        ms, mv = self.propagate(
            sa=s,
            sb=self.scalar_net(s),
            vb=self.vector_net(v),
            edge_attr=(d, r),
            edge_index=edge_index,
            dim_size=s.size(0),
        )

        s = ms + s
        v = mv + v

        ms, mv = self.update_net(x=(s, v))

        s = ms + s
        v = mv + v

        return s, v

    def aggregate(
        self,
            inputs: Tuple[Tensor, Tensor],
            index: Tensor,
            dim_size: Optional[int] = None
    ) -> Tuple[Tensor, Tensor]:
        s = scatter(inputs[0], index=index, dim=0, reduce="add", dim_size=dim_size)
        v = scatter(inputs[1], index=index, dim=0, reduce=self.vector_aggr, dim_size=dim_size)
        return s, v

    def message(
        self,
        sa_i: Tensor,
        sa_j: Tensor,
        sb_j: Tensor,
        vb_j: Tensor,
        index: Tensor,
        edge_attr: Tuple[Tensor, Tensor],
        dim_size: Optional[int]
    ) -> Tuple[Tensor, Tensor]:

        d, r = edge_attr

        de = self.distance_expansion(d)
        dc = self.cutoff_fnc(d)
        de = dc.view(-1, 1) * de

        aij = torch.cat([sa_i, sa_j, de], dim=-1)
        aij = self.edge_net(aij)

        if self.has_v_in:
            aij, vij0 = aij.split([self.si, 2*self.vi], dim=-1)
            vij0 = vij0.unsqueeze(1)
            vij0, vij1 = vij0.chunk(2, dim=-1)
        else:
            aij, vij0 = aij.split([self.si, self.vi], dim=-1)
            vij0 = vij0.unsqueeze(1)

        # feature attention
        aij = scatter_softmax(aij, index=index, dim=0, dim_size=dim_size)
        ns_j = aij * sb_j
        nv0_j = r.unsqueeze(-1) * vij0

        if self.has_v_in:
            nv1_j = vij1 * vb_j
            nv_j = nv0_j + nv1_j
        else:
            nv_j = nv0_j

        return ns_j, nv_j


if __name__ == '__main__':
    sdim = 128
    vdim = 32

    module = EQGATConv(in_dims=(sdim, vdim),
                       out_dims=(sdim, vdim),
                       num_radial=32,
                       cutoff=5.0,
                       has_v_in=True)


    print(sum(m.numel() for m in module.parameters() if m.requires_grad))

    from torch_geometric.nn import radius_graph
    s = torch.randn(30, sdim, requires_grad=True)
    v = torch.randn(30, 3, vdim, requires_grad=True)
    pos = torch.empty(30, 3).normal_(mean=0.0, std=3.0)
    edge_index = radius_graph(pos, r=5.0, flow="source_to_target")
    j, i = edge_index
    p_ij = pos[j] - pos[i]
    d_ij = torch.pow(p_ij, 2).sum(-1).sqrt()
    p_ij_n = p_ij / d_ij.unsqueeze(-1)

    os, ov = module(x=(s, v),
                    edge_index=edge_index,
                    edge_attr=(d_ij, p_ij_n)
                    )

    from scipy.spatial.transform import Rotation

    Q = torch.tensor(Rotation.random().as_matrix(), dtype=torch.get_default_dtype())

    vR = torch.einsum('ij, njk -> nik', Q, v)
    p_ij_n_R = torch.einsum('ij, nj -> ni', Q, p_ij_n)

    ovR_ = torch.einsum('ij, njk -> nik', Q, ov)

    osR, ovR = module(x=(s, vR),
                      edge_index=edge_index,
                      edge_attr=(d_ij, p_ij_n_R)
                      )

    print(torch.norm(os-osR, p=2))
    print(torch.norm(ovR_-ovR, p=2))

    module = EQGATConvNoCross(in_dims=(sdim, vdim),
                              out_dims=(sdim, vdim),
                              num_radial=32,
                              cutoff=5.0,
                              has_v_in=True)

    print(sum(m.numel() for m in module.parameters() if m.requires_grad))
    os, ov = module(x=(s, v),
                    edge_index=edge_index,
                    edge_attr=(d_ij, p_ij_n)
                    )
    osR, ovR = module(x=(s, vR),
                      edge_index=edge_index,
                      edge_attr=(d_ij, p_ij_n_R)
                      )
    ovR_ = torch.einsum('ij, njk -> nik', Q, ov)

    print(torch.norm(os - osR, p=2))
    print(torch.norm(ovR_ - ovR, p=2))