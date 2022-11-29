from typing import Optional, Tuple

import torch
import torch.nn.functional as F
from torch import Tensor, nn
from torch_geometric.nn import MessagePassing
from torch_geometric.nn.inits import reset
from torch_scatter import scatter

from eqgat.modules import GaussianExpansion, CosineCutoff, DenseLayer, BesselExpansion

"""
Unofficial Pytorch Geometric Implementation of 
`Equivariant message passing for the prediction of tensorial properties and molecular spectra` (Schütt et al. 2021)
https://arxiv.org/abs/2102.03150


@InProceedings{pmlr-v139-schutt21a,
  title = 	 {Equivariant message passing for the prediction of tensorial properties and molecular spectra},
  author =       {Sch{\"u}tt, Kristof and Unke, Oliver and Gastegger, Michael},
  booktitle = 	 {Proceedings of the 38th International Conference on Machine Learning},
  pages = 	 {9377--9388},
  year = 	 {2021},
  editor = 	 {Meila, Marina and Zhang, Tong},
  volume = 	 {139},
  series = 	 {Proceedings of Machine Learning Research},
  month = 	 {18--24 Jul},
  publisher =    {PMLR},
  pdf = 	 {http://proceedings.mlr.press/v139/schutt21a/schutt21a.pdf},
  url = 	 {https://proceedings.mlr.press/v139/schutt21a.html},
}

"""


class PaiNNUpdate(nn.Module):
    def __init__(
        self,
        in_dims: Tuple[int, Optional[int]],
        out_dims: Tuple[int, Optional[int]],
        norm_eps: float = 1e-6,
    ):
        super(PaiNNUpdate, self).__init__()
        self.si, self.vi = in_dims
        self.so, self.vo = out_dims
        self.norm_eps = norm_eps

        self.U = DenseLayer(self.vi, self.vi, bias=False)
        self.V = DenseLayer(self.vi, self.si, bias=False)

        self.W0 = DenseLayer(self.si + self.si, self.si, bias=True, activation=nn.SiLU())
        self.W1 = DenseLayer(self.si, self.vi + 2*self.si, bias=True)

        self.apply(fn=reset)

    def forward(self, x: Tuple[Tensor, Tensor]) -> Tuple[Tensor, Tensor]:

        s, v = x
        uv = self.U(v)
        vv = self.V(v)

        vv_dot = torch.clamp(vv.square().sum(dim=1), min=self.norm_eps)
        vv_norm = vv_dot.sqrt()

        s = torch.cat([s, vv_norm], dim=-1)
        s = self.W1(self.W0(s))

        # split
        avv, asv, ass = s.split([self.vi, self.si, self.si], dim=-1)

        s = vv_dot * asv + ass
        v = uv * avv.unsqueeze(1)

        return s, v


class PaiNNConv(MessagePassing):
    def __init__(
        self,
        in_dims: Tuple[int, Optional[int]],
        out_dims: Tuple[int, Optional[int]],
        has_v_in: bool = True,
        norm_eps: float = 1e-6,
        aggr: str = "mean",
        num_radial: int = 32,
        cutoff: float = 5.0,
    ):
        super(PaiNNConv, self).__init__(node_dim=0, aggr=aggr, flow="source_to_target")

        self.si, self.vi = in_dims
        self.so, self.vo = out_dims
        self.norm_eps = norm_eps
        self.num_radial = num_radial
        self.cutoff = cutoff
        self.has_v_in = has_v_in
        if has_v_in:
            self.vmul = 2
        else:
            self.vmul = 1

        self.Ws0 = DenseLayer(self.si, self.si, bias=True, activation=nn.SiLU())
        self.Ws1 = DenseLayer(self.si, self.vmul * self.vi + self.si, bias=True)

        self.distance_expansion = GaussianExpansion(max_value=cutoff, K=num_radial)   # might work better on proteins.
        # self.distance_expansion = BesselExpansion(max_value=cutoff, K=num_radial)
        self.cutoff_fnc = CosineCutoff(cutoff=cutoff)

        self.Wf = DenseLayer(num_radial, self.si + self.vmul * self.vi, bias=True)
        self.update_nn = PaiNNUpdate(in_dims=in_dims, out_dims=out_dims)

        self.apply(fn=reset)

    def forward(
        self,
        x: Tuple[Tensor, Tensor],
        edge_attr: Tuple[Tensor, Tensor],
        edge_index: Tensor,
    ) -> Tuple[Tensor, Tensor]:

        s, v = x

        sn = self.Ws1(self.Ws0(s))

        os, ov = self.propagate(
            edge_index=edge_index, s=sn, v=v, edge_attr=edge_attr, dim_size=s.size(0)
        )

        s = s + os
        v = v + ov

        os, ov = self.update_nn(x=(s, v))

        s = s + os
        v = v + ov

        return s, v

    def aggregate(
        self, inputs: Tuple[Tensor, Tensor], index: Tensor, dim_size: Optional[int]
    ) -> Tuple[Tensor, Tensor]:

        msg_s, msg_v = inputs
        msg_s = scatter(msg_s, index, dim=0, reduce=self.aggr, dim_size=dim_size)
        msg_v = scatter(msg_v, index, dim=0, reduce=self.aggr, dim_size=dim_size)
        return msg_s, msg_v

    def message(
        self, s_j: Tensor, v_j: Tensor, edge_attr: Tuple[Tensor, Tensor]
    ) -> Tuple[Tensor, Tensor]:

        d, r = edge_attr
        de = self.distance_expansion(d)
        w = self.Wf(de)
        dc = self.cutoff_fnc(d)
        w = dc.view(-1, 1) * w

        wphi = w * s_j

        if self.has_v_in:
            s_j, v_gate, r_gate = wphi.split([self.si, self.vi, self.vi], dim=-1)
            v_j = v_gate.unsqueeze(1) * v_j + r_gate.unsqueeze(1) * r.unsqueeze(-1)
        else:
            s_j, r_gate = wphi.split([self.si, self.vi], dim=-1)
            v_j = r_gate.unsqueeze(1) * r.unsqueeze(-1)

        return s_j, v_j



if __name__ == '__main__':

    sdim = 128
    vdim = 32

    module = PaiNNConv(in_dims=(sdim, vdim),
                       out_dims=(sdim, vdim),
                       num_radial=32,
                       cutoff=5.0)


    print(sum(m.numel() for m in module.parameters() if m.requires_grad))
    # 122784

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