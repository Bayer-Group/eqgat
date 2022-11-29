from typing import Optional, Tuple

import torch
from torch import Tensor, nn
from torch_geometric.nn import MessagePassing
from torch_geometric.nn.inits import reset
from torch_scatter import scatter

from eqgat.modules import GaussianExpansion, CosineCutoff, ShiftedSoftPlus, DenseLayer


"""
Unofficial Pytorch Geometric Implementation of 
`SchNet: A continuous-filter convolutional neural network for modeling quantum interactions` (Schütt et al. 2017)
https://arxiv.org/abs/1706.08566

@inproceedings{NIPS2017_303ed4c6,
author = {Sch\"{u}tt, Kristof and Kindermans, Pieter-Jan and Sauceda Felix, Huziel Enoc and Chmiela, Stefan and Tkatchenko, Alexandre and M\"{u}ller, Klaus-Robert},
booktitle = {Advances in Neural Information Processing Systems},
editor = {I. Guyon and U. Von Luxburg and S. Bengio and H. Wallach and R. Fergus and S. Vishwanathan and R. Garnett},
publisher = {Curran Associates, Inc.},
title = {SchNet: A continuous-filter convolutional neural network for modeling quantum interactions},
url = {https://proceedings.neurips.cc/paper/2017/file/303ed4c69846ab36c2904d3ba8573050-Paper.pdf},
volume = {30},
year = {2017},
Bdsk-Url-1 = {https://proceedings.neurips.cc/paper/2017/file/303ed4c69846ab36c2904d3ba8573050-Paper.pdf}}

"""


class SchNetConv(MessagePassing):
    def __init__(
        self,
        in_dims: int,
        out_dims: int,
        aggr: str = "mean",
        num_radial: int = 32,
        cutoff: float = 5.0,
    ):
        super(SchNetConv, self).__init__(node_dim=0, aggr=aggr, flow="source_to_target")

        self.si = in_dims
        self.so = out_dims
        self.num_radial = num_radial
        self.cutoff = cutoff
        self.distance_expansion = GaussianExpansion(max_value=cutoff, K=num_radial)
        self.cutoff_fnc = CosineCutoff(cutoff=cutoff)
        self.Wn = DenseLayer(self.si, self.si, bias=True)

        self.Wf = nn.Sequential(DenseLayer(num_radial, self.si, bias=True),
                                ShiftedSoftPlus(),
                                DenseLayer(self.si, self.si, bias=True),
                                ShiftedSoftPlus()
                                )

        self.Wu = nn.Sequential(
            DenseLayer(self.si, self.si, bias=True),
            ShiftedSoftPlus(),
            DenseLayer(self.si, self.si, bias=True)
        )

        self.apply(fn=reset)

    def forward(
        self,
        x: Tensor,
        edge_attr: Tensor,
        edge_index: Tensor,
    ) -> Tensor:

        sn = self.Wn(x)

        os = self.propagate(
            edge_index=edge_index, s=sn, edge_attr=edge_attr, dim_size=x.size(0)
        )

        os = self.Wu(os)
        s = x + os

        return s

    def aggregate(
        self, inputs: Tensor, index: Tensor, dim_size: Optional[int]
    ) -> Tensor:

        msg_s= inputs
        msg_s = scatter(msg_s, index, dim=0, reduce=self.aggr, dim_size=dim_size)
        return msg_s

    def message(
        self, s_j: Tensor, edge_attr: Tensor
    ) -> Tensor:

        d = edge_attr
        de = self.distance_expansion(d)
        w = self.Wf(de)
        dc = self.cutoff_fnc(d)
        w = dc.view(-1, 1) * w

        s_j = w * s_j

        return s_j


if __name__ == '__main__':
    sdim = 128

    module = SchNetConv(in_dims=sdim,
                        out_dims=sdim,
                        num_radial=32,
                        cutoff=5.0)

    print(sum(m.numel() for m in module.parameters() if m.requires_grad))
    # 70272

    from torch_geometric.nn import radius_graph
    s = torch.randn(30, sdim, requires_grad=True)
    pos = torch.empty(30, 3).normal_(mean=0.0, std=3.0)
    edge_index = radius_graph(pos, r=5.0, flow="source_to_target")
    j, i = edge_index
    p_ij = pos[j] - pos[i]
    d_ij = torch.pow(p_ij, 2).sum(-1).sqrt()

    os = module(x=s,
                edge_index=edge_index,
                edge_attr=d_ij
                )
