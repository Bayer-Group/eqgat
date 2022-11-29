from typing import Callable, Optional, Tuple

import torch
import torch.nn.functional as F
from torch import Tensor, nn
from torch_geometric.nn import radius_graph
from torch_geometric.nn.inits import reset
from torch_scatter import scatter

from eqgat.convs.gvp import GVP, GVPConvLayer, LayerNorm
from eqgat.modules import GaussianExpansion


class GVPGNN(nn.Module):
    """
    Implements the GVP GNN Backbone
    """

    def __init__(
        self,
        node_dims: Tuple[int, Optional[int]] = (128, 16),
        edge_dims: Tuple[int, Optional[int]] = (16, 1),
        depth: int = 3,
        drop_rate: float = 0.0,
        n_message: int = 3,
        n_feedforward: int = 2,
        vector_gate: bool = True,
        activations: Tuple[Callable, Optional[Callable]] = (F.relu, None),
    ):
        super(GVPGNN, self).__init__()
        self.node_dims = node_dims
        self.edge_dims = edge_dims
        self.depth = depth
        self.activations = activations
        self.drop_rate = drop_rate

        self.convs = nn.ModuleList()
        for i in range(depth):
            self.convs.append(
                GVPConvLayer(
                    node_dims=node_dims,
                    edge_dims=edge_dims,
                    n_message=n_message,
                    n_feedforward=n_feedforward,
                    vector_gate=vector_gate,
                    activations=activations,
                    drop_rate=drop_rate,
                )
            )

        self.apply(fn=reset)

    def forward(self, x, edge_index, edge_attr):
        for conv in self.convs:
            x = conv(x, edge_index, edge_attr)
        return x


class GVPNetwork(nn.Module):
    """
    Implements the GVP Network including
    initial featurizer, GNN backbone and possible downstream layer
    on scalar part
    """

    def __init__(
        self,
        in_dim: int = 11,
        out_dim: int = 1,
        node_dims: Tuple[int, Optional[int]] = (128, 16),
        edge_dims: Tuple[int, Optional[int]] = (16, 1),
        # num_radial: int = 20,
        depth: int = 3,
        drop_rate: float = 0.0,
        n_message: int = 3,
        n_feedforward: int = 2,
        vector_gate: bool = True,
        activations: Tuple[Callable, Optional[Callable]] = (F.relu, None),
        r_cutoff: float = 10.0,
        max_num_neighbors: int = 32,
        regression_type: str = "graph",
    ):
        super(GVPNetwork, self).__init__()

        self.r_cutoff = r_cutoff
        num_radial = edge_dims[0]
        dim = node_dims[0]

        self.init_embedding = nn.Embedding(num_embeddings=in_dim, embedding_dim=node_dims[0])

        self.W_v = nn.Sequential(
            # LayerNorm((node_dims[0], 0)),
            GVP((node_dims[0], 0), node_dims, activations=(None, None), vector_gate=True),
        )

        self.rbf = GaussianExpansion(max_value=r_cutoff, K=num_radial)
        self.W_e = nn.Sequential(
            # LayerNorm((num_radial, 1)),
            GVP((num_radial, 1), edge_dims, activations=(None, None), vector_gate=True),
        )

        self.gnn = GVPGNN(
            node_dims=node_dims,
            edge_dims=edge_dims,
            depth=depth,
            drop_rate=drop_rate,
            n_message=n_message,
            n_feedforward=n_feedforward,
            vector_gate=vector_gate,
            activations=activations,
        )

        self.regression_type = regression_type
        self.downstream = nn.Sequential(
            nn.Linear(dim, dim, bias=True),
            nn.ReLU(),
            nn.Linear(dim, out_dim, bias=True),
        )
        self.max_num_neighbors = max_num_neighbors

        self.apply(fn=reset)

    # need to rewrite forward function as edge-attributes are treated seperately
    # https://github.com/drorlab/gvp-pytorch/blob/main/gvp/atom3d.py
    def forward(self, x: Tensor, pos: Tensor, batch: Optional[Tensor] = None) -> Tensor:
        if batch is None:
            batch = torch.zeros(size=x.size(0), dtype=torch.long, device=x.device)

        edge_index = radius_graph(x=pos, r=self.r_cutoff, batch=batch)
        row, col = edge_index
        rel_pos = pos[row] - pos[col]

        d = rel_pos.norm(dim=-1)
        rel_pos = F.normalize(rel_pos, dim=-1)

        # GVP uses the distance expansion as scalar edge-feature of shape [Ne, self.edge_dims[0]]
        # and the normed relative position between two nodes as vector edge-feature shape [Ne, self.edge_dims[1],  3]

        # distance expansion
        d = self.rbf(d)
        edge_attr = d, rel_pos.unsqueeze(-2)
        edge_attr = self.W_e(edge_attr)

        s = self.init_embedding(x).squeeze()
        s, v = self.W_v(s)
        s, v = self.gnn(x=(s, v), edge_index=edge_index, edge_attr=edge_attr)

        # node regression/classification ...
        if self.regression_type == "node":
            s = self.downstream(s)
        # graph regression/classification ...
        elif self.regression_type == "graph":
            s = scatter(src=s, index=batch, dim=0, reduce="mean")
            s = self.downstream(s)

        return s


if __name__ == '__main__':
    model = GVPNetwork(in_dim=11,
                       out_dim=1,
                       depth=5,
                       node_dims=(100, 1),
                       edge_dims=(16, 1),
                       r_cutoff=5.0
                       )
    print(sum(m.numel() for m in model.parameters() if m.requires_grad))
    # 639553
    x = torch.randint(0, 11, size=(100,))
    pos = torch.randn(100, 3).normal_(0, 2.0)
    batch=torch.arange(2).repeat_interleave(50)
    out = model(x=x, pos=pos, batch=batch)