import torch
import torch.nn as nn
from torch import Tensor
from torch_geometric.data import Batch
import torch.nn.functional as F
from torch_scatter import scatter
from torch_geometric.nn.inits import reset
from typing import Optional

from eqgat.modules import DenseLayer, GatedEquivBlock, LayerNorm, GatedEquivBlockTP
from eqgat.models import EQGATGNN, PaiNNGNN, SchNetGNN, GVPNetwork

# SEGNN
from eqgat.segnn.segnn import SEGNN
from eqgat.segnn.balanced_irreps import BalancedIrreps
from e3nn.o3 import Irreps
from e3nn.o3 import spherical_harmonics


class BaseModel(nn.Module):
    def __init__(
        self,
        num_elements: int,
        out_units: int,
        sdim: int = 128,
        vdim: int = 16,
        depth: int = 3,
        r_cutoff: float = 5.0,
        num_radial: int = 32,
        model_type: str = "eqgat",
        graph_level: bool = True,
        dropout: float = 0.1,
        use_norm: bool = True,
        aggr: str = "mean",
        graph_pooling: str = "mean",
        cross_ablate: bool = False,
        no_feat_attn: bool = False,
    ):
        if not model_type.lower() in ["painn", "eqgat", "schnet"]:
            print("Wrong selecte model type")
            print("Exiting code")
            exit()

        super(BaseModel, self).__init__()

        self.sdim = sdim
        self.vdim = vdim
        self.depth = depth
        self.r_cutoff = r_cutoff
        self.num_radial = num_radial
        self.model_type = model_type.lower()
        self.graph_level = graph_level
        self.num_elements = num_elements
        self.out_units = out_units

        self.init_embedding = nn.Embedding(num_embeddings=num_elements, embedding_dim=sdim)

        if self.model_type == "painn":
            self.gnn = PaiNNGNN(
                dims=(sdim, vdim),
                depth=depth,
                num_radial=num_radial,
                cutoff=r_cutoff,
                aggr=aggr,
                use_norm=use_norm,
            )
        elif self.model_type == "eqgat":
            self.gnn = EQGATGNN(
                dims=(sdim, vdim),
                depth=depth,
                cutoff=r_cutoff,
                num_radial=num_radial,
                use_norm=use_norm,
                basis="bessel",
                use_mlp_update=True,
                use_cross_product=not cross_ablate,
                no_feat_attn=no_feat_attn,
                vector_aggr=aggr
            )
        elif self.model_type == "schnet":
            self.gnn = SchNetGNN(
                dims=sdim,
                depth=depth,
                cutoff=r_cutoff,
                num_radial=num_radial,
                aggr=aggr,
                use_norm=use_norm,
            )

        self.use_norm = use_norm

        if self.model_type == "schnet":
            if use_norm:
                self.post_norm = LayerNorm(dims=(sdim, None))
            else:
                self.post_norm = None
            self.post_lin = DenseLayer(sdim, sdim, bias=False)
        else:
            if use_norm:
                self.post_norm = LayerNorm(dims=(sdim, vdim))
            else:
                self.post_norm = None
            self.post_lin = GatedEquivBlock(in_dims=(sdim, vdim),
                                            out_dims=(sdim, None),
                                            hs_dim=sdim, hv_dim=vdim,
                                            use_mlp=False)
        self.downstream = nn.Sequential(
            DenseLayer(sdim, sdim, activation=nn.SiLU(), bias=True),
            nn.Dropout(dropout),
            DenseLayer(sdim, out_units, bias=True)
        )

        self.graph_pooling = graph_pooling
        self.apply(reset)

    def forward(self, data: Batch, subset_idx: Optional[Tensor] = None) -> Tensor:
        s, pos, batch = data.x, data.pos, data.batch
        edge_index, d = data.edge_index, data.edge_weights

        s = self.init_embedding(s)

        row, col = edge_index
        rel_pos = pos[row] - pos[col]
        rel_pos = F.normalize(rel_pos, dim=-1, eps=1e-6)
        edge_attr = d, rel_pos

        if self.model_type in ["painn", "eqgat"]:
            v = torch.zeros(size=[s.size(0), 3, self.vdim], device=s.device)
            s, v = self.gnn(x=(s, v), edge_index=edge_index, edge_attr=edge_attr, batch=batch)
            if self.use_norm:
                s, v = self.post_norm(x=(s, v), batch=batch)
            s, _ = self.post_lin(x=(s, v))
        else:
            s = self.gnn(x=s, edge_index=edge_index, edge_attr=d, batch=batch)
            if self.use_norm:
                s, _ = self.post_norm(x=(s, None), batch=batch)
            s = self.post_lin(s)

        if self.graph_level:
            y_pred = scatter(s, index=batch, dim=0, reduce=self.graph_pooling)
        else:
            y_pred = s

        if subset_idx is not None:
            y_pred = y_pred[subset_idx]

        y_pred = self.downstream(y_pred)

        return y_pred



class SEGNNModel(nn.Module):
    def __init__(
        self,
        num_elements: int,
        out_units: int,
        hidden_dim: int,
        lmax: int = 2,
        depth: int = 3,
        graph_level: bool = True,
        use_norm: bool = True,
    ):
        super(SEGNNModel, self).__init__()
        self.init_embedding = nn.Embedding(num_embeddings=num_elements, embedding_dim=num_elements)
        self.input_irreps = Irreps(f"{num_elements}x0e")    # element embedding
        self.edge_attr_irreps = Irreps.spherical_harmonics(lmax)  # Spherical harmonics projection of relative pos.
        self.node_attr_irreps = Irreps.spherical_harmonics(lmax)    # aggregation of spherical harmonics projection
        self.hidden_irreps = BalancedIrreps(lmax, hidden_dim, sh_type=True)   # only considering SO(3)
        self.additional_message_irreps = Irreps("1x0e")  # euclidean distance
        self.output_irreps = Irreps(f"{out_units}x0e")  # SO(3) invariant output quantity
        self.model = SEGNN(self.input_irreps,
                           self.hidden_irreps,
                           self.output_irreps,
                           self.edge_attr_irreps,
                           self.node_attr_irreps,
                           num_layers=depth,
                           norm="instance" if use_norm else None,
                           pool="mean",
                           task="graph" if graph_level else "node",
                           additional_message_irreps=self.additional_message_irreps)
        self.model.init_pooler(pool="avg")


    def forward(self, data: Batch, subset_idx: Optional[Tensor ] = None) -> Tensor:
        x, pos, batch = data.x, data.pos, data.batch
        edge_index, d = data.edge_index, data.edge_weights
        x = self.init_embedding(x)

        row, col = edge_index
        rel_pos = pos[row] - pos[col]
        rel_pos = F.normalize(rel_pos, dim=-1, eps=1e-6)

        edge_attr = spherical_harmonics(l=self.edge_attr_irreps, x=rel_pos,
                                        normalize=True,
                                        normalization="component"
                                        )
        node_attr = scatter(edge_attr, col, dim=0, reduce="mean", dim_size=x.size(0))

        # to match https://github.com/RobDHess/Steerable-E3-GNN/blob/main/models/segnn/segnn.py#L101-L109
        new_data = data.clone()
        new_data.x = x
        new_data.edge_attr = edge_attr
        new_data.node_attr = node_attr
        new_data.additional_message_features = d.unsqueeze(-1)
        out = self.model(new_data)
        if subset_idx is not None:
            out = out[subset_idx]

        return out


if __name__ == '__main__':

    model0 = BaseModel(num_elements=9,
                       out_units=1,
                       sdim=128,
                       vdim=16,
                       depth=3,
                       r_cutoff=5.0,
                       num_radial=32,
                       model_type="eqgat",
                       graph_level=True,
                       use_norm=True)

    print(sum(m.numel() for m in model0.parameters() if m.requires_grad))
    # 375841
    model1 = SEGNNModel(num_elements=9,
                        out_units=1,
                        hidden_dim=128 + 3*16 + 5*8,
                        lmax=2,
                        depth=3,
                        graph_level=True,
                        use_norm=True)

    print(sum(m.numel() for m in model1.parameters() if m.requires_grad))
    # 350038
