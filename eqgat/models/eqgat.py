from typing import Optional, Tuple
import torch
from torch import Tensor, nn
from torch_geometric.nn.inits import reset
from torch_geometric.typing import OptTensor

from eqgat.modules import BatchNorm, LayerNorm
from eqgat.convs import EQGATConv, EQGATConvNoCross, EQGATNoFeatAttnConv


class EQGATGNN(nn.Module):
    def __init__(
        self,
        dims: Tuple[int, int] = (128, 32),
        depth: int = 5,
        eps: float = 1e-6,
        cutoff: Optional[float] = 5.0,
        num_radial: Optional[int] = 32,
        use_norm: bool = False,
        basis: str = "bessel",
        use_mlp_update: bool = True,
        use_cross_product: bool = True,
        no_feat_attn: bool = False,
        vector_aggr: str = "mean"
    ):
        super(EQGATGNN, self).__init__()
        self.dims = dims
        self.depth = depth
        self.use_norm = use_norm
        self.convs = nn.ModuleList()
        self.norms = nn.ModuleList()
        self.use_cross_product = use_cross_product
        self.vector_aggr = vector_aggr

        if use_cross_product:
            print("Using Cross Product")
            if no_feat_attn:
                print("Without Feature Attention")
                module = EQGATNoFeatAttnConv
            else:
                print("With Feature Attention")
                module = EQGATConv
        else:
            print("Using No Cross Product with Feature Attention")
            module = EQGATConvNoCross

        for i in range(depth):
            self.convs.append(
                module(
                    in_dims=dims,
                    has_v_in=i > 0,
                    out_dims=dims,
                    cutoff=cutoff,
                    num_radial=num_radial,
                    eps=eps,
                    basis=basis,
                    use_mlp_update=use_mlp_update,
                    vector_aggr=vector_aggr
                )
            )
            if use_norm:
                self.norms.append(
                    LayerNorm(dims=dims, affine=True)
                )
        self.apply(fn=reset)

    def forward(
        self,
        x: Tuple[Tensor, Tensor],
        edge_index: Tensor,
        edge_attr: Tuple[Tensor, Tensor],
        batch: Tensor
    ) -> Tuple[Tensor, Tensor]:

        s, v = x
        for i in range(len(self.convs)):
            s, v = self.convs[i](x=(s, v), edge_index=edge_index, edge_attr=edge_attr)
            if self.use_norm:
                s, v = self.norms[i](x=(s, v), batch=batch)
        return s, v



if __name__ == '__main__':
    from torch_geometric.nn import radius_graph
    from scipy.spatial.transform import Rotation

    sdim = 128
    vdim = 32

    model = EQGATGNN(dims=(sdim, vdim),
                     depth=5,
                     num_radial=32,
                     cutoff=5.0)


    print(sum(m.numel() for m in model.parameters() if m.requires_grad))
    # 629440

    # create two graphs of size (30, 30)
    s = torch.randn(30 * 2, sdim, requires_grad=True)
    v = torch.zeros(30 * 2, 3, vdim)
    pos = torch.empty(30 * 2, 3).normal_(mean=0.0, std=3.0)
    batch = torch.zeros(30, dtype=torch.long)
    batch = torch.concat([batch, torch.ones(30, dtype=torch.long)])


    edge_index = radius_graph(pos, r=5.0, batch=batch, flow="source_to_target")
    j, i = edge_index
    p_ij = pos[j] - pos[i]
    d_ij = torch.pow(p_ij, 2).sum(-1).sqrt()
    p_ij_n = p_ij / d_ij.unsqueeze(-1)


    os, ov = model(x=(s, v),
                   batch=batch,
                   edge_index=edge_index,
                   edge_attr=(d_ij, p_ij_n))

    Q = torch.tensor(Rotation.random().as_matrix(), dtype=torch.get_default_dtype())

    vR = torch.einsum('ij, njk -> nik', Q, v)  # should be zero anyways, since v init is 0.
    p_ij_n_R = torch.einsum('ij, nj -> ni', Q, p_ij_n)

    ovR_ = torch.einsum('ij, njk -> nik', Q, ov)

    osR, ovR = model(x=(s, vR),
                     batch=batch,
                     edge_index=edge_index,
                     edge_attr=(d_ij, p_ij_n_R))

    print(torch.norm(os-osR, p=2))
    print(torch.norm(ovR_-ovR, p=2))
    # tensor(1.5796e-05, grad_fn=<NormBackward1>)
    # tensor(2.2695e-06, grad_fn=<NormBackward1>)
