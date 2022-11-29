from typing import Optional, Tuple

from torch import Tensor, nn
from torch_geometric.nn.inits import reset
from torch_geometric.typing import OptTensor

from eqgat.modules import BatchNorm, LayerNorm
from eqgat.convs import PaiNNConv


class PaiNNGNN(nn.Module):
    def __init__(
            self,
            dims: Tuple[int, int] = (128, 32),
            depth: int = 5,
            aggr: str = "mean",
            eps: float = 1e-6,
            cutoff: Optional[float] = 5.0,
            num_radial: Optional[int] = 32,
            use_norm: bool = False,
    ):
        super(PaiNNGNN, self).__init__()
        self.dims = dims
        self.depth = depth
        self.use_norm = use_norm
        self.convs = nn.ModuleList()
        self.norms = nn.ModuleList()
        for i in range(depth):
            self.convs.append(
                PaiNNConv(
                    in_dims=dims,
                    out_dims=dims,
                    has_v_in=i>0,
                    aggr=aggr,
                    cutoff=cutoff,
                    num_radial=num_radial,
                    norm_eps=eps,

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
