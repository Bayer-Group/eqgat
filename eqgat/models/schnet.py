from typing import Optional, Tuple

from torch import Tensor, nn
from torch_geometric.nn.inits import reset
from torch_geometric.typing import OptTensor

from eqgat.modules import BatchNorm, LayerNorm
from eqgat.convs import SchNetConv


class SchNetGNN(nn.Module):
    def __init__(
            self,
            dims: int,
            depth: int = 5,
            aggr: str = "mean",
            cutoff: Optional[float] = 5.0,
            num_radial: Optional[int] = 32,
            use_norm: bool = False,
    ):
        super(SchNetGNN, self).__init__()
        self.dims = dims
        self.depth = depth
        self.use_norm = use_norm
        self.convs = nn.ModuleList()
        self.norms = nn.ModuleList()
        for i in range(depth):
            self.convs.append(
                SchNetConv(
                    in_dims=dims,
                    out_dims=dims,
                    aggr=aggr,
                    cutoff=cutoff,
                    num_radial=num_radial,
                )
            )
            if use_norm:
                self.norms.append(
                    LayerNorm(dims=(dims, None), affine=True)
                )

        self.apply(fn=reset)

    def forward(
            self,
            x: Tensor,
            edge_index: Tensor,
            edge_attr: Tensor,
            batch: Tensor
    ) -> Tuple[Tensor, Tensor]:

        for i, conv in enumerate(self.convs):
            x = conv(x=x, edge_index=edge_index, edge_attr=edge_attr)
            if self.use_norm:
                x, _ = self.norms[i](x=(x, None), batch=batch)
        return x
