## Example
Here we provide an example for using the proposed equivariant graph attention convolution.

```python3
import torch
from torch_geometric.nn import radius_graph
from scipy.spatial.transform import Rotation
from eqgat.convs import EQGATConv
from eqgat.models import EQGATGNN


sdim = 128
vdim = 32

module = EQGATConv(in_dims=(sdim, vdim),
                   out_dims=(sdim, vdim),
                   num_radial=32,
                   cutoff=5.0,
                   has_v_in=True)

print(sum(m.numel() for m in module.parameters() if m.requires_grad))
# 127744

# create some random data, one graph with 30 nodes
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


# Check for invariance and equivariance
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
#tensor(3.1618e-06, grad_fn=<NormBackward1>)
#tensor(4.1737e-06, grad_fn=<NormBackward1>)

#####################################################

# Let's check for EQGAT-GNN encoder model 
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
```