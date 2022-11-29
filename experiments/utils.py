import torch
import pandas as pd
from atom3d.util.graph import mol_atoms, one_of_k_encoding_unk
from torch_geometric.data import Data

import torch_cluster
from typing import List

# https://github.com/drorlab/gvp-pytorch/blob/main/gvp/atom3d.py
NUM_ATOM_TYPES = 9
element_mapping = lambda x: {
    'H' : 0,
    'C' : 1,
    'N' : 2,
    'O' : 3,
    'F' : 4,
    'S' : 5,
    'Cl': 6, 'CL': 6,
    'P' : 7
}.get(x, 8)
amino_acids = lambda x: {
    'ALA': 0,
    'ARG': 1,
    'ASN': 2,
    'ASP': 3,
    'CYS': 4,
    'GLU': 5,
    'GLN': 6,
    'GLY': 7,
    'HIS': 8,
    'ILE': 9,
    'LEU': 10,
    'LYS': 11,
    'MET': 12,
    'PHE': 13,
    'PRO': 14,
    'SER': 15,
    'THR': 16,
    'TRP': 17,
    'TYR': 18,
    'VAL': 19
}.get(x, 20)



def prot_graph_transform(
    atom_df: pd.DataFrame,
    cutoff: float = 4.5,
    feat_col: str = "element",
    max_num_neighbors: int = 32,
    init_dtype: torch.dtype = torch.float64,
) -> Data:

    pos = torch.as_tensor(atom_df[["x", "y", "z"]].values, dtype=init_dtype)
    edge_index = torch_cluster.radius_graph(
        pos, r=cutoff, loop=False, max_num_neighbors=max_num_neighbors
    )
    dist = torch.pow(pos[edge_index[0]] - pos[edge_index[1]], exponent=2).sum(-1).sqrt()

    node_feats = torch.as_tensor(list(map(element_mapping, atom_df[feat_col])), dtype=torch.long)

    graph = Data(
        x=node_feats,
        pos=pos.to(torch.get_default_dtype()),
        edge_weights=dist.to(torch.get_default_dtype()),
        edge_index=edge_index,
    )

    return graph