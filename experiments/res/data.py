import os.path as osp
import random
import math
import numpy as np

import torch
from atom3d.datasets import LMDBDataset

from torch.utils.data import IterableDataset
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader

from experiments.utils import prot_graph_transform

try:
    PATH = osp.join(osp.dirname(osp.realpath(__file__)), "data", "RES", "raw")
except NameError:
    PATH = "experiments/res/data/RES/raw"

DATA_DIR = osp.join(PATH, "data")
SPLIT_DIR = osp.join(PATH, "indices")


_NUM_ATOM_TYPES = 9
_element_mapping = lambda x: {
    'H' : 0,
    'C' : 1,
    'N' : 2,
    'O' : 3,
    'F' : 4,
    'S' : 5,
    'Cl': 6, 'CL': 6,
    'P' : 7
}.get(x, 8)
_amino_acids = lambda x: {
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



class GNNResTransform(object):
    def __init__(
        self,
            cutoff: float = 4.5,
            remove_hydrogens: bool = True,
            max_num_neighbors: int = 32,
            init_dtype: torch.dtype = torch.float32
    ):
        self.cutoff = cutoff
        self.remove_hydrogens = remove_hydrogens
        self.max_num_neighbors = max_num_neighbors
        self.init_dtype = init_dtype

    def __call__(self, atom_df) -> Data:
        if self.remove_hydrogens:
            atom_df = atom_df[atom_df.element != "H"].reset_index(drop=True)
        graph = prot_graph_transform(
            atom_df=atom_df, cutoff=self.cutoff, max_num_neighbors=self.max_num_neighbors, init_dtype=self.init_dtype
        )
        return graph


class RESDataset(IterableDataset):
    '''
    A `torch.utils.data.IterableDataset` wrapper around a
    ATOM3D RES dataset.

    On each iteration, returns a `torch_geometric.data.Data`
    graph with the attribute `label` encoding the masked residue
    identity, `ca_idx` for the node index of the alpha carbon,
    and all structural attributes as described in GNNResTransform.

    :param lmdb_dataset: path to ATOM3D dataset
    :param split_path: path to the ATOM3D split file
    '''
    def __init__(self, split: str = "train", transform=GNNResTransform(remove_hydrogens=True,
                                                                       cutoff=4.5)):
        self.dataset = LMDBDataset(DATA_DIR)
        self.split_file = osp.join(SPLIT_DIR, f"{split}_indices.txt")
        self.idx = list(map(int, open(osp.join(SPLIT_DIR, f"{split}_indices.txt")).read().split()))
        self.transform = transform

    def __iter__(self):
        worker_info = torch.utils.data.get_worker_info()
        if worker_info is None:
            gen = self._dataset_generator(list(range(len(self.idx))),
                                          shuffle=True)
        else:
            per_worker = int(math.ceil(len(self.idx) / float(worker_info.num_workers)))
            worker_id = worker_info.id
            iter_start = worker_id * per_worker
            iter_end = min(iter_start + per_worker, len(self.idx))
            gen = self._dataset_generator(list(range(len(self.idx)))[iter_start:iter_end],
                                          shuffle=True)
        return gen

    def _dataset_generator(self, indices, shuffle=True):
        if shuffle: random.shuffle(indices)
        with torch.no_grad():
            for idx in indices:
                data = self.dataset[self.idx[idx]]
                atoms = data['atoms']
                for sub in data['labels'].itertuples():
                    _, num, aa = sub.subunit.split('_')
                    num, aa = int(num), _amino_acids(aa)
                    if aa == 20: continue
                    my_atoms = atoms.iloc[data['subunit_indices'][sub.Index]].reset_index(drop=True)
                    ca_idx = np.where((my_atoms.residue == num) & (my_atoms.name == 'CA'))[0]
                    if len(ca_idx) != 1:
                        continue
                    with torch.no_grad():
                        graph = self.transform(my_atoms)
                        graph.label = aa
                        graph.ca_idx = int(ca_idx)
                        yield graph


def check_atom3d_iterable():
    dataset = RESDataset(split="test")
    loader = DataLoader(dataset, batch_size=16, num_workers=4)
    data = next(iter(loader))
    print(data)
    ca_idx_pointer = data.ca_idx + data.ptr[:-1]
    print(ca_idx_pointer)
    print(data.label)
    return loader


if __name__ == '__main__':
    loader = check_atom3d_iterable()
    from tqdm import tqdm
    cnt = 0
    for data in tqdm(loader):
        if cnt == 100:
            break
        cnt += 1

