import os.path as osp

import torch
import pandas as pd
import math
import scipy
import numpy as np

from atom3d.datasets import LMDBDataset
import random
from torch.utils.data import IterableDataset
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader

from experiments.utils import prot_graph_transform
from experiments.ppi import neighbors as nb


try:
    PATH = osp.join(osp.dirname(osp.realpath(__file__)), "data")
except NameError:
    PATH = "experiments/ppi/data/"

DATA_DIR = osp.join(PATH, "DIPS-split/data")


class GNNPPITransform(object):
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


class PPIDataset(IterableDataset):
    '''
    A `torch.utils.data.IterableDataset` wrapper around a
    ATOM3D PPI dataset. Extracts (many) individual amino acid pairs
    from each structure of two interacting proteins. The returned graphs
    are seperate and each represents a 30 angstrom radius from the
    selected residue's alpha carbon.

    On each iteration, returns a pair of `torch_geometric.data.Data`
    graphs with the (same) attribute `label` which is 1 if the two
    amino acids interact and 0 otherwise, `ca_idx` for the node index
    of the alpha carbon, and all structural attributes as
    described in GNNPPITransform.

    Modified from
    https://github.com/drorlab/atom3d/blob/master/examples/ppi/gnn/data.py

    Excludes hydrogen atoms.

    :param lmdb_dataset: path to ATOM3D dataset
    '''

    def __init__(self, split: str = "train", transform=GNNPPITransform(cutoff=4.5,
                                                               remove_hydrogens=True)
                 ):
        if not split in ["train", "val", "test"]:
            print("Wrong selected split. Have to choose between ['train', 'val', 'test']")
            print("Exiting code")
            exit()

        self.split = split
        self.dataset = LMDBDataset(osp.join(DATA_DIR, split))
        self.transform = transform

    def __iter__(self):
        worker_info = torch.utils.data.get_worker_info()
        if worker_info is None:
            gen = self._dataset_generator(list(range(len(self.dataset))), shuffle=True)
        else:
            per_worker = int(math.ceil(len(self.dataset) / float(worker_info.num_workers)))
            worker_id = worker_info.id
            iter_start = worker_id * per_worker
            iter_end = min(iter_start + per_worker, len(self.dataset))
            gen = self._dataset_generator(
                list(range(len(self.dataset)))[iter_start:iter_end],
                shuffle=True)
        return gen

    def _df_to_graph(self, struct_df, chain_res, label):

        struct_df = struct_df[struct_df.element != 'H'].reset_index(drop=True)

        chain, resnum = chain_res
        res_df = struct_df[(struct_df.chain == chain) & (struct_df.residue == resnum)]
        if 'CA' not in res_df.name.tolist():
            return None
        ca_pos = res_df[res_df['name'] == 'CA'][['x', 'y', 'z']].astype(np.float32).to_numpy()[0]

        kd_tree = scipy.spatial.KDTree(struct_df[['x', 'y', 'z']].to_numpy())
        graph_pt_idx = kd_tree.query_ball_point(ca_pos, r=30.0, p=2.0)
        graph_df = struct_df.iloc[graph_pt_idx].reset_index(drop=True)

        ca_idx = np.where((graph_df.chain == chain) & (graph_df.residue == resnum) & (graph_df.name == 'CA'))[0]
        if len(ca_idx) != 1:
            return None

        data = self.transform(graph_df)
        data.label = label

        data.ca_idx = int(ca_idx)
        data.n_nodes = data.num_nodes

        return data

    def _dataset_generator(self, indices, shuffle=True):
        if shuffle: random.shuffle(indices)
        with torch.no_grad():
            for idx in indices:
                data = self.dataset[idx]

                neighbors = data['atoms_neighbors']
                pairs = data['atoms_pairs']

                for i, (ensemble_name, target_df) in enumerate(pairs.groupby(['ensemble'])):
                    sub_names, (bound1, bound2, _, _) = nb.get_subunits(target_df)
                    positives = neighbors[neighbors.ensemble0 == ensemble_name]
                    negatives = nb.get_negatives(positives, bound1, bound2)
                    negatives['label'] = 0
                    labels = self._create_labels(positives, negatives, num_pos=10, neg_pos_ratio=1)

                    for index, row in labels.iterrows():

                        label = float(row['label'])
                        chain_res1 = row[['chain0', 'residue0']].values
                        chain_res2 = row[['chain1', 'residue1']].values
                        graph1 = self._df_to_graph(bound1, chain_res1, label)
                        graph2 = self._df_to_graph(bound2, chain_res2, label)
                        if (graph1 is None) or (graph2 is None):
                            continue
                        yield graph1, graph2

    def _create_labels(self, positives, negatives, num_pos, neg_pos_ratio):
        frac = min(1, num_pos / positives.shape[0])
        positives = positives.sample(frac=frac)
        n = positives.shape[0] * neg_pos_ratio
        n = min(negatives.shape[0], n)
        negatives = negatives.sample(n, random_state=0, axis=0)
        labels = pd.concat([positives, negatives])[['chain0', 'residue0', 'chain1', 'residue1', 'label']]
        return labels



if __name__ == '__main__':
    dataset = PPIDataset(split="train",
                         transform=GNNPPITransform(cutoff=4.5,
                                                   remove_hydrogens=True)
                         )

    loader = DataLoader(dataset, batch_size=8)
    data0, data1 = next(iter(loader))
    print(data0.ca_idx, data1.ca_idx)
    print(data0.label, data1.label)
