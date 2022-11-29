import os
import os.path as osp
from argparse import ArgumentParser
from typing import Dict

import torch
from atom3d.datasets import LMDBDataset

from torch_geometric.data import Data
from tqdm import tqdm

from experiments.utils import prot_graph_transform

try:
    PATH = osp.join(osp.dirname(osp.realpath(__file__)), "data")
except NameError:
    PATH = "experiments/rsr/data/"

DATA_DIR = osp.join(PATH, "candidates-split-by-time//data")


class GNNTransformRSR(object):
    def __init__(
        self, cutoff: float = 4.5,
            remove_hydrogens: bool = True,
            max_num_neighbors: int = 32
    ):
        self.cutoff = cutoff
        self.remove_hydrogens = remove_hydrogens
        self.max_num_neighbors = max_num_neighbors

    def __call__(self, item: Dict) -> Data:
        atom_df = item["atoms"]
        if self.remove_hydrogens:
            atom_df = atom_df[atom_df.element != "H"].reset_index(drop=True)
        labels = item["scores"]
        graph = prot_graph_transform(
            atom_df=atom_df, cutoff=self.cutoff,
            max_num_neighbors=self.max_num_neighbors
        )
        graph.y = torch.FloatTensor([labels["rms"]])
        split = item["id"].split("'")
        graph.target = split[1]
        graph.decoy = split[3]
        return graph


if __name__ == "__main__":

    parser = ArgumentParser(
        description="Dataloading script to access the RSR Graph Dataset directly from LMDB."
    )
    parser.add_argument(
        "--cutoff",
        help="radial cutoff for connecting edges. Defaults to 4.5",
        type=float,
        default=4.5,
    )

    args = parser.parse_args()

    transform = GNNTransformRSR(
        cutoff=args.cutoff,
        remove_hydrogens=True,
    )

    train_dataset = LMDBDataset(osp.join(DATA_DIR, "train"), transform=transform)
    ntrain = len(train_dataset)
    val_dataset = LMDBDataset(osp.join(DATA_DIR, "val"), transform=transform)
    nvalid = len(val_dataset)
    test_dataset = LMDBDataset(osp.join(DATA_DIR, "test"), transform=transform)
    ntest = len(test_dataset)

    print(
        "Just using the LMBDDataset and transform. Loading 100 data samples in train/val/test"
    )
    for i, data in tqdm(enumerate(train_dataset), total=len(train_dataset)):
        if i == 100:
            break

    for i, data in tqdm(enumerate(val_dataset), total=len(val_dataset)):
        if i == 100:
            break

    for i, data in tqdm(enumerate(test_dataset), total=len(test_dataset)):
        if i == 100:
            break
