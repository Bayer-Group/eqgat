import os
import os.path as osp
from argparse import ArgumentParser
from typing import Dict, List, Tuple, Union

import pandas as pd
import torch
from atom3d.datasets import LMDBDataset


from torch_geometric.data import Data, Dataset
from torch_geometric.loader import DataLoader
from tqdm import tqdm

from experiments.utils import prot_graph_transform


def chunks_n_sized(l, n):
    """Yield successive n-sized chunks from lst."""
    for i in range(0, len(l), n):
        yield l[i : i + n]


def chunks_n(l, n):
    """Yield n number of striped chunks from l."""
    for i in range(0, n):
        yield l[i::n]


try:
    PATH = osp.join(osp.dirname(osp.realpath(__file__)), "data")
except NameError:
    PATH = "experiments/lba/data/"


DATA_DIR = osp.join(PATH, "split-by-sequence-identity-30/data")
# DATA_DIR = osp.join(PATH, "split-by-sequence-identity-60/data")


class GNNTransformLBA(object):
    def __init__(
        self,
        cutoff: float = 4.5,
        remove_hydrogens: bool = False,
        pocket_only: bool = True,
        max_num_neighbors: int = 32,
    ):
        self.pocket_only = pocket_only
        self.cutoff = cutoff
        self.remove_hydrogens = remove_hydrogens
        self.max_num_neighbors = max_num_neighbors

    def __call__(self, item: Dict) -> Data:
        ligand_df = item["atoms_ligand"]
        if self.pocket_only:
            protein_df = item["atoms_pocket"]
        else:
            protein_df = item["atoms_protein"]

        atom_df = pd.concat([protein_df, ligand_df], axis=0)

        if self.remove_hydrogens:
            # remove hydrogens
            atom_df = atom_df[atom_df.element != "H"].reset_index(drop=True)

        labels = item["scores"]
        graph = prot_graph_transform(
            atom_df=atom_df, cutoff=self.cutoff, max_num_neighbors=self.max_num_neighbors
        )
        graph.y = torch.FloatTensor([labels["neglog_aff"]])

        lig_flag = torch.zeros(atom_df.shape[0], dtype=torch.bool)
        lig_flag[-len(ligand_df):] = 1
        graph.lig_flag = lig_flag

        graph.prot_id = item["id"]
        graph.smiles = item["smiles"]

        return graph


class CustomLBADataset(Dataset):
    """
    Simply subclassing torch_geometric.data.Dataset by using the processed files
    Allows faster loading of batches
    """

    def __init__(self, root: str = PATH, split: str = "train"):
        # no transform, as we have processed the files already
        super(CustomLBADataset, self).__init__(root)
        if not split in ["train", "val", "test"]:
            print("Wrong selected split. Have to choose between ['train', 'val', 'test']")
            print("Exiting code")
            exit()
        root_dir = os.path.join(root, split)
        self.files = [os.path.join(root_dir, f) for f in os.listdir(root_dir)]

    @property
    def raw_file_names(self) -> Union[str, List[str], Tuple]:
        return self.files

    @property
    def processed_file_names(self) -> Union[str, List[str], Tuple]:
        return self.files

    def len(self):
        return len(self.processed_file_names)

    def get(self, idx):
        # print(idx)
        datadict = torch.load(self.files[idx])
        # __getitem__ handled by parent-class.
        # https://github.com/rusty1s/pytorch_geometric/blob/master/torch_geometric/data/dataset.py#L184-L203
        return datadict


if __name__ == "__main__":

    parser = ArgumentParser(
        description="Preprocessing script to obtain pytorch graph files for the LBA."
    )
    parser.add_argument(
        "--cutoff",
        help="radial cutoff for connecting edges. Defaults to 4.5",
        type=float,
        default=4.5,
    )
    parser.add_argument(
        "--remove_hydrogens",
        help="If hydrogen atoms in the protein/pocket graph should be removed. Defaults to False.",
        default=False,
        action="store_true",
    )

    args = parser.parse_args()

    train_dir = osp.join(PATH, "train")
    val_dir = osp.join(PATH, "val")
    test_dir = osp.join(PATH, "test")
    os.makedirs(train_dir, exist_ok=True)
    os.makedirs(val_dir, exist_ok=True)
    os.makedirs(test_dir, exist_ok=True)

    transform = GNNTransformLBA(
        cutoff=args.cutoff,
        remove_hydrogens=args.remove_hydrogens,
        pocket_only=True,
    )

    train_dataset = LMDBDataset(osp.join(DATA_DIR, "train"), transform=transform)
    ntrain = len(train_dataset)
    val_dataset = LMDBDataset(osp.join(DATA_DIR, "val"), transform=transform)
    nvalid = len(val_dataset)
    test_dataset = LMDBDataset(osp.join(DATA_DIR, "test"), transform=transform)
    ntest = len(test_dataset)

    print("Processing Training set")
    for i in tqdm(range(ntrain), total=ntrain):
        save_path = osp.join(train_dir, f"data_{i}.pth")
        if not osp.exists(save_path):
            data = train_dataset[i]
            torch.save(data, f=save_path)

    print("Processing Validation set")
    for i in tqdm(range(nvalid), total=nvalid):
        save_path = osp.join(val_dir, f"data_{i}.pth")
        if not osp.exists(save_path):
            data = val_dataset[i]
            torch.save(data, f=save_path)

    print("Processing Test set")
    for i in tqdm(range(ntest), total=ntest):
        save_path = osp.join(test_dir, f"data_{i}.pth")
        if not osp.exists(save_path):
            data = test_dataset[i]
            torch.save(data, f=save_path)

    print("Try dataloading with batch-size 16 on train/val/test dataset")
    # we just want to load the processed files
    train_dataset = CustomLBADataset(split="train")
    val_dataset = CustomLBADataset(split="val")
    test_dataset = CustomLBADataset(split="test")

    train_loader = DataLoader(
        dataset=train_dataset, batch_size=16, shuffle=False, num_workers=4
    )
    val_loader = DataLoader(
        dataset=val_dataset, batch_size=16, shuffle=False, num_workers=4
    )
    test_loader = DataLoader(
        dataset=test_dataset, batch_size=16, shuffle=False, num_workers=4
    )

    for data in tqdm(train_loader, total=len(train_loader)):
        pass

    for data in tqdm(val_loader, total=len(val_loader)):
        pass

    for data in tqdm(test_loader, total=len(test_loader)):
        pass
