import os
import os.path as osp
import numpy as np
from argparse import ArgumentParser
import random
import math
from typing import Tuple
from scipy.spatial.distance import pdist
from tqdm import tqdm
import torch
from torch_geometric.data import Data, InMemoryDataset
from torch_geometric.loader import DataLoader
from typing import Optional

from pytorch_lightning import LightningDataModule

try:
    DATA_DIR = osp.join(osp.dirname(osp.realpath(__file__)), "data")
except NameError:
    DATA_DIR = "experiments/synthetic/data/"


def generate_random_3d_point_within_sphere(max_r: float = 10.0) -> np.ndarray:
  """
  https://math.stackexchange.com/questions/87230/picking-random-points-in-the-volume-of-sphere-with-uniform-probability/87238#87238
  """
  u = np.random.rand(1, )
  xyz = np.random.randn(1, 3)
  d = np.sqrt(np.sum(np.power(xyz, 2), -1))
  xyz /= d
  c = np.cbrt(u)
  xyz *= (c * max_r)

  return xyz


def generate_point_cloud(max_r: float,
                         select_k: int = 3,
                         min_dist: float = 2.0,
                         num_points: int = 100) -> Tuple[np.ndarray, np.ndarray,
                                                         np.ndarray, np.ndarray]:
  pos = []
  i = len(pos)
  while i < num_points:
    p = generate_random_3d_point_within_sphere(max_r)
    if i == 0:
      pos.append(p)
    else:
      pos_mat = np.concatenate(pos, axis=0)
      pos_mat = np.concatenate([pos_mat, p], axis=0)
      dist = pdist(pos_mat)
      b = np.all(dist > min_dist)
      if b:
        pos.append(p)
    i = len(pos)

  selection_points = np.random.randint(0, num_points, (select_k, ))
  rest_points = np.delete(np.arange(num_points), selection_points)
  mask = np.zeros((num_points, 1))
  mask[selection_points] = 1.0
  pos = np.concatenate(pos, axis=0)
  out = pos, selection_points, rest_points, mask
  return out


def generate_triangle_dataset(num_samples: int = 50_000,
                              min_dist: float = 2.0,
                              max_r: float = 10.0,
                              num_points: int = 100) -> list:
  datalist = [None] * num_samples
  for i in tqdm(range(num_samples)):
    pos, triangle_points, rest_points, mask = generate_point_cloud(max_r=max_r,
                                                                   min_dist=min_dist,
                                                                   select_k=3,
                                                                   num_points=num_points)
    pos_triangles = pos[triangle_points]

    # creating labels computing distance, perimeter and area
    com_pointcloud = pos.mean(0)
    com_triangle = pos_triangles.mean(0)
    # distance COM pointcloud to COM of triangle
    distance = np.sqrt(np.power(com_pointcloud - com_triangle, 2).sum())
    A, B, C = np.split(pos_triangles, 3, axis=0)
    AB = B - A
    AC = C - A
    BC = C - B
    # triangle properties
    area = 0.5 * np.sqrt(np.power(np.cross(AB, AC), 2).sum())
    perimeter = np.sqrt(np.power(AB, 2).sum()) + np.sqrt(np.power(AC, 2).sum()) + np.sqrt(np.power(BC, 2).sum())

    y = np.array([distance, area, perimeter])
    datadict = {'x': mask.astype(np.int64).squeeze(), 'mask': mask.astype(bool).squeeze(), 'pos': pos, 'y': y}
    datalist[i] = datadict

  return datalist


def radian_to_degree(radian: float) -> float:
  return radian * (180. / math.pi)


def generate_cone_dataset(num_samples: int = 50_000,
                          min_dist: float = 2.0,
                          max_r: float = 10.0,
                          num_points: int = 100,
                          angle_radian: float = 45.0) -> list:

  datalist = [None] * num_samples
  for i in tqdm(range(num_samples)):
    pos, chone_anchor_point, rest_points, mask = generate_point_cloud(max_r=max_r,
                                                                      min_dist=min_dist,
                                                                      select_k=1,
                                                                      num_points=num_points)
    pos_anchor = pos[chone_anchor_point]
    # creating labels by computing angles arccos( a.b / (|a| * |b]) )
    a = (pos_anchor * pos).sum(-1)
    b = np.sqrt(np.power(pos_anchor, 2).sum(-1)) * np.sqrt(np.power(pos, 2).sum(-1))
    angles = np.arccos(a / b)  # will return 0. in one element because its the self angle
    y = np.array(radian_to_degree(angles) < angle_radian, dtype=np.float64)
    datadict = {'x': mask.astype(np.int64).squeeze(), 'mask': mask.astype(bool).squeeze(), 'pos': pos, 'y': y}
    datalist[i] = datadict
  return datalist


class SyntheticDataset(InMemoryDataset):
  def __init__(self, root, transform=None, pre_transform=None, pre_filter=None):
    super().__init__(root, transform, pre_transform, pre_filter)
    self.data, self.slices = torch.load(self.processed_paths[0])

  @property
  def processed_file_names(self):
    return ['data.pt']

  def process(self):
    root = self.root
    save_file = osp.join(root, 'raw_datalist.pth')
    data_list = torch.load(save_file)

    datalist = []
    for d in data_list:
      x = torch.from_numpy(d['x']).long()
      mask = torch.from_numpy(d['mask']).bool()
      pos = torch.from_numpy(d['pos']).float()
      y = torch.from_numpy(d['y']).float()
      if len(y) < len(x):
        y = y.unsqueeze(0)

      datalist.append(
        Data(x=x, mask=mask, pos=pos, y=y)
      )

    if self.pre_filter is not None:
      datalist = [data for data in datalist if self.pre_filter(data)]

    if self.pre_transform is not None:
      datalist = [self.pre_transform(data) for data in datalist]

    data, slices = self.collate(datalist)
    torch.save((data, slices), self.processed_paths[0])



class SyntheticDataModule(LightningDataModule):
  def __init__(self,
               kind: str,
               batch_size: int,
               num_workers: int = 1,
               ):
    super(SyntheticDataModule, self).__init__()

    if not kind in ['triangle', 'cone']:
      print("Selected wrong 'kind' argument. Exiting code")
      exit()

    self.batch_size = batch_size
    self.num_workers = num_workers
    self.kind = kind

  @property
  def num_features(self) -> int:
    return 1

  @property
  def num_classes(self) -> int:
    return 3 if self.kind == 'triangle' else 1

  def prepare_data(self):
    _ = SyntheticDataset(root=osp.join(DATA_DIR, self.kind))
    return None

  def setup(self, stage: Optional[str] = None):

    dataset = SyntheticDataset(root=osp.join(DATA_DIR, self.kind))
    n = len(dataset)
    ntrain, nval, ntest = int(0.8 * n), int(0.1 * n), int(0.1 * n)
    ntrain_ids = np.arange(ntrain)
    nval_ids = np.arange(ntrain, ntrain + nval)
    ntest_ids = np.arange(ntrain + nval, n)
    self.train_dataset = dataset.copy(idx=ntrain_ids)
    self.val_dataset = dataset.copy(idx=nval_ids)
    self.test_dataset = dataset.copy(idx=ntest_ids)


  def train_dataloader(self, shuffle: bool = True):
    return DataLoader(
      self.train_dataset,
      batch_size=self.batch_size,
      shuffle=shuffle,
      num_workers=self.num_workers,
      pin_memory=True
    )

  def val_dataloader(self, shuffle: bool = False):
    return DataLoader(
      self.val_dataset,
      batch_size=self.batch_size,
      shuffle=shuffle,
      num_workers=self.num_workers,
      pin_memory=True
    )

  def test_dataloader(self, shuffle: bool = False):
    return DataLoader(
      self.test_dataset,
      batch_size=self.batch_size,
      shuffle=shuffle,
      num_workers=self.num_workers,
      pin_memory=True
    )


if __name__ == '__main__':
  parser = ArgumentParser(
    description="Synthetic Dataset Generation Script for Triangle and Cone Classification"
  )
  parser.add_argument("--num_samples", type=int, default=50_000)
  parser.add_argument("--num_points", type=int, default=100)
  parser.add_argument("--max_r", type=float, default=10.0)
  parser.add_argument("--min_dist", type=float, default=2.0)
  parser.add_argument("--save_dir", type=str, default="data/")
  parser.add_argument("--seed", type=int, default=0)

  args = parser.parse_args()
  seed = args.seed
  save_dir = args.save_dir
  np.random.seed(seed)
  random.seed(seed)
  # Triangle
  save_dir_dataset = os.path.join(save_dir, "triangle")
  if not osp.exists(save_dir_dataset):
    os.makedirs(save_dir_dataset)
  save_file = osp.join(save_dir_dataset, "raw_datalist.pth")
  if not osp.exists(save_file):
    print(f"Generating Triangle dataset and saving into {save_file}")
    # generate triangle dataset
    dataset = generate_triangle_dataset(num_samples=args.num_samples,
                                        min_dist=args.min_dist,
                                        max_r=args.max_r,
                                        num_points=args.num_points)
    torch.save(obj=dataset, f=save_file)
  else:
    print(f"Triangle dataset already exists at {save_file}")


  dataset = SyntheticDataset(root="data/triangle/")
  print(dataset)
  n = len(dataset)
  ntrain, nval, ntest = int(0.8 * n), int(0.1 * n), int(0.1 * n)
  ntrain_ids = np.arange(ntrain)
  nval_ids = np.arange(ntrain, ntrain + nval)
  ntest_ids = np.arange(ntrain + nval, n)
  trainset = dataset.copy(idx=ntrain_ids)
  valset = dataset.copy(idx=nval_ids)
  testset = dataset.copy(idx=ntest_ids)
  trainloader = iter(DataLoader(trainset, batch_size=16))
  data = next(trainloader)
  print(data)


  datamodule = SyntheticDataModule(kind="triangle", batch_size=16, num_workers=4)
  datamodule.setup()
  trainloader = datamodule.train_dataloader()

  for data in trainloader:
    print(data)
    break
