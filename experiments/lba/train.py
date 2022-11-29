import json
import os
import os.path as osp
from argparse import ArgumentParser
from datetime import datetime
from typing import Optional, Tuple

import math
import pandas as pd
import pytorch_lightning as pl

import torch
from pytorch_lightning.callbacks import (
    LearningRateMonitor,
    ModelCheckpoint,
    ModelSummary
)
from torch import Tensor, nn
from torch_geometric.data import Batch
from torch_geometric.loader import DataLoader

from experiments.model import BaseModel, SEGNNModel

from experiments.lba.data import (
    DATA_DIR,
    CustomLBADataset,
    GNNTransformLBA,
    LMDBDataset,
)

try:
    PATH = osp.join(osp.dirname(osp.realpath(__file__)), "data")
    MODEL_DIR = osp.join(osp.dirname(osp.realpath(__file__)), "models")
except NameError:
    PATH = "experiments/lba/data"
    MODEL_DIR = "experiments/lba/models"


if not osp.exists(MODEL_DIR):
    os.makedirs(MODEL_DIR)


class LBADataLightning(pl.LightningDataModule):
    def __init__(
        self,
        root,
        transform,
        processed: bool = False,
        batch_size: int = 128,
        num_workers: int = 4,
    ):
        super(LBADataLightning, self).__init__()

        self.root = root
        self.transform = transform
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.processed = processed

    @property
    def num_features(self) -> int:
        return 9

    @property
    def num_classes(self) -> int:
        return 1

    def prepare_data(self):
        if not self.processed:
            _ = LMDBDataset(osp.join(self.root, "train"), transform=self.transform)
        else:
            _ = CustomLBADataset(split="train")
        return None

    def setup(self, stage: Optional[str] = None):
        if not self.processed:
            self.train_dataset = LMDBDataset(
                osp.join(self.root, "train"), transform=self.transform
            )
            self.val_dataset = LMDBDataset(
                osp.join(self.root, "val"), transform=self.transform
            )
            self.test_dataset = LMDBDataset(
                osp.join(self.root, "test"), transform=self.transform
            )
        else:
            self.train_dataset = CustomLBADataset(split="train")
            self.val_dataset = CustomLBADataset(split="val")
            self.test_dataset = CustomLBADataset(split="test")

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


class LBAmodel(pl.LightningModule):
    def __init__(
        self,
        args,
        sdim: int = 128,
        vdim: int = 16,
        depth: int = 5,
        r_cutoff: float = 5.0,
        num_radial: int = 32,
        model_type: str = "eqgat",
        learning_rate: float = 1e-4,
        weight_decay: float = 0.0,
        patience_scheduler: int = 10,
        factor_scheduler: float = 0.75,
        max_epochs: int = 30,
        use_norm: bool = True,
        aggr: str = "mean"
    ):
        if not model_type.lower() in ["painn", "eqgat", "schnet", "segnn"]:
            print("Wrong selecte model type")
            print("Exiting code")
            exit()

        super(LBAmodel, self).__init__()
        self.save_hyperparameters(args)

        self.sdim = sdim
        self.vdim = vdim
        self.depth = depth
        self.r_cutoff = r_cutoff
        self.num_radial = num_radial
        self.model_type = model_type.lower()
        self.learning_rate = learning_rate
        self.patience_scheduler = patience_scheduler
        self.factor_scheduler = factor_scheduler
        self.weight_decay = weight_decay
        self.max_epochs = max_epochs

        self.l1 = nn.L1Loss()
        self.l2 = nn.MSELoss()

        if model_type != "segnn":
            self.model = BaseModel(sdim=sdim,
                                   vdim=vdim,
                                   depth=depth,
                                   r_cutoff=r_cutoff,
                                   num_radial=num_radial,
                                   model_type=model_type,
                                   graph_level=True,
                                   num_elements=9,
                                   out_units=1,
                                   dropout=0.0,
                                   use_norm=use_norm,
                                   aggr=aggr,
                                   cross_ablate=args.cross_ablate,
                                   no_feat_attn=args.no_feat_attn
                                   )
        else:
            self.model = SEGNNModel(num_elements=9,
                                    out_units=1,
                                    hidden_dim=sdim + 3 * vdim + 5 * vdim // 2,
                                    lmax=2,
                                    depth=depth,
                                    graph_level=True,
                                    use_norm=False)


    def configure_optimizers(self):
        optimizer = torch.optim.Adam(
            self.parameters(), lr=self.learning_rate, weight_decay=self.weight_decay
        )

        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode="min",
            factor=self.factor_scheduler,
            patience=self.patience_scheduler,
            min_lr=1e-7,
            verbose=True,
        )

        schedulers = [
            {
                "scheduler": scheduler,
                "interval": "epoch",
                "frequency": 1,
                "monitor": "val_eloss",
            }
        ]

        return [optimizer], schedulers

    def forward(self, data: Batch) -> Tuple[Tensor, Tensor]:
        y_true = data.y.view(-1, )
        y_pred = self.model(data=data).view(-1, )
        return y_pred, y_true

    def training_step(self, batch, batch_idx):
        y_pred, y_true = self(batch)

        # calculate batch-loss
        l2_loss = self.l2(y_pred, y_true)
        with torch.no_grad():
            l1_loss = self.l1(y_pred, y_true)

        self.log("train_loss", l2_loss, on_step=True, on_epoch=False, prog_bar=True)
        self.log("train_mae", l1_loss, on_step=True, on_epoch=False, prog_bar=True)

        return {"loss": l2_loss, "l1": l1_loss}

    def validation_step(self, batch, batch_idx):
        y_pred, y_true = self(batch)

        with torch.no_grad():
            # calculate batch-loss
            l2_loss = self.l2(y_pred, y_true)
            l1_loss = self.l1(y_pred, y_true)

        self.log("val_loss", l2_loss, on_step=True, on_epoch=False, prog_bar=False)
        self.log("val_mae", l1_loss, on_step=True, on_epoch=False, prog_bar=False)

        return {"y_true": y_true,
                "y_pred": y_pred
                }

    def test_step(self, batch, batch_idx):
        y_pred, y_true = self(batch)
        return {
            "y_true": y_true,
            "y_pred": y_pred
        }

    def training_epoch_end(self, outputs):
        avg_loss = torch.stack([x["loss"] for x in outputs]).mean().item()
        mae_loss = torch.stack([x["l1"] for x in outputs]).mean().item()
        to_print = (
            f"{self.current_epoch:<10}: "
            f"MAE: {round(mae_loss, 4)}, "
            f"MSE: {round(avg_loss, 4)}, "
            f"RMSE: {round(math.sqrt(avg_loss), 4)}"
        )
        print(" TRAIN", to_print)
        self.log("train_eloss", avg_loss, on_step=False, on_epoch=True, prog_bar=False)
        self.log("train_el1", mae_loss, on_step=False, on_epoch=True, prog_bar=False)

    def validation_epoch_end(self, outputs):

        y_pred = torch.concat([x["y_pred"] for x in outputs])
        y_true = torch.concat([x["y_true"] for x in outputs])

        mse = self.l2(y_pred, y_true).item()
        mae = self.l1(y_pred, y_true).item()
        rmse = math.sqrt(mse)

        to_print = (
            f"{self.current_epoch:<10}: "
            f"MAE: {round(mae, 4)}, "
            f"MSE: {round(mse, 4)}, "
            f"RMSE: {round(rmse, 4)}"
        )

        print("VALID:", to_print)

        self.log("val_eloss", mse, on_step=False, on_epoch=True, prog_bar=False)
        self.log("val_el1", mae, on_step=False, on_epoch=True, prog_bar=False)
        self.log("val_ermse", rmse, on_step=False, on_epoch=True, prog_bar=False)

    def test_epoch_end(self, outputs):

        y_pred = torch.concat([x["y_pred"] for x in outputs])
        y_true = torch.concat([x["y_true"] for x in outputs])

        mse = self.l2(y_pred, y_true).item()
        mae = self.l1(y_pred, y_true).item()
        rmse = math.sqrt(mse)

        to_print = (
            f"{self.current_epoch:<10}: "
            f"MAE: {round(mae, 4)}, "
            f"MSE: {round(mse, 4)}, "
            f"RMSE: {round(rmse, 4)}"
        )

        print(" TEST", to_print)

        self.res = {"mse": mse, "rmse": rmse, "mae": mae}

        return mse, mae


def get_argparse():
    parser = ArgumentParser(
        description="Main training script for Equivariant GNNs on LBA Data."
    )

    # Training Setting
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--sdim", type=int, default=100)
    parser.add_argument("--vdim", type=int, default=16)
    parser.add_argument("--depth", type=int, default=3)
    parser.add_argument(
        "--model_type", type=str, default="eqgat", choices=["eqgat", "painn", "schnet", "segnn"]
    )
    parser.add_argument("--cross_ablate", default=False, action="store_true")
    parser.add_argument("--no_feat_attn", default=False, action="store_true")

    parser.add_argument("--nruns", type=int, default=3)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--num_radial", type=int, default=32)

    parser.add_argument("--num_workers", type=int, default=4)

    parser.add_argument("--learning_rate", type=float, default=1e-4)
    parser.add_argument("--batch_accum_grad", type=int, default=1)

    parser.add_argument("--weight_decay", type=float, default=0.0)
    parser.add_argument("--max_epochs", type=int, default=20)
    parser.add_argument("--gradient_clip_val", type=float, default=10)
    parser.add_argument("--patience_epochs_lr", type=int, default=10)

    parser.add_argument("--save_dir", type=str, default="base_eqgat")
    parser.add_argument("--device", default="0", type=str)
    parser.add_argument("--load_ckpt", default=None)

    args = parser.parse_args()

    return args


if __name__ == "__main__":

    args = get_argparse()
    device = args.device
    if device != "cpu":
        device = int(device)

    transform = GNNTransformLBA(
        cutoff=4.5,
        remove_hydrogens=False,
        max_num_neighbors=32
    )

    model = LBAmodel(
        args=args,
        sdim=args.sdim,
        vdim=args.vdim,
        depth=args.depth,
        model_type=args.model_type,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        r_cutoff=4.5,
        num_radial=args.num_radial,
        max_epochs=args.max_epochs,
        factor_scheduler=0.75,
    )
    print(
        f"Model consists of {sum(p.numel() for p in model.parameters() if p.requires_grad)} trainable params."
    )

    model_dir = osp.join(MODEL_DIR, args.save_dir)
    if not osp.exists(model_dir):
        os.makedirs(model_dir)

    run_results = []
    seed = args.seed
    for run in range(args.nruns):
        pl.seed_everything(seed, workers=True)
        seed += run
        print(f"Starting run {run} with seed {seed}")
        datamodule = LBADataLightning(
            root=DATA_DIR,
            transform=transform,
            processed=True,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
        )
        model = LBAmodel(
            args=args,
            sdim=args.sdim,
            vdim=args.vdim,
            depth=args.depth,
            model_type=args.model_type,
            learning_rate=args.learning_rate,
            weight_decay=args.weight_decay,
            r_cutoff=4.5,
            num_radial=args.num_radial,
            max_epochs=args.max_epochs,
            factor_scheduler=0.75,
            aggr="mean",
            use_norm=False
        )

        checkpoint_callback = ModelCheckpoint(
            monitor="val_eloss",
            filename="model-{epoch:02d}-{val_eloss:.4f}",
            mode="min",
            save_last=False,
            verbose=True,
        )

        trainer = pl.Trainer(
            devices=[device] if device != "cpu" else None,
            accelerator="gpu" if device != "cpu" else "cpu",
            max_epochs=args.max_epochs,
            precision=32,
            amp_backend="native",
            callbacks=[
                checkpoint_callback,
                LearningRateMonitor(),
                ModelSummary(max_depth=2)
            ],
            default_root_dir=model_dir,
            gradient_clip_val=args.gradient_clip_val,
            accumulate_grad_batches=args.batch_accum_grad,
        )

        start_time = datetime.now()

        trainer.fit(model, datamodule=datamodule, ckpt_path=args.load_ckpt)

        end_time = datetime.now()
        time_diff = end_time - start_time
        print(f"Training time: {time_diff}")

        # running test set
        _ = trainer.test(ckpt_path="best", datamodule=datamodule)
        res = model.res
        run_results.append(res)

    with open(os.path.join(model_dir, "res.json"), "w") as f:
        json.dump(run_results, f)

    # aggregate over runs..
    results_df = pd.DataFrame(run_results)
    print(results_df.describe())