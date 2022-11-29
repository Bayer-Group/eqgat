import json
import os
import os.path as osp
from argparse import ArgumentParser
from datetime import datetime
from typing import Optional, Tuple

import numpy as np
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

from experiments.rsr.data import (
    DATA_DIR,
    GNNTransformRSR,
    LMDBDataset,
)

try:
    PATH = osp.join(osp.dirname(osp.realpath(__file__)), "data")
    MODEL_DIR = osp.join(osp.dirname(osp.realpath(__file__)), "models")
except NameError:
    PATH = "experiments/rsr/data"
    MODEL_DIR = "experiments/rsr/models"


if not osp.exists(MODEL_DIR):
    os.makedirs(MODEL_DIR)


def compute_correlations(results: pd.DataFrame, verbose: bool = True) -> dict:
    per_target = []
    for key, val in results.groupby(["target"]):
        # Ignore target with 2 decoys only since the correlations are
        # not really meaningful.
        if val.shape[0] < 3:
            continue
        true = val["true"].astype(float)
        pred = val["pred"].astype(float)
        pearson = true.corr(pred, method="pearson")
        kendall = true.corr(pred, method="kendall")
        spearman = true.corr(pred, method="spearman")
        per_target.append((key, pearson, kendall, spearman))
    per_target = pd.DataFrame(
        data=per_target, columns=["target", "pearson", "kendall", "spearman"]
    )

    res = {}
    all_true = results["true"].astype(float)
    all_pred = results["pred"].astype(float)
    res["all_pearson"] = all_true.corr(all_pred, method="pearson")
    res["all_kendall"] = all_true.corr(all_pred, method="kendall")
    res["all_spearman"] = all_true.corr(all_pred, method="spearman")

    res["per_target_pearson"] = per_target["pearson"].mean()
    res["per_target_kendall"] = per_target["kendall"].mean()
    res["per_target_spearman"] = per_target["spearman"].mean()
    if verbose:
        print(
            "\nCorrelations (Pearson, Kendall, Spearman)\n"
            "    per-target: ({:.3f}, {:.3f}, {:.3f})\n"
            "    global    : ({:.3f}, {:.3f}, {:.3f})".format(
                float(res["per_target_pearson"]),
                float(res["per_target_kendall"]),
                float(res["per_target_spearman"]),
                float(res["all_pearson"]),
                float(res["all_kendall"]),
                float(res["all_spearman"]),
            )
        )

    return res


def correlation_coeff(y_pred: Tensor,y_true: Tensor) -> Tensor:
    true_mean = y_true.mean(dim=0, keepdim=True)
    pred_mean = y_pred.mean(dim=0, keepdim=True)
    true_mean_dev = y_true - true_mean
    pred_mean_dev = y_pred - pred_mean
    a = torch.sum(true_mean_dev * pred_mean_dev)
    b = torch.rsqrt(torch.sum(true_mean_dev ** 2)) * torch.rsqrt(torch.sum(pred_mean_dev ** 2))
    corrcoeff = a * b
    return corrcoeff


class RSRDataLightning(pl.LightningDataModule):
    def __init__(
        self,
        root,
        transform,
        batch_size: int = 16,
        num_workers: int = 4,
    ):
        super(RSRDataLightning, self).__init__()

        self.root = root
        self.transform = transform
        self.batch_size = batch_size
        self.num_workers = num_workers

    @property
    def num_features(self) -> int:
        return 9

    @property
    def num_classes(self) -> int:
        return 1

    def prepare_data(self):

        _ = LMDBDataset(osp.join(self.root, "train"), transform=self.transform)

        return None

    def setup(self, stage: Optional[str] = None):

        self.train_dataset = LMDBDataset(
            osp.join(self.root, "train"), transform=self.transform
        )
        self.val_dataset = LMDBDataset(
            osp.join(self.root, "val"), transform=self.transform
        )
        self.test_dataset = LMDBDataset(
            osp.join(self.root, "test"), transform=self.transform
        )

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


class RSRmodel(pl.LightningModule):
    def __init__(
        self,
        args,
        sdim: int = 128,
        vdim: int = 16,
        depth: int = 5,
        r_cutoff: float = 4.5,
        num_radial: int = 32,
        model_type: str = "eqgat",
        learning_rate: float = 1e-4,
        weight_decay: float = 0.0,
        patience_scheduler: int = 10,
        factor_scheduler: float = 0.75,
        max_epochs: int = 30,
        use_norm: bool = False,
        aggr: str = "mean",
        alpha: float = 0.5,
    ):
        if not model_type.lower() in ["painn", "eqgat", "schnet", "segnn"]:
            print("Wrong selecte model type")
            print("Exiting code")
            exit()

        super(RSRmodel, self).__init__()
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
        self.alpha = alpha

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

        corrcoeff = correlation_coeff(y_pred, y_true)
        # calculate batch-loss
        l2_loss = self.l2(y_pred, y_true)
        with torch.no_grad():
            l1_loss = self.l1(y_pred, y_true)
        loss = self.alpha * (1.0 - corrcoeff) + (1.0 - self.alpha) * l2_loss

        self.log("train_loss", loss, on_step=True, on_epoch=False, prog_bar=True)
        self.log("train_l2", l2_loss, on_step=True, on_epoch=False, prog_bar=True)
        self.log("train_corr", corrcoeff, on_step=True, on_epoch=False, prog_bar=True)
        self.log("train_l1", l1_loss, on_step=True, on_epoch=False, prog_bar=True)

        return {"loss": loss, "l1": l1_loss, "l2": l2_loss, "corr": corrcoeff}

    def validation_step(self, batch, batch_idx):
        y_pred, y_true = self(batch)

        with torch.no_grad():
            # calculate batch-loss
            l2_loss = self.l2(y_pred, y_true)
            l1_loss = self.l1(y_pred, y_true)
            corrcoeff = correlation_coeff(y_pred, y_true)

        loss = self.alpha * (1.0 - corrcoeff) + (1.0 - self.alpha) * l2_loss

        self.log("val_loss", loss, on_step=True, on_epoch=False, prog_bar=False)
        self.log("val_l1", l1_loss, on_step=True, on_epoch=False, prog_bar=False)
        self.log("val_l2", l2_loss, on_step=True, on_epoch=False, prog_bar=False)
        self.log("val_corr", corrcoeff, on_step=True, on_epoch=False, prog_bar=False)

        return {"y_true": y_true,
                "y_pred": y_pred,
                "targets": batch.target,
                "decoys": batch.decoy
                }

    def test_step(self, batch, batch_idx):
        y_pred, y_true = self(batch)
        return {
            "y_true": y_true,
            "y_pred": y_pred,
            "targets": batch.target,
            "decoys": batch.decoy,
        }

    def training_epoch_end(self, outputs):

        avg_loss = torch.stack([x["loss"] for x in outputs]).mean().item()
        mse_loss = torch.stack([x["l2"] for x in outputs]).mean().item()
        mae_loss = torch.stack([x["l1"] for x in outputs]).mean().item()
        corr = torch.stack([x["corr"] for x in outputs]).mean().item()

        to_print = (
            f"{self.current_epoch:<10}: "
            f"MAE: {round(mae_loss, 4)}, "
            f"MSE: {round(mse_loss, 4)}, "
            f"CORR: {round(corr, 4)}, "
            f"LOSS: {round(avg_loss, 4)}"
        )
        print(" TRAIN", to_print)

        self.log("train_eloss", avg_loss, on_step=False, on_epoch=True, prog_bar=False)
        self.log("train_el1", mae_loss, on_step=False, on_epoch=True, prog_bar=False)

    def validation_epoch_end(self, outputs):
        y_pred = torch.concat([x["y_pred"] for x in outputs])
        y_true = torch.concat([x["y_true"] for x in outputs])

        mse = self.l2(y_pred, y_true).item()
        mae = self.l1(y_pred, y_true).item()

        y_true = [x["y_true"].view(-1, 1) for x in outputs]
        y_pred = [x["y_pred"].view(-1, 1) for x in outputs]

        y_true = (
            torch.cat(y_true, dim=0)
                .view(
                -1,
            )
                .detach()
                .cpu()
                .numpy()
                .tolist()
        )
        y_pred = (
            torch.cat(y_pred, dim=0)
                .view(
                -1,
            )
                .detach()
                .cpu()
                .numpy()
                .tolist()
        )

        targets = np.concatenate([x["targets"] for x in outputs]).tolist()
        decoys = np.concatenate([x["decoys"] for x in outputs]).tolist()

        arr = np.array([targets, decoys, y_true, y_pred]).T
        df = pd.DataFrame(arr, columns=["target", "decoy", "true", "pred"])
        res = compute_correlations(df, verbose=False)

        to_print = (
            f"{self.current_epoch:<10}: "
            f"MAE:{round(mae, 4)}, "
            f"MSE:{round(mse, 4)}, "
            f"Per-target Spearman R:{round(res['per_target_spearman'], 4)}, "
            f"Global Spearman R:{round(res['all_spearman'], 4)}"
        )
        print("VALID:", to_print)

        self.log("val_mean_R", res['per_target_spearman'], on_step=False, on_epoch=True, prog_bar=False)
        self.log("val_global_R", res['all_spearman'], on_step=False, on_epoch=True, prog_bar=False)
        self.log("val_eloss", mse, on_step=False, on_epoch=True, prog_bar=False)
        self.log("val_el1", mae, on_step=False, on_epoch=True, prog_bar=False)

    def test_epoch_end(self, outputs):

        y_pred = torch.concat([x["y_pred"] for x in outputs])
        y_true = torch.concat([x["y_true"] for x in outputs])

        mse = self.l2(y_pred, y_true).item()
        mae = self.l1(y_pred, y_true).item()

        to_print = (
            f"{self.current_epoch:<10}: "
            f"MSE: {round(mse, 4)}, "
            f"MAE: {round(mae, 4)}"
        )

        print(" TEST", to_print)

        y_true = [x["y_true"].view(-1, 1) for x in outputs]
        y_pred = [x["y_pred"].view(-1, 1) for x in outputs]

        y_true = (
            torch.cat(y_true, dim=0)
            .view(
                -1,
            )
            .detach()
            .cpu()
            .numpy()
            .tolist()
        )
        y_pred = (
            torch.cat(y_pred, dim=0)
            .view(
                -1,
            )
            .detach()
            .cpu()
            .numpy()
            .tolist()
        )

        targets = np.concatenate([x["targets"] for x in outputs]).tolist()
        decoys = np.concatenate([x["decoys"] for x in outputs]).tolist()

        arr = np.array([targets, decoys, y_true, y_pred]).T
        df = pd.DataFrame(arr, columns=["target", "decoy", "true", "pred"])

        self.res = compute_correlations(df)

        return mse, mae


def get_argparse():
    parser = ArgumentParser(
        description="Main training script for Equivariant GNNs on RSR Data."
    )

    # Training Setting
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--sdim", type=int, default=100)
    parser.add_argument("--vdim", type=int, default=16)
    parser.add_argument("--depth", type=int, default=5)
    parser.add_argument(
        "--model_type", type=str, default="eqgat", choices=["eqgat", "painn", "schnet", "segnn"]
    )
    parser.add_argument("--cross_ablate", default=False, action="store_true")
    parser.add_argument("--no_feat_attn", default=False, action="store_true")

    parser.add_argument("--nruns", type=int, default=3)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--num_radial", type=int, default=32)

    parser.add_argument("--num_workers", type=int, default=4)

    parser.add_argument("--learning_rate", type=float, default=1 * 1e-4)
    parser.add_argument("--batch_accum_grad", type=int, default=1)

    parser.add_argument("--weight_decay", type=float, default=0.0)
    parser.add_argument("--max_epochs", type=int, default=30)
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

    transform = GNNTransformRSR(
        cutoff=4.5,
        remove_hydrogens=True,
        max_num_neighbors=32
    )

    model = RSRmodel(
        args=args,
        sdim=args.sdim,
        vdim=args.vdim,
        depth=args.depth,
        model_type=args.model_type,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        r_cutoff=4.5,
        num_radial=args.num_radial,
        patience_scheduler=args.patience_epochs_lr,
        factor_scheduler=0.75,
        max_epochs=args.max_epochs,
        alpha=0.8
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
        datamodule = RSRDataLightning(
            root=DATA_DIR,
            transform=transform,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
        )
        model = RSRmodel(
            args=args,
            sdim=args.sdim,
            vdim=args.vdim,
            depth=args.depth,
            model_type=args.model_type,
            learning_rate=args.learning_rate,
            weight_decay=args.weight_decay,
            r_cutoff=4.5,
            num_radial=args.num_radial,
            patience_scheduler=args.patience_epochs_lr,
            factor_scheduler=0.75,
            max_epochs=args.max_epochs,
            aggr="mean",
            use_norm=True,
            alpha=0.8
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