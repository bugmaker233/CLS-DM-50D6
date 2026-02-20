import os
import sys

import hydra
import lightning as pl
import torch
from lightning.pytorch.callbacks import ModelCheckpoint
from lightning.pytorch.strategies import DDPStrategy
from omegaconf import OmegaConf
from torch.utils.data import DataLoader

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), os.path.pardir)))

from dataset.monai_nii_dataset1 import AlignDataSet
from ldm.coarse_conditioner import CoarseConditioner

torch.set_float32_matmul_precision("high")


@hydra.main(config_path="../conf", config_name="config", version_base="1.3")
def train(config):
    config = config["config"]

    checkpoint_callback_best = ModelCheckpoint(
        monitor="val/voxel_loss",
        dirpath=config.hydra_path,
        filename="pl_train_coarsecond-epoch{epoch:02d}-val_voxel{val/voxel_loss:.4f}",
        save_top_k=1,
        mode="min",
        auto_insert_metric_name=False,
    )
    checkpoint_callback_latest = ModelCheckpoint(
        dirpath=config.hydra_path,
        filename="latest",
    )
    checkpoint_callback_every_10_epochs = ModelCheckpoint(
        dirpath=config.hydra_path,
        filename="epoch{epoch:03d}",
        every_n_epochs=10,
        save_top_k=-1,
        auto_insert_metric_name=False,
    )

    train_ds = AlignDataSet(config, split="train")
    val_ds = AlignDataSet(config, split="val")
    train_dl = DataLoader(
        dataset=train_ds,
        shuffle=True,
        pin_memory=True,
        drop_last=True,
        num_workers=config.num_workers,
        batch_size=config.batch_size,
    )
    val_dl = DataLoader(
        dataset=val_ds,
        shuffle=False,
        pin_memory=True,
        drop_last=True,
        num_workers=config.num_workers,
        batch_size=config.batch_size,
    )

    model = CoarseConditioner(save_path=config.hydra_path, config=config)

    trainer_kwargs = OmegaConf.to_container(config["trainer"], resolve=True)
    devices = trainer_kwargs.get("devices")
    if isinstance(devices, list) and len(devices) > 1:
        trainer_kwargs["strategy"] = DDPStrategy(find_unused_parameters=True)

    trainer = pl.Trainer(
        **trainer_kwargs,
        callbacks=[checkpoint_callback_best, checkpoint_callback_latest, checkpoint_callback_every_10_epochs],
        default_root_dir=config.hydra_path,
    )
    trainer.fit(model=model, train_dataloaders=train_dl, val_dataloaders=val_dl)


if __name__ == "__main__":
    train()

