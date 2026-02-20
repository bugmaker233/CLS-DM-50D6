import os
import sys

import hydra
import lightning as pl
import torch
from lightning.pytorch.callbacks import ModelCheckpoint
from torch.utils.data import DataLoader

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), os.path.pardir)))

from dataset.monai_nii_dataset1 import AlignDataSet
from ldm.ddpm_coarse import LatentDiffusionCoarse

torch.multiprocessing.set_sharing_strategy("file_system")
torch.set_float32_matmul_precision("high")


@hydra.main(config_path="../conf", config_name="config", version_base="1.3")
def train(config):
    config = config["config"]

    filename = "pl_train_ssldm_coarse-epoch{epoch:02d}" + f"-val_ssim{{{config['latent_diffusion'].monitor}:.3f}}"
    checkpoint_callback = ModelCheckpoint(
        monitor=config["latent_diffusion"].monitor,
        dirpath=config.hydra_path,
        filename=filename,
        save_top_k=1,
        mode="max",
        auto_insert_metric_name=False,
    )
    checkpoint_callback_latest = ModelCheckpoint(
        dirpath=config.hydra_path,
        filename="latest",
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

    model = LatentDiffusionCoarse(root_path=config.hydra_path, config=config, **config["latent_diffusion"])

    trainer = pl.Trainer(
        **config["trainer"],
        callbacks=[checkpoint_callback, checkpoint_callback_latest],
        default_root_dir=config.hydra_path,
    )
    trainer.fit(model=model, train_dataloaders=train_dl, val_dataloaders=val_dl)


if __name__ == "__main__":
    train()

