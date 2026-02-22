from typing import List

import lightning as pl
import torch
import torch.nn as nn
import torch.nn.functional as F

from ldm.autoencoderkl.autoencoder import AutoencoderKL


def disabled_train(self, mode=True):
    return self


class CoarseConditioner(pl.LightningModule):
    """
    Stage-2 conditioner:
    1) pixel-wise weighting over multi-view X-ray inputs
    2) fused 2D feature encoding
    3) coarse 3D reconstruction
    4) AE latent export for Stage-3 conditioning

    Training supervision is voxel-space only.
    """

    def __init__(self, save_path: str, config):
        super().__init__()
        self.root_path = save_path
        self.config = config

        self.cond_nums = list(config.cond_nums)
        self.cond_num = int(1 in self.cond_nums) + int(2 in self.cond_nums) + int(3 in self.cond_nums)
        if self.cond_num == 0:
            raise ValueError("cond_nums must contain at least one of [1, 2, 3].")

        self.sync_dist = bool(config.model.sync_dist)
        self.learning_rate = float(config.model.base_learning_rate)
        self.voxel_loss = str(getattr(config, "voxel_loss", "l1")).lower()
        self.weight_temperature = float(getattr(config, "weight_temperature", 1.0))
        self.weight_smooth_ratio = float(getattr(config, "weight_smooth_ratio", 0.0))
        self.weight_sparse_ratio = float(getattr(config, "weight_sparse_ratio", 0.0))

        self.target_depth = int(config.ct_channel)
        self.target_hw = int(config.fine_size)

        coarse_cfg = config.coarse_model
        base_channels = int(coarse_cfg.base_channels)
        fusion_channels = int(coarse_cfg.fusion_channels)
        volume_channels = int(coarse_cfg.volume_channels)
        depth_bins = int(coarse_cfg.depth_bins)
        self._volume_channels = volume_channels
        self._depth_bins = depth_bins

        self.weight_net = nn.Sequential(
            nn.Conv2d(self.cond_num, base_channels, kernel_size=3, padding=1),
            nn.SiLU(),
            nn.Conv2d(base_channels, base_channels, kernel_size=3, padding=1),
            nn.SiLU(),
            nn.Conv2d(base_channels, self.cond_num, kernel_size=1),
        )

        self.encoder2d = nn.Sequential(
            nn.Conv2d(self.cond_num, base_channels, kernel_size=3, stride=1, padding=1),
            nn.SiLU(),
            nn.Conv2d(base_channels, base_channels * 2, kernel_size=4, stride=2, padding=1),  # 128 -> 64
            nn.SiLU(),
            nn.Conv2d(base_channels * 2, base_channels * 4, kernel_size=4, stride=2, padding=1),  # 64 -> 32
            nn.SiLU(),
            nn.Conv2d(base_channels * 4, fusion_channels, kernel_size=4, stride=2, padding=1),  # 32 -> 16
            nn.SiLU(),
        )
        self.fusion = nn.Sequential(
            nn.Conv2d(fusion_channels, fusion_channels, kernel_size=3, padding=1),
            nn.SiLU(),
            nn.Conv2d(fusion_channels, fusion_channels, kernel_size=3, padding=1),
            nn.SiLU(),
        )
        self.lift_to_3d = nn.Conv2d(
            fusion_channels,
            volume_channels * depth_bins,
            kernel_size=1,
        )

        self.decoder3d = nn.Sequential(
            nn.ConvTranspose3d(volume_channels, volume_channels // 2, kernel_size=4, stride=2, padding=1),  # 16 -> 32
            nn.SiLU(),
            nn.ConvTranspose3d(volume_channels // 2, volume_channels // 4, kernel_size=4, stride=2, padding=1),  # 32 -> 64
            nn.SiLU(),
            nn.ConvTranspose3d(volume_channels // 4, volume_channels // 8, kernel_size=4, stride=2, padding=1),  # 64 -> 128
            nn.SiLU(),
            nn.Conv3d(volume_channels // 8, 1, kernel_size=3, padding=1),
            nn.Tanh(),
        )

        self.init_ae_model(save_path, config)

    def init_ae_model(self, save_path, config):
        model = AutoencoderKL(save_path=save_path, config=config, **config["model"])
        model.init_from_ckpt(config.ae_ckpt)
        model.eval()
        model.train = disabled_train
        for p in model.parameters():
            p.requires_grad = False
        self.ae_model = model

    @property
    def cond_stage_model(self):
        return self

    @property
    def encoder(self):
        return self.ae_model.encoder

    @property
    def decoder(self):
        return self.ae_model.decoder

    @property
    def encode(self):
        return self.ae_model.encode

    @property
    def decode(self):
        return self.ae_model.decode

    @property
    def quant_conv(self):
        return self.ae_model.quant_conv

    @property
    def post_quant_conv(self):
        return self.ae_model.post_quant_conv

    def init_from_ckpt(self, path, ignore_keys: List[str] = None):
        ignore_keys = ignore_keys or []
        sd = torch.load(path, map_location="cpu")
        if "state_dict" in sd:
            sd = sd["state_dict"]
        keys = list(sd.keys())
        for k in keys:
            for ik in ignore_keys:
                if k.startswith(ik):
                    del sd[k]
        self.load_state_dict(sd, strict=False)
        print(f"Restored from {path}")

    def _stack_conditions(self, batch):
        conds = []
        if 1 in self.cond_nums:
            conds.append(torch.as_tensor(batch["cond1"]))
        if 2 in self.cond_nums:
            conds.append(torch.as_tensor(batch["cond2"]))
        if 3 in self.cond_nums:
            conds.append(torch.as_tensor(batch["cond3"]))
        return torch.cat(conds, dim=1)

    def get_cond(self, batch, _type="cat"):
        return self._stack_conditions(batch)

    def _pixel_weight(self, cond):
        logits = self.weight_net(cond)
        # Independent gate per view/channel. This removes the sum-to-one constraint
        # and allows jointly suppressing non-spine regions.
        weights = torch.sigmoid(logits / max(self.weight_temperature, 1e-6))
        weighted = cond * weights
        return weighted, weights

    def _decode_coarse_volume(self, weighted_cond):
        feat = self.encoder2d(weighted_cond)
        feat = self.fusion(feat)
        lifted = self.lift_to_3d(feat)
        b, _, h, w = lifted.shape
        lifted = lifted.view(b, self._volume_channels, self._depth_bins, h, w)
        coarse = self.decoder3d(lifted)
        coarse = F.interpolate(
            coarse,
            size=(self.target_depth, self.target_hw, self.target_hw),
            mode="trilinear",
            align_corners=False,
        )
        return coarse

    def forward(self, cond):
        weighted_cond, weights = self._pixel_weight(cond)
        coarse = self._decode_coarse_volume(weighted_cond)
        return coarse, weights

    def encode_condition_from_batch(self, batch, device=None, sample=False):
        cond = self._stack_conditions(batch)
        if device is not None:
            cond = cond.to(device)
        coarse, _ = self(cond)
        posterior = self.ae_model.encode(coarse)
        return posterior.sample() if sample else posterior.mode()

    def _smoothness_loss(self, weights):
        dh = (weights[:, :, 1:, :] - weights[:, :, :-1, :]).abs().mean()
        dw = (weights[:, :, :, 1:] - weights[:, :, :, :-1]).abs().mean()
        return dh + dw

    def _voxel_recon_loss(self, pred, target):
        if self.voxel_loss == "mse":
            return F.mse_loss(pred, target)
        return F.l1_loss(pred, target)

    def _shared_step(self, batch, stage: str):
        target = torch.as_tensor(batch["image"])
        cond = self._stack_conditions(batch)
        coarse, weights = self(cond)

        voxel_loss = self._voxel_recon_loss(coarse, target)
        smooth_loss = self._smoothness_loss(weights)
        sparse_loss = weights.mean()
        loss = voxel_loss + self.weight_smooth_ratio * smooth_loss + self.weight_sparse_ratio * sparse_loss

        self.log(f"{stage}/loss", loss, prog_bar=True, logger=True, on_step=(stage == "train"), on_epoch=True, sync_dist=self.sync_dist)
        self.log(
            f"{stage}/voxel_loss",
            voxel_loss,
            prog_bar=True,
            logger=True,
            on_step=(stage == "train"),
            on_epoch=True,
            sync_dist=self.sync_dist,
        )
        self.log(
            f"{stage}/weight_smooth",
            smooth_loss,
            prog_bar=False,
            logger=True,
            on_step=False,
            on_epoch=True,
            sync_dist=self.sync_dist,
        )
        self.log(
            f"{stage}/weight_mean",
            weights.mean(),
            prog_bar=False,
            logger=True,
            on_step=False,
            on_epoch=True,
            sync_dist=self.sync_dist,
        )
        self.log(
            f"{stage}/weight_sparse",
            sparse_loss,
            prog_bar=False,
            logger=True,
            on_step=False,
            on_epoch=True,
            sync_dist=self.sync_dist,
        )
        self.log(
            f"{stage}/weight_sum_mean",
            weights.sum(dim=1).mean(),
            prog_bar=False,
            logger=True,
            on_step=False,
            on_epoch=True,
            sync_dist=self.sync_dist,
        )
        return loss

    def training_step(self, batch, batch_idx):
        return self._shared_step(batch, "train")

    def validation_step(self, batch, batch_idx):
        self._shared_step(batch, "val")

    def test_step(self, batch, batch_idx):
        self._shared_step(batch, "test")

    def configure_optimizers(self):
        trainable_params = [p for p in self.parameters() if p.requires_grad]
        return torch.optim.AdamW(trainable_params, lr=self.learning_rate)
