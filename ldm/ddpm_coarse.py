import torch

from .coarse_conditioner import CoarseConditioner
from .ddpm import LatentDiffusion, disabled_train


class LatentDiffusionCoarse(LatentDiffusion):
    def get_first_stage_encoding(self, encoder_posterior):
        try:
            return super().get_first_stage_encoding(encoder_posterior)
        except NotImplementedError:
            # Compatibility fallback: class identity may differ across module paths.
            if hasattr(encoder_posterior, "sample"):
                z = encoder_posterior.sample()
            elif torch.is_tensor(encoder_posterior):
                z = encoder_posterior
            else:
                raise
            return self.scale_factor * z

    def instantiate_first_stage_and_cond_stage(self, config, global_config):
        cond_flag = str(getattr(global_config, "cond_flag", ""))
        if cond_flag != "coarse":
            return super().instantiate_first_stage_and_cond_stage(config, global_config)

        model = CoarseConditioner(save_path=self.config.hydra_path, config=config)
        ckpt_path = str(getattr(config, "coarsecond_ckpt", ""))
        if ckpt_path and ckpt_path != "path":
            model.init_from_ckpt(ckpt_path)
        self.first_stage_model = model.eval()
        self.first_stage_model.train = disabled_train
        for p in self.first_stage_model.parameters():
            p.requires_grad = False

    @torch.no_grad()
    def get_input(
        self,
        batch,
        k,
        return_first_stage_outputs=False,
        force_c_encode=False,
        cond_key=None,
        return_original_cond=False,
        bs=None,
    ):
        x = torch.as_tensor(batch["image"])
        if bs is not None:
            x = x[:bs]
        x = x.to(self.device)

        encoder_posterior = self.encode_first_stage(x)
        try:
            z = self.get_first_stage_encoding(encoder_posterior).detach()
        except NotImplementedError:
            # Compatibility fallback: some runs load the Gaussian posterior class
            # from a different module path, making isinstance checks fail.
            if hasattr(encoder_posterior, "sample"):
                z = (self.scale_factor * encoder_posterior.sample()).detach()
            elif torch.is_tensor(encoder_posterior):
                z = (self.scale_factor * encoder_posterior).detach()
            else:
                raise

        c = self.first_stage_model.encode_condition_from_batch(batch, device=self.device, sample=False).detach()
        if bs is not None:
            c = c[:bs]

        out = [z, c]
        if return_first_stage_outputs:
            xrec = self.decode_first_stage(z)
            out.extend([x, xrec])

        if return_original_cond:
            cond1 = torch.as_tensor(batch["cond1"]) if "cond1" in batch else None
            cond2 = torch.as_tensor(batch["cond2"]) if "cond2" in batch else None
            cond3 = torch.as_tensor(batch["cond3"]) if "cond3" in batch else None
            if bs is not None:
                cond1 = cond1[:bs] if cond1 is not None else None
                cond2 = cond2[:bs] if cond2 is not None else None
                cond3 = cond3[:bs] if cond3 is not None else None
            out.extend([cond1, cond2, cond3])

        return out
