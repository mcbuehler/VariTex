import pytorch_lightning as pl
import torch

from varitex.data.keys_enum import DataItemKey as DIK
from varitex.data.uv_factory import BFMUVFactory
from varitex.visualization.batch import CombinedVisualizer, SampledVisualizer, UVVisualizer, InterpolationVisualizer, \
    NeuralTextureVisualizer


class ImageLogCallback(pl.Callback):
    """
    Logs a variety of visualizations during training.
    """

    def __init__(self, opt, use_bfm_gpu=False):
        self.MASK_VALUE = opt.uv_mask_value
        self.opt = opt

        uv_factory_bfm = BFMUVFactory(opt, use_bfm_gpu)

        self.visualizer_combined = CombinedVisualizer(opt, mask_value=self.MASK_VALUE)
        self.visualizer_sampled = SampledVisualizer(opt, n_samples=3, mask_value=self.MASK_VALUE)
        self.visualizer_uv = UVVisualizer(opt, bfm_uv_factory=uv_factory_bfm, mask_value=self.MASK_VALUE)
        self.visualizer_interpolation = InterpolationVisualizer(opt, mask_value=self.MASK_VALUE)
        self.visualizer_neural_texture = NeuralTextureVisualizer(opt)

    def log_image(self, logger, key, vis, step):
        if self.opt.logger == "tensorboard":
            logger.experiment.add_image(key, vis, step)
        elif self.opt.logger == "wandb":
            import wandb
            logger.experiment.log(
                {key: wandb.Image(vis)}
            )

    def _combined(self, pl_module, batch, prefix):
        vis_combined = self.visualizer_combined.visualize(batch)
        self.log_image(pl_module.logger, "{}/combined".format(prefix), vis_combined, pl_module.global_step)

    def _sampled(self, pl_module, batch, batch_idx, prefix):
        vis_sampled = self.visualizer_sampled.visualize(batch, batch_idx, pl_module, std_multiplier=3)
        self.log_image(pl_module.logger, "{}/outputs_sampled".format(prefix), vis_sampled, pl_module.global_step)

        vis_sampled = self.visualizer_sampled.visualize_unseen(batch, batch_idx, pl_module, std_multiplier=2)
        self.log_image(pl_module.logger, "{}/outputs_sampled_gaussian_std2".format(prefix), vis_sampled,
                       pl_module.global_step)

    def _posed(self, pl_module, batch, batch_idx, prefix):
        deg_range = torch.arange(-45, 45 + 1, 30)
        vis = self.visualizer_uv.visualize_grid(pl_module, batch, batch_idx, deg_range)
        self.log_image(pl_module.logger, "{}/outputs_posed".format(prefix), vis, pl_module.global_step)

    def _interpolated(self, pl_module, batch, batch_idx, prefix, n=5):
        batch2 = pl_module.forward_sample_style(batch.copy(), batch_idx, std_multiplier=4)  # Random new style code
        vis = self.visualizer_interpolation.visualize(pl_module, batch, batch2, n, bidirectional=False,
                                                      include_gt=False)
        self.log_image(pl_module.logger, "{}/interpolation/random_std2".format(prefix), vis, pl_module.global_step)

        batch2 = batch.copy()
        batch2[DIK.STYLE_LATENT] = torch.zeros_like(batch[DIK.STYLE_LATENT]).to(batch[DIK.STYLE_LATENT].device)
        vis = self.visualizer_interpolation.visualize(pl_module, batch, batch2, n, bidirectional=False,
                                                      include_gt=False)
        self.log_image(pl_module.logger, "{}/interpolation/zeros".format(prefix), vis, pl_module.global_step)

        batch2 = batch.copy()
        batch2[DIK.STYLE_LATENT] = torch.randn_like(batch[DIK.STYLE_LATENT]).to(batch[DIK.STYLE_LATENT].device)
        vis = self.visualizer_interpolation.visualize(pl_module, batch, batch2, n, bidirectional=False,
                                                      include_gt=False)
        self.log_image(pl_module.logger, "{}/interpolation/standard_gaussian".format(prefix), vis,
                       pl_module.global_step)

    def _neural_texture(self, pl_module, batch, batch_idx, prefix):
        vis = self.visualizer_neural_texture.visualize_interior(batch, batch_idx)
        self.log_image(pl_module.logger, "{}/neural_texture/interior".format(prefix), vis,
                       pl_module.global_step)

        vis = self.visualizer_neural_texture.visualize_interior_sampled(batch, batch_idx)
        self.log_image(pl_module.logger, "{}/neural_texture/interior_sampled".format(prefix), vis,
                       pl_module.global_step)

        vis = self.visualizer_neural_texture.visualize_exterior_sampled(batch, batch_idx)
        self.log_image(pl_module.logger, "{}/neural_texture/exterior_sampled".format(prefix), vis,
                       pl_module.global_step)

        vis = self.visualizer_neural_texture.visualize_enhanced(batch, batch_idx)
        self.log_image(pl_module.logger, "{}/neural_texture/enhanced".format(prefix), vis,
                       pl_module.global_step)

    def log_batch(self, pl_module, batch, batch_idx, prefix):
        self._combined(pl_module, batch, prefix)
        self._sampled(pl_module, batch, batch_idx, prefix)
        self._interpolated(pl_module, batch, batch_idx, prefix)
        self._neural_texture(pl_module, batch, batch_idx, prefix)
        self._posed(pl_module, batch, batch_idx, prefix)

    def batch2gpu(self, batch):
        for k, v in batch.items():
            if isinstance(v, torch.Tensor):
                batch[k] = v.cuda()
        return batch

    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx, dataloader_idx):
        if batch_idx % self.opt.display_freq == 0:
            if trainer.gpus:
                batch = self.batch2gpu(batch)

            batch = pl_module(batch, batch_idx)
            self.log_batch(pl_module, batch, batch_idx, "train")

    def on_validation_batch_end(self, trainer, pl_module, outputs, batch, batch_idx, dataloader_idx):
        if batch_idx == 1:
            if trainer.gpus:
                batch = self.batch2gpu(batch)

            batch = pl_module(batch, batch_idx)
            self.log_batch(pl_module, batch, batch_idx, "val")
