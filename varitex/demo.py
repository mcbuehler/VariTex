import imageio
import torch

# try:
from mutil.object_dict import ObjectDict
from varitex.data.keys_enum import DataItemKey as DIK
from varitex.data.uv_factory import BFMUVFactory
from varitex.modules.pipeline import PipelineModule
from varitex.visualization.batch import CompleteVisualizer
from varitex.options import varitex_default_options
# except ModuleNotFoundError as e:
#     print(e)
#     print("Have you added VariTex to your pythonpath?")
#     print('To fix this error, go to the root path of the repository ".../VariTex/" \n '
#           'and run \n'
#           "export PYTHONPATH=$PYTHONPATH:$(pwd)")
#     exit()


class Demo:
    def __init__(self, opt):
        default_opt = varitex_default_options()
        default_opt.update(opt)
        self.opt = ObjectDict(default_opt)

        uv_factory_bfm = BFMUVFactory(opt=self.opt, use_bfm_gpu=self.opt.device == 'cuda')
        self.visualizer_complete = CompleteVisualizer(opt=self.opt, bfm_uv_factory=uv_factory_bfm)

        self.pipeline = PipelineModule.load_from_checkpoint(self.opt.checkpoint, opt=self.opt, strict=False).to(
            self.opt.device).eval()
        self.device = self.pipeline.device

    def run(self, z, sp, ep, theta, t=torch.Tensor([0, -2, -57])):
        batch = {
            DIK.STYLE_LATENT: z,
            DIK.COEFF_SHAPE: sp,
            DIK.COEFF_EXPRESSION: ep,
            DIK.T: t
        }
        batch = {k: v.to(self.device) for k, v in batch.items()}

        batch = self.visualizer_complete.visualize_single(self.pipeline, batch, 0, theta_all=theta,
                                                          forward_type='style2image')
        batch = {k: v.detach().cpu() for k, v in batch.items()}
        return batch

    def to_image(self, batch_or_batch_list):
        if isinstance(batch_or_batch_list, list):
            out = torch.cat([batch_out[DIK.IMAGE_OUT][0] for batch_out in batch_or_batch_list], -1)
        elif isinstance(batch_or_batch_list, dict):
            out = batch_or_batch_list[DIK.IMAGE_OUT][0]
        else:
            raise Warning("Invalid type: '{}'".format(type(batch_or_batch_list)))
        return self.visualizer_complete.tensor2image(out, return_format='pil')

    def to_video(self, batch_list, path_out, fps=15, quality=9, reverse=False):
        assert path_out.endswith(".mp4"), "Path should end with .mp4"
        frames = [self.to_image(batch) for batch in batch_list]
        if reverse:
            frames = frames + frames[::-1]
        imageio.mimwrite(path_out, frames, fps=fps, quality=quality)

    def load_shape_expressions(self):
        import numpy as np
        import os
        validation_indices = list(np.load(os.path.join(self.opt.dataroot_npy, "dataset_splits.npz"))["val"])
        sp = np.load(os.path.join(self.opt.dataroot_npy, "sp.npy"))[validation_indices]
        ep = np.load(os.path.join(self.opt.dataroot_npy, "ep.npy"))[validation_indices]
        return torch.Tensor(sp), torch.Tensor(ep)
