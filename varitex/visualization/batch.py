import numpy as np
import torch
import torchvision.transforms
from torch.nn.functional import grid_sample
from torchvision.utils import make_grid

from mutil.data_types import to_np
from mutil.np_util import interpolation
from mutil.pytorch_utils import ImageNetNormalizeTransformInverse, to_tensor, theta2rotation_matrix
from varitex.data.keys_enum import DataItemKey as DIK


class Visualizer:
    def __init__(self, opt, n_samples=1, mask_value=0, return_format='torch', mask_key=None, device='cuda'):
        self.opt = opt
        self.dataset = self.opt.dataset
        self.n_samples = n_samples
        self.mask_value = mask_value
        self.unnormalize = ImageNetNormalizeTransformInverse()
        self.return_format = return_format
        self.mask_key = mask_key
        self.device = device

    def uv2rgb(self, uv_image):
        assert len(uv_image.shape) == 3, "Invalid shape, should have 3 channels."
        uv_image = uv_image.clone().permute(2, 0, 1)  # HWC to CHW
        uv_image_rgb = [uv_image, -1 * torch.ones((1, uv_image.shape[1], uv_image.shape[2])).to(uv_image.device)]
        uv_image_rgb = torch.cat(uv_image_rgb, 0)
        uv_image_rgb = (uv_image_rgb + 1) / 2
        return uv_image_rgb

    def tensor2image(self, image_tensor, mask=None, clamp=True, batch=None, white_bg=True, return_format=None):
        image = self.unnormalize(image_tensor)
        if batch is not None:
            image = self.mask(image, batch, white_bg=white_bg)
        if mask is not None:
            image[~mask] = self.mask_value
        if clamp:
            image = image.clamp(0, 1)
        out = image.detach().cpu()
        if return_format is not None:
            out = self.format_output(out, return_format)
        return out

    def format_output(self, vis, return_format=None):
        if return_format is None:
            return_format = self.return_format

        if return_format == "torch":
            return vis
        elif return_format == "pil":
            to_pil = torchvision.transforms.ToPILImage()
            vis = vis.detach().cpu()
            return to_pil(vis)
        elif return_format == "np":
            return np.array(self.format_output(vis, return_format='pil'))
        raise Warning("Invalid return format: {}".format(return_format))

    def _sample(self, batch, std_multiplier):
        q = torch.distributions.Normal(batch[DIK.STYLE_LATENT_MU], batch[DIK.STYLE_LATENT_STD] * std_multiplier)
        z = q.rsample()
        batch[DIK.STYLE_LATENT] = z
        return batch

    def detach(self, o, to_cpu=False):
        if isinstance(o, list):
            o = [self.detach(o_i, to_cpu) for o_i in o]
        elif isinstance(o, dict):
            o = {k: self.detach(v, to_cpu) for k, v in o.items()}
        elif isinstance(o, torch.Tensor):
            o = o.clone().detach()
            if to_cpu:
                o = o.cpu()
        return o

    def _debug_view(self, vis):
        import matplotlib.pyplot as plt
        vis = self.format_output(vis, 'np')
        plt.imshow(np.array(vis))
        plt.show()

    def _placeholder(self, like):
        return torch.zeros_like(like, device=like.device)

    def mask(self, img_out, batch, t=0.7, white_bg=False):
        if self.mask_key is not None:
            mask = batch[self.mask_key][0].expand_as(img_out).to(img_out.device)
            mask[mask < t] = 0
            img_out = img_out * mask
            if white_bg:
                img_out[~mask.bool()] = 1
        return img_out


class CombinedVisualizer(Visualizer):
    def zoom(self, image, factor=3):
        assert len(image.shape) == 3, "No batch dim plz"
        h, w = self.opt.image_h, self.opt.image_w
        center_y, center_x = h // factor, w // factor
        image_cropped = image[:, center_y:center_y + h // factor, center_x:center_x + w // factor]
        zoomed = torch.nn.functional.interpolate(image_cropped.unsqueeze(0), image.shape[-2:], mode='bilinear').clamp(0,
                                                                                                                      1)
        return zoomed.squeeze()

    def visualize(self, batch):
        batch = self.detach(batch)
        image_in = self.tensor2image(batch[DIK.IMAGE_IN][0])
        image_in_encode = self.tensor2image(batch[DIK.IMAGE_IN_ENCODE][0])
        uv_image = self.uv2rgb(batch[DIK.UV_RENDERED][0])
        segmentation_pred = batch[DIK.SEGMENTATION_PREDICTED][0][0]  # zero is background
        segmentation_pred = segmentation_pred.expand_as(image_in)
        segmentation_gt = batch[DIK.SEGMENTATION_MASK][0].expand_as(image_in)
        image_out = self.tensor2image(batch[DIK.IMAGE_OUT][0])
        zoomed_out = self.zoom(image_out)
        zoomed_in = self.zoom(image_in)

        vis = [uv_image, segmentation_gt, image_in, zoomed_in,
               image_in_encode, segmentation_pred, image_out, zoomed_out
               ]
        vis = self.detach(vis, to_cpu=True)
        vis = make_grid(vis, nrow=4)
        vis = self.format_output(vis)
        return vis


class SampledVisualizer(Visualizer):
    def visualize(self, batch, batch_idx, pipeline, std_multiplier=1):
        batch = self.detach(batch)
        images_out = list()
        for s_i in range(self.n_samples):
            batch2 = batch.copy()
            batch2 = self._sample(batch2, std_multiplier)
            batch2 = pipeline.forward_latent2image(batch2, batch_idx)
            img_out = batch2[DIK.IMAGE_OUT]
            img_out = img_out[0]
            img_out = self.tensor2image(img_out)

            if self.mask_key is not None:
                img_out = img_out * batch[self.mask_key][0].expand_as(img_out).cpu()
            images_out.append(img_out)
        images_out = self.detach(images_out, to_cpu=True)
        vis = make_grid(images_out, nrow=len(images_out))
        return self.format_output(vis)

    def visualize_unseen(self, batch, batch_idx, pipeline, std_multiplier=1):
        batch = self.detach(batch)
        images_out = list()
        for s_i in range(self.n_samples):
            batch2 = batch.copy()
            batch2[DIK.STYLE_LATENT] = torch.randn_like(batch2[DIK.STYLE_LATENT]).to(
                batch2[DIK.STYLE_LATENT].device) * std_multiplier
            batch2 = pipeline.forward_latent2image(batch2, batch_idx)
            img_out = batch2[DIK.IMAGE_OUT]
            img_out = img_out[0]
            img_out = self.tensor2image(img_out)

            if self.mask_key is not None:
                img_out = img_out * batch[self.mask_key][0].expand_as(img_out).cpu()
            images_out.append(img_out)
        images_out = self.detach(images_out, to_cpu=True)
        vis = make_grid(images_out, nrow=len(images_out))
        return self.format_output(vis)

    def visualize_grid(self, batch, batch_idx, pipeline, std_multipliers=(1, 2, 3, 4)):
        sampled_vis = list()
        for multiplier in std_multipliers:
            vis = self.visualize(batch, 0, pipeline, std_multiplier=multiplier)
            sampled_vis.append(vis)
        vis = torch.cat(sampled_vis, -2)
        vis = torch.clamp(vis, 0, 1)
        return self.format_output(vis)

    def visualize_new_sample(self, batch, batch_idx, pipeline, std_multiplier):
        batch = pipeline.forward_sample_style(batch.copy(), batch_idx, std_multiplier)
        batch = pipeline.forward_latent2image(batch, batch_idx)
        vis = self.tensor2image(batch[DIK.IMAGE_OUT][0], batch=batch)
        return self.format_output(vis)


class UVVisualizer(Visualizer):

    def __init__(self, *args, bfm_uv_factory, **kwargs):
        super().__init__(*args, **kwargs)
        self.bfm_uv_factory = bfm_uv_factory

    def run(self, pipeline, batch, batch_idx, return_format=None, forward_type="style2image"):
        """

        Args:
            pipeline:
            batch:
            batch_idx:
            return_format:
            forward_type: include "fullbatch" if the function should return a tuple img, batch

        Returns:

        """
        if "interior2image" in forward_type:
            batch = pipeline.forward_interior2image(batch, batch_idx)
        elif "style2image" in forward_type:
            batch = pipeline.forward_latent2image(batch, batch_idx)
        elif "texture2image" in forward_type:
            batch = pipeline.forward_texture2image(batch, batch_idx)
        else:
            raise Warning("invalid forward type")
        img_out = batch[DIK.IMAGE_OUT][0]
        img_out = self.tensor2image(img_out)
        img_out = self.mask(img_out, batch, white_bg=False)
        # img_out = self.uv2rgb(batch[DIK.UV_RENDERED][0])
        if return_format is not None:
            img_out = self.format_output(img_out, return_format=return_format)
        if "fullbatch" in forward_type:
            return img_out, batch
        else:
            return img_out

    def get_neutral_t(self, batch):
        t = torch.Tensor((0, 0, batch[DIK.T][0, 2])).expand_as(batch[DIK.T])
        return t

    def visualize_grid(self, pipeline, batch, batch_idx, deg_range, return_format=None):
        result = list()
        idx = 0

        for theta_y in deg_range:
            for theta_x in deg_range:
                batch2 = batch.copy()
                theta = [theta_x, theta_y, 0]
                R = theta2rotation_matrix(theta_all=theta).to(batch2[DIK.R].device).unsqueeze(0)
                t = self.get_neutral_t(batch)
                uv = self.bfm_uv_factory.getUV(R=R, t=t, s=batch2[DIK.SCALE], sp=batch2[DIK.COEFF_SHAPE],
                                               ep=batch2[DIK.COEFF_EXPRESSION])

                batch2[DIK.UV_RENDERED] = uv.expand_as(batch[DIK.UV_RENDERED]).to(batch[DIK.UV_RENDERED].device)
                img_out = self.run(pipeline, batch2, idx)
                result.append(img_out)

        vis = make_grid(result, nrow=int(np.sqrt(len(result))))
        return self.format_output(vis, return_format=return_format)

    def visualize_row(self, pipeline, batch, batch_idx, deg_range, axis=1, return_format=None):
        if not isinstance(axis, list):
            axis = [axis]
        result = list()
        idx = 0
        if 1 in axis:
            deg_range = -1 * deg_range
        for theta_d in deg_range:
            batch2 = batch.copy()
            batch_size = batch2[DIK.R].shape[0]
            theta = [0, 0, 0]
            for i in range(len(axis)):
                theta[axis[i]] = theta_d
            R = theta2rotation_matrix(theta_all=theta).to(batch2[DIK.R].device).unsqueeze(0)
            t = self.get_neutral_t(batch)
            uv = self.bfm_uv_factory.getUV(R=R, t=t, s=batch2[DIK.SCALE], sp=batch2[DIK.COEFF_SHAPE],
                                           ep=batch2[DIK.COEFF_EXPRESSION])

            batch2[DIK.UV_RENDERED] = uv.expand_as(batch[DIK.UV_RENDERED]).to(batch[DIK.UV_RENDERED].device)
            img_out = self.run(pipeline, batch2, idx)
            result.append(img_out)

        nrow = len(result) if 1 in axis else 1
        vis = make_grid(result, nrow=nrow)
        return self.format_output(vis, return_format=return_format)

    def visualize_show(self, pipeline, batch, batch_idx, n_y, n_x, pose_range_x=30, pose_range_y=30):
        theta_x = torch.linspace(-pose_range_x, pose_range_x, n_x) if pose_range_x > 0 else [0]
        theta_y = torch.linspace(-pose_range_y, pose_range_y, n_y) if pose_range_y > 0 else [0]
        top = [[theta_x[0], theta_y[i], 0] for i in range(n_y - 1)] if n_y > 0 else []
        down = [[theta_x[i], theta_y[-1], 0] for i in range(n_x - 1)] if n_x > 0 else []
        bottom = [[theta_x[-1], theta_y[-i - 1], 0] for i in range(n_y - 1)] if n_y > 0 else []
        up = [[theta_x[-i - 1], theta_y[0], 0] for i in range(n_x - 1)] if n_x > 0 else []
        all = top + down + bottom + up

        all = [theta2rotation_matrix(theta_all=theta) for theta in all]

        result = list()
        for i, R in enumerate(all):
            batch2 = batch.copy()
            t = batch2[DIK.T]
            uv = self.bfm_uv_factory.getUV(sp=batch[DIK.COEFF_SHAPE], ep=batch[DIK.COEFF_EXPRESSION], R=R, t=t)
            batch2[DIK.UV_RENDERED] = uv.expand_as(batch2[DIK.UV_RENDERED]).to(batch2[DIK.UV_RENDERED].device)
            img_out = self.run(pipeline, batch2, batch_idx, return_format='np')
            result.append(img_out)
        return result

    def visualize_show_expression(self, pipeline, batch, batch_idx, n, expression_list):
        result = list()
        dev = batch[DIK.STYLE_LATENT].device
        R = torch.eye(3).unsqueeze(0)
        ep_prev = expression_list[0]
        expression_list = list(expression_list) + [expression_list[0]]
        for i in range(1, len(expression_list)):
            for ep in interpolation(n, ep_prev, expression_list[i]):
                ep = torch.Tensor(ep).unsqueeze(0).to(dev)
                R = R.to(dev)
                batch2 = batch.copy()
                uv = self.bfm_uv_factory.getUV(sp=batch[DIK.COEFF_SHAPE], ep=ep, R=R, t=batch[DIK.T],
                                               s=batch[DIK.SCALE])
                batch2[DIK.UV_RENDERED] = uv.expand_as(batch2[DIK.UV_RENDERED]).to(batch2[DIK.UV_RENDERED].device)
                img_out = self.run(pipeline, batch2, batch_idx, return_format='np')
                result.append(img_out)
            ep_prev = expression_list[i]
        return result

    def visualize_single(self, pipeline, batch, batch_idx, theta_all=None, R=None, forward_type='style2image'):
        batch2 = self.detach(batch)
        if R is None:
            R = theta2rotation_matrix(theta_all=theta_all)
        uv = self.bfm_uv_factory.getUV(sp=batch[DIK.COEFF_SHAPE], ep=batch[DIK.COEFF_EXPRESSION], R=R, t=batch[DIK.T])
        batch2[DIK.UV_RENDERED] = uv.expand_as(batch2[DIK.UV_RENDERED]).to(batch2[DIK.UV_RENDERED].device)
        out = self.run(pipeline, batch2, batch_idx, return_format='pil', forward_type=forward_type)
        return out

    def visualize_row_expressions(self, pipeline, batch, batch_idx,
                                  list_ep, return_format=None):
        result = list()
        for i in range(len(list_ep)):
            batch2 = batch.copy()
            t = torch.Tensor((0, 0, batch[DIK.T][0, 2])).expand_as(batch[DIK.T])
            R = torch.eye(3).unsqueeze(0)
            ep = list_ep[i]
            uv = self.bfm_uv_factory.getUV(R=R, t=t, s=batch2[DIK.SCALE], sp=batch2[DIK.COEFF_SHAPE], ep=ep)
            batch2[DIK.UV_RENDERED] = uv.expand_as(batch[DIK.UV_RENDERED]).to(batch[DIK.UV_RENDERED].device)
            img_out = self.run(pipeline, batch2, batch_idx)
            result.append(img_out)

        nrow = len(result)
        vis = make_grid(result, nrow=nrow)
        return self.format_output(vis, return_format=return_format)

    def visualize_row_shapes(self, pipeline, batch, batch_idx,
                             list_sp, return_format=None):
        result = list()
        for i in range(len(list_sp)):
            batch2 = batch.copy()
            t = torch.Tensor((0, 0, batch[DIK.T][0, 2])).expand_as(batch[DIK.T])
            R = torch.eye(3).unsqueeze(0)
            sp = list_sp[i]
            uv = self.bfm_uv_factory.getUV(R=R, t=t, s=batch2[DIK.SCALE], ep=batch2[DIK.COEFF_EXPRESSION], sp=sp)
            batch2[DIK.UV_RENDERED] = uv.expand_as(batch[DIK.UV_RENDERED]).to(batch[DIK.UV_RENDERED].device)
            img_out = self.run(pipeline, batch2, batch_idx)
            result.append(img_out)

        nrow = len(result)
        vis = make_grid(result, nrow=nrow)
        return self.format_output(vis, return_format=return_format)


class InterpolationVisualizer(Visualizer):

    def run(self, pipeline, batch, latent_from, latent_to, n):
        result = list()
        # linear interpolation
        all_latents = interpolation(n, latent_from=latent_from, latent_to=latent_to)
        all_latents = to_tensor(all_latents, batch[DIK.STYLE_LATENT].device)
        for latent in all_latents:
            batch2 = batch.copy()
            batch2[DIK.STYLE_LATENT] = latent.reshape(batch[DIK.STYLE_LATENT].shape)
            batch2 = pipeline.forward_latent2image(batch2, 0)
            img_out = batch2[DIK.IMAGE_OUT][0]
            img_out = self.tensor2image(img_out)

            if self.mask_key is not None:
                img_out = img_out * batch[self.mask_key][0].expand_as(img_out).cpu()
            result.append(img_out)
        return result

    def visualize(self, pipeline, batch_from, batch_to, n, bidirectional=True, include_gt=True):
        latent_from = to_np(batch_from[DIK.STYLE_LATENT])
        latent_to = to_np(batch_to[DIK.STYLE_LATENT])
        result = self.run(pipeline, batch_from, latent_from, latent_to, n)

        if include_gt:
            n = n + 2
            result = [self.tensor2image(batch_from[DIK.IMAGE_IN][0])] + result + [
                self.tensor2image(batch_to[DIK.IMAGE_IN][0])]

        if bidirectional:
            result_backward = self.run(pipeline, batch_to, latent_from, latent_to, n)

            if include_gt:
                result_backward = [self.tensor2image(batch_from[DIK.IMAGE_IN][0])] + result_backward + [
                    self.tensor2image(batch_to[DIK.IMAGE_IN][0])]

            result = result + result_backward

        vis = make_grid(result, nrow=n)
        return self.format_output(vis)


class NeuralTextureVisualizer(Visualizer):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def run(self, texture, return_format=None):
        n_channels = texture.shape[0]
        result = list()
        for c in range(n_channels):
            texture_c = texture[c].unsqueeze(0)
            result.append(texture_c)

        vis = make_grid(result, nrow=int(np.sqrt(n_channels)), scale_each=True, normalize=True)
        vis = self.format_output(vis, return_format=return_format)
        return vis

    def visualize_interior(self, batch, batch_idx, return_format=None):
        batch = self.detach(batch)
        texture = batch[DIK.TEXTURE_PERSON][0].clone()
        return self.run(texture, return_format)

    def visualize_interior_sampled(self, batch, batch_idx, return_format=None):
        batch = self.detach(batch)
        texture = batch[DIK.FACE_FEATUREIMAGE][0].clone()
        return self.run(texture, return_format)

    def visualize_exterior_sampled(self, batch, batch_id, return_format=None):
        batch = self.detach(batch)
        texture = batch[DIK.ADDITIVE_FEATUREIMAGE][0].clone()
        return self.run(texture, return_format)

    def visualize_enhanced(self, batch, batch_id, return_format=None):
        batch = self.detach(batch)
        texture = batch[DIK.ADDITIVE_FEATUREIMAGE][0].clone()
        return self.run(texture, return_format)


class CompleteVisualizer(Visualizer):

    def __init__(self, *args, bfm_uv_factory, **kwargs):
        super().__init__(*args, **kwargs)
        self.bfm_uv_factory = bfm_uv_factory

    def run(self, pipeline, batch, batch_idx, return_format=None, forward_type="style2image"):
        if forward_type == "interior2image":
            batch = pipeline.forward_interior2image(batch, batch_idx)
        elif forward_type == "style2image":
            batch = pipeline.forward_latent2image(batch, batch_idx)
        elif forward_type == "texture2image":
            batch = pipeline.forward_texture2image(batch, batch_idx)
        else:
            raise Warning("invalid forward type")
        return batch

    def visualize_single(self, pipeline, batch, batch_idx, theta_all=None, R=None, forward_type='style2image',
                         correct_translation=True):
        batch2 = batch
        if R is None:
            R = theta2rotation_matrix(theta_all=theta_all).to(self.device)
        batch2[DIK.R] = R.unsqueeze(0)
        uv = self.bfm_uv_factory.getUV(sp=batch[DIK.COEFF_SHAPE], ep=batch[DIK.COEFF_EXPRESSION], R=R, t=batch[DIK.T],
                                       correct_translation=correct_translation)
        batch2[DIK.UV_RENDERED] = uv.expand(batch2[DIK.COEFF_SHAPE].shape[0], -1, -1, -1).to(self.device)
        batch2 = self.run(pipeline, batch2, batch_idx, return_format='pil', forward_type=forward_type)
        return batch2
