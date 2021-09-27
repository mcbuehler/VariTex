import os

import numpy as np
import torch
from mutil.files import mkdir
from mutil.pytorch_utils import theta2rotation_matrix
from torch.utils.data import DataLoader
from tqdm import tqdm

from varitex.data.keys_enum import DataItemKey as DIK
from varitex.data.npy_dataset import NPYDataset
from varitex.data.uv_factory import BFMUVFactory
from varitex.modules.pipeline import PipelineModule
from varitex.visualization.batch import Visualizer


def get_model(opt):
    model = PipelineModule.load_from_checkpoint(opt.checkpoint, opt=opt, strict=False)
    model = model.eval()
    model = model.cuda()
    return model


def inference_ffhq(opt, results_folder, n=3000):
    """
    Runs inference on FFHQ. Save the resulting images in the results_folder.
    Also saves the latent codes and distributions.
    """
    print("Running inference on FFHQ. Using the extracted face model parameters and poses, and predicted latent codes.")
    mkdir(results_folder)
    visualizer = Visualizer(opt, return_format='pil')

    dataset = NPYDataset(opt, augmentation=False, split='val')
    dataloader = iter(DataLoader(dataset, batch_size=1, num_workers=0, shuffle=False))
    model = get_model(opt)
    visualizer.mask_key = DIK.SEGMENTATION_PREDICTED

    result = list()

    for i, batch in tqdm(enumerate(dataloader)):
        batch = model.forward(batch, i, std_multiplier=0)
        file_id = batch[DIK.FILENAME][0]
        latent_code = batch[DIK.STYLE_LATENT_MU].detach().cpu().numpy()[0]
        latent_std = batch[DIK.STYLE_LATENT_STD].detach().cpu().numpy()[0]
        result.append([latent_code, latent_std])

        out = visualizer.tensor2image(batch[DIK.IMAGE_OUT][0], batch=batch)
        out = visualizer.mask(out, batch, white_bg=False)

        out = visualizer.format_output(out, return_format='pil')
        out.save(os.path.join(results_folder, "{}.png".format(file_id)))

        if i >= n:
            break

    result = np.array(result)
    np.save(os.path.join(results_folder, "latents.npy"), result)
    print("Done")


def inference_posed(opt, results_folder, path_latent, n=3000):
    """
    Runs different poses for each latent code extracted from the FFHQ dataset.
    Uses random shape and expression parameters.
    """

    print("Running inference on FFHQ. Using random face model parameters, various poses, and predicted latent codes. This can be slow.")
    visualizer = Visualizer(opt, return_format='pil')
    visualizer.mask_key = DIK.SEGMENTATION_PREDICTED

    # We use the latent codes extracted from FFHQ, but they could also be sampled from the extracted distributions.
    latents = np.load(path_latent)[:, 0]  # We want to use the distribution means directly (index 0)

    mkdir(results_folder)

    model = get_model(opt)

    all_pose_axis = [[0], [1], [0, 1]]  # List of all axis: 0 is pitch, 1 is yaw

    uv_factory_bfm = BFMUVFactory(opt, use_bfm_gpu=True)

    s = torch.Tensor([0.1])
    t = torch.Tensor([[0, -2, -57]])
    results = list()
    all_poses = np.arange(-45, 46, 15)

    with torch.no_grad():
        for i in tqdm(range(n)):
            sp = uv_factory_bfm.bfm.sample_shape(1)
            ep = uv_factory_bfm.bfm.sample_expression(1)
            latent = torch.Tensor([latents[i]])

            for pose_axis in all_pose_axis:
                for pose in all_poses:
                    theta = torch.Tensor([0, 0, 0])
                    theta[pose_axis] = pose
                    R = theta2rotation_matrix(theta_all=theta).unsqueeze(0)
                    uv_tensor = uv_factory_bfm.getUV(R, t, s, sp, ep, correct_translation=True)

                    batch = {
                        DIK.UV_RENDERED: uv_tensor,
                        DIK.R: R,
                        DIK.T: t,
                        DIK.SCALE: s,
                        DIK.STYLE_LATENT: latent
                    }
                    batch = {k: v.cuda() for k, v in batch.items()}
                    batch_out = model.forward_latent2image(batch, 0)
                    out = visualizer.tensor2image(batch_out[DIK.IMAGE_OUT][0], batch=batch_out, white_bg=False)
                    results.append(out)
                    out = visualizer.format_output(out, return_format='pil')

                    folder_out = os.path.join(results_folder, "axis{}_theta{}".format(pose_axis, pose))
                    mkdir(folder_out)
                    out.save(os.path.join(results_folder, folder_out, "{:05d}.png".format(i)))


def inference_posed_ffhq(opt, results_folder, n=3000):
    """
    Runs different poses for each latent code extracted from the FFHQ dataset.
    Uses shape and expression parameters from validation set.
    """
    print("Running inference on FFHQ. Using the extracted face model parameters, various poses, and predicted latent codes. This can be slow.")
    mkdir(results_folder)
    visualizer = Visualizer(opt, return_format='pil')

    dataset = NPYDataset(opt, augmentation=False, split='val')
    dataloader = iter(DataLoader(dataset, batch_size=1, num_workers=0, shuffle=False))
    model = get_model(opt)
    visualizer.mask_key = DIK.SEGMENTATION_PREDICTED

    n = min(n, 3000)
    all_pose_axis = [[0], [1], [0, 1]]

    uv_factory_bfm = BFMUVFactory(opt, use_bfm_gpu=True)
    s = torch.Tensor([0.1])
    t = torch.Tensor([[0, -2, -57]])

    all_poses = np.arange(-45, 46, 15)

    for i, batch_orig in tqdm(enumerate(dataloader)):
        batch_orig = model.forward(batch_orig, i, std_multiplier=0)
        file_id = batch_orig[DIK.FILENAME][0]
        sp = batch_orig[DIK.COEFF_SHAPE].clone()
        ep = batch_orig[DIK.COEFF_EXPRESSION].clone()
        latent = batch_orig[DIK.STYLE_LATENT].clone()

        for pose_axis in all_pose_axis:
            for pose in all_poses:
                theta = torch.Tensor([0, 0, 0])
                theta[pose_axis] = pose
                R = theta2rotation_matrix(theta_all=theta).unsqueeze(0)
                uv_tensor = uv_factory_bfm.getUV(R, t, s, sp, ep, correct_translation=True)

                batch = {
                    DIK.UV_RENDERED: uv_tensor,
                    DIK.R: R,
                    DIK.T: t,
                    DIK.SCALE: s,
                    DIK.STYLE_LATENT: latent
                }
                batch = {k: v.cuda() for k, v in batch.items()}
                batch_out = model.forward_latent2image(batch, 0)
                out = visualizer.tensor2image(batch_out[DIK.IMAGE_OUT][0], batch=batch_out, white_bg=False)
                out = visualizer.format_output(out, return_format='pil')

                folder_out = os.path.join(results_folder, "axis{}_theta{}".format(pose_axis, pose))
                mkdir(folder_out)
                out.save(os.path.join(results_folder, folder_out, "{}.png".format(file_id)))

        if i >= n:
            break

    print("Done")
