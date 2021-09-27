import os

import cv2
import numpy as np
import torch
from mutil.pytorch_utils import ImageNetNormalizeTransformForward
from scipy import ndimage
from skimage.morphology import convex_hull_image
from torchvision import transforms

from varitex.data.augmentation import CustomRandomAffine
from varitex.data.keys_enum import DataItemKey as DIK
from varitex.data.custom_dataset import CustomDataset


class NPYDataset(CustomDataset):
    # Rotation, translation, scale, shape, expression, segmentation, uv
    keys = ["R", "t", "s", "sp", "ep", "segmentation", "uv"]

    def __init__(self, opt, *args, **kwargs):
        super().__init__(opt, *args, **kwargs)
        # We need to call this to set self.__len()__
        # h5py can only be read from multiple workers if the file is opened in sub-processes, not threads.
        # We need to open it later again for the __getitem()__
        self.image_folder = self.opt.image_folder

        self.dataroot_npy = opt.dataroot_npy
        self.data = {
            key: np.load(os.path.join(self.dataroot_npy, "{}.npy".format(key)), 'r') for key in self.keys
        }

        # These are strings
        self.data["filename"] = np.load(os.path.join(self.dataroot_npy, "filename.npy"), allow_pickle=True)
        self.data["dataset_splits"] = np.load(os.path.join(self.dataroot_npy, "dataset_splits.npz"), allow_pickle=True)

        self.set_indices()

        self.N = len(self.indices)

        assert self.opt.image_h == self.opt.image_w, "Only square images supported!"

    def set_indices(self, indices=None):
        # We can provide indices or load them from h5
        if indices is None:
            if self.split == "all":
                self.indices = list(range(self.data["filename"].shape[0]))
            elif self.split in self.data["dataset_splits"].keys():
                self.indices = self.data["dataset_splits"][self.split][:]
            else:
                raise Warning("Invalid split: {}".format(self.split))
        else:
            self.indices = indices

    @classmethod
    def _center_crop(cls, image, square_dim):
        if square_dim is None:
            square_dim = min(image.shape[:2])
        new_height, new_width = square_dim, square_dim

        # width, height = image.size   # Get dimensions
        height, width = image.shape[:2]
        center_h = height // 2
        center_w = width // 2

        left = center_w - new_width // 2
        top = center_h - new_height // 2
        right = left + new_width
        bottom = top + new_height

        # Crop the center of the image
        image = image[top:bottom, left:right]
        return image

    @classmethod
    def _apply_transforms(cls, ndarray_in, height, width, interpolation_nearest=False, affine_transform=None,
                          center_crop_dim=None):
        # Semantic masks should use 'nearest' interpolation.
        interpolation = cv2.INTER_NEAREST if interpolation_nearest else cv2.INTER_LINEAR

        if affine_transform is not None:
            ndarray = affine_transform(ndarray_in)
        else:
            ndarray = ndarray_in

        ndarray = cls._center_crop(ndarray, square_dim=center_crop_dim)

        if ndarray.shape[0] != height or ndarray.shape[1] != width:
            ndarray = cv2.resize(ndarray, (width, height), interpolation)
        return ndarray

    @classmethod
    def preprocess_image(cls, img, height, width, affine_transform=None, center_crop_dim=None):
        img = cls._apply_transforms(img, height, width, affine_transform=affine_transform,
                                    center_crop_dim=center_crop_dim)

        all_transforms = [
            transforms.ToTensor(),  # Converts image to range [0, 1]
            ImageNetNormalizeTransformForward()  # Uses ImageNet means and std
        ]

        img_tensor = transforms.Compose(all_transforms)(img)
        return img_tensor  # img_tensor can be < 1 and > 1, but is roughly centered at 0

    def preprocess_segmentation(self, segmentation, height, width, affine_transform=None, mask=None,
                                center_crop_dim=None):
        segmentation_np = np.zeros(segmentation.shape)
        for region_idx in self.opt.semantic_regions:
            segmentation_np[segmentation == region_idx] = 1
        # We now have a binary mask with 1s for all regions that we specified

        segmentation = self._apply_transforms(segmentation_np, height, width, interpolation_nearest=True,
                                              affine_transform=affine_transform, center_crop_dim=center_crop_dim)

        segmentation_tensor = torch.from_numpy(segmentation)
        segmentation = segmentation_tensor.unsqueeze(0)

        if mask is not None:
            segmentation[mask] = 0

        return segmentation.float()

    @classmethod
    def preprocess_uv(cls, uv, height, width, affine_transform=None, center_crop_dim=None):
        uv = cls._apply_transforms(uv, height, width, interpolation_nearest=True, affine_transform=affine_transform,
                                   center_crop_dim=center_crop_dim)
        # same as ToTensor, but it's better to keep this explicit
        uv_tensor = torch.from_numpy(uv)

        if not (-1 <= uv_tensor.min() and uv_tensor.max() <= 1):
            raise ValueError("UV not in range [-1, 1]! min: {} max: {}".format(uv.min(), uv.max()))

        return uv_tensor.float()

    @classmethod
    def preprocess_mask(cls, uv, height, width, affine_transform=None):
        # We compute the mask from the UV (invalid values will be marked as -1)
        uv = cls.preprocess_uv(uv, height, width, affine_transform)
        mask_tensor = torch.logical_and((uv[:, :, 0] != -1), (uv[:, :, 1] != -1))
        return mask_tensor.unsqueeze(0)

    @staticmethod
    def convex_hull_mask_tensor(uv_mask, segmentations):
        mask = uv_mask
        for semantic_mask in segmentations:
            mask = mask.logical_or(semantic_mask.bool())

        mask_np = mask.squeeze().detach().cpu().numpy()
        mask_image = convex_hull_image(mask_np)
        mask_image_tensor = torch.from_numpy(mask_image).unsqueeze(0)
        return mask_image_tensor

    def full_mask(self, uv_mask, segmentation_mask):
        full_mask = uv_mask.logical_or(segmentation_mask)

        mask_image = ndimage.binary_fill_holes(full_mask.squeeze().numpy())
        mask_image_tensor = torch.from_numpy(mask_image).unsqueeze(0)
        return mask_image_tensor

    def preprocess_expressions(self, frame_id):
        return torch.tensor(self.data['ep'][frame_id])

    def _read_image(self, filename, size):
        path_image = os.path.join(self.image_folder, "{}.png".format(filename))
        if not os.path.exists(path_image):
            raise FileNotFoundError("This image has not been found: '{}'".format(path_image))
        image_bgr = cv2.imread(path_image)
        image = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
        image = cv2.resize(image, size, interpolation=cv2.INTER_LINEAR)
        return image

    def __getitem__(self, index):
        frame_id = self.indices[index]  # We keep the dataset splits loaded as indices
        filename = self.data["filename"][frame_id]
        height, width = self.opt.image_h, self.opt.image_w

        if self.augmentation:
            # Create a random affine transform for each iteration, but use the same transform for all in- and outputs
            affine_transform = CustomRandomAffine([self.initial_width, self.initial_height], **self.transform_params)
        else:
            affine_transform = None
        try:
            image_raw = self._read_image(filename, (self.initial_width, self.initial_height))
            uv = self.data["uv"][frame_id].astype(np.float32)  # opencv expects float 32
            segmentation = self.data["segmentation"][frame_id]

            img_tensor_clean = self.preprocess_image(image_raw.copy(), height, width)
            img_tensor = self.preprocess_image(image_raw.copy(), height, width, affine_transform)
            uv_tensor = self.preprocess_uv(uv.copy(), height, width, affine_transform)
            mask_tensor = self.preprocess_mask(uv.copy(), height, width,
                                               affine_transform)  # We compute the mask from the uv

            segmentation_tensor = self.preprocess_segmentation(segmentation.copy(), height, width, affine_transform)

            mask_full_tensor = self.full_mask(mask_tensor, segmentation_tensor)

            if not self.opt.keep_background:
                # We remove the background to focus the network capacity on the relevant regions
                if self.opt.bg_color == 'black':
                    img_tensor[~mask_full_tensor.expand_as(img_tensor)] = img_tensor.min()
                else:
                    img_tensor[~mask_full_tensor.expand_as(img_tensor)] = 0.0

                # We need to compute the full mask on the non augmented variant such that we can mask the encoding image
                mask_tensor_clean = self.preprocess_mask(uv.copy(), height, width)  # We compute the mask from the uv
                segmentation_tensor_clean = self.preprocess_segmentation(segmentation.copy(),
                                                                         height, width)
                mask_full_tensor_clean = self.full_mask(mask_tensor_clean, segmentation_tensor_clean)
                if self.opt.bg_color == 'black':
                    img_tensor_clean[~mask_full_tensor_clean.expand_as(img_tensor_clean)] = img_tensor_clean.min()
                else:
                    img_tensor_clean[~mask_full_tensor_clean.expand_as(img_tensor_clean)] = 0
            #
            return {
                DIK.IMAGE_IN_ENCODE: img_tensor_clean,
                DIK.IMAGE_IN: img_tensor,
                DIK.SEGMENTATION_MASK: segmentation_tensor,
                DIK.UV_RENDERED: uv_tensor,
                DIK.MASK_UV: mask_tensor,
                DIK.MASK_FULL: mask_full_tensor,  # Used to mask input before encoding
                DIK.FILENAME: filename,
                DIK.COEFF_SHAPE: self.data["sp"][frame_id].copy().astype(np.float32),
                DIK.COEFF_EXPRESSION: self.data["ep"][frame_id].copy().astype(np.float32),
                DIK.R: torch.from_numpy(self.data["R"][frame_id].copy()).float(),
                DIK.T: torch.from_numpy(self.data["t"][frame_id].copy()).float(),
                DIK.SCALE: torch.from_numpy(self.data["s"][frame_id].copy()).float()
            }
        except ValueError as e:
            print("Value error when processing index {}".format(index))
            print(e)
            exit(0)

    def __len__(self):
        return self.N

    def get_unsqueezed(self, index, device='cuda'):
        batch = self[index]
        batch_unsqueeze = {}
        for k, v in batch.items():
            if hasattr(v, "unsqueeze"):
                batch_unsqueeze[k] = v.unsqueeze(0).to(device)
            elif k == DIK.FILENAME:
                batch_unsqueeze[k] = [v]
            else:
                batch_unsqueeze[k] = torch.Tensor(v).unsqueeze(0).to(device)

        return batch_unsqueeze

    def get_raw_image(self, index):
        frame_id = self.indices[index]  # We keep the dataset splits loaded as indices
        filename = self.data["filename"][frame_id]
        image_raw = self._read_image(filename, (self.initial_width, self.initial_height))

        height, width = self.opt.image_h, self.opt.image_w
        return self._apply_transforms(image_raw, height, width)
