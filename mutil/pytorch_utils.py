import torch
import numpy as np

from mutil.threed_utils import eulerAnglesToRotationMatrix
from torchvision.transforms import transforms


def to_tensor(a, device='cpu', dtype=None):
    if isinstance(a, torch.Tensor):
        return a.to(device)
    return torch.tensor(np.array(a, dtype=dtype)).to(device)


def tensor2np(t):
    if isinstance(t, torch.Tensor):
        return t.detach().clone().cpu().numpy()
    return t


def get_device():
    # Set the device
    if torch.cuda.is_available():
        device = torch.device("cuda:0")
    else:
        device = torch.device("cpu")
        print("WARNING: CPU only, this will be slow!")
    return device


# Normalize images with the imagenet means and stds
class ImageNetNormalizeTransform(transforms.Normalize):
    """
    Test case:
        img_tensor = transforms.Compose([transforms.ToTensor()])(img)
        describe_tensor(img_tensor, title="img tensor to tensor")

        img_tensor = transforms.Compose([
            transforms.ToTensor(),  # Converts image to range [0, 1]
            ImageNetNormalizeTransformForward()  # Uses ImageNet means and std
        ])(img)
        describe_tensor(img_tensor, title="img tensor")

        img_tensor = transforms.Compose([ImageNetNormalizeTransformInverse()])(img_tensor)
        describe_tensor(img_tensor, title="img")
    """
    NORMALIZE_MEAN = (0.485, 0.456, 0.406)
    NORMALIZE_STD = (0.229, 0.224, 0.225)


class ImageNetNormalizeTransformForward(ImageNetNormalizeTransform):
    """
    Expects inputs in range [0, 1] (e.g. from transforms.ToTensor)
    """
    def __init__(self):
        super().__init__(mean=self.NORMALIZE_MEAN, std=self.NORMALIZE_STD)


class ImageNetNormalizeTransformInverse(ImageNetNormalizeTransform):
    """
    Expects inputs as after using ImageNetNormalizeTransformForward.
    If this is the case, the inputs are returned in the range [0, 1] * scale

    Can handle tensors with and without batch dimension
    """
    def __init__(self, scale=1.0):
        mean = torch.as_tensor(self.NORMALIZE_MEAN)
        std = torch.as_tensor(self.NORMALIZE_STD)
        std_inv = 1 / (std + 1e-7)
        mean_inv = -mean * std_inv
        super().__init__(mean=mean_inv, std=std_inv)

        self.scale = scale  # Can also be 255

    def __call__(self, tensor):
        if len(tensor.shape) == 3:  # Image
            return super().__call__(tensor.clone()) * self.scale
        elif len(tensor.shape) == 4:  # With batch dimensions
            results = [self(t) for t in tensor]
            return torch.stack(results)


def theta2rotation_matrix(theta_x=0, theta_y=0, theta_z=0, theta_all=None):
    # Angles should be in degrees
    if theta_all is not None:
        theta = [np.deg2rad(t) for t in theta_all]
    else:
        theta = [np.deg2rad(theta_x), np.deg2rad(theta_y),
                 np.deg2rad(-theta_z)]  # x looking from bottom, y looking to right, z tilting to left
    R = eulerAnglesToRotationMatrix(theta)
    R = torch.Tensor(R).float()
    return R
