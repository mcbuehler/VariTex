import numpy as np
from mutil.data_types import to_np


def adjust_width(img, target_width, is_uv=False):
    assert len(img.shape) == 3, "Are you using a batch dimension?"
    if is_uv:
        assert img.shape[-1] == 2
        n_channels = 2
        fill_value = -1.0
    else:
        assert img.shape[-1] == 3
        n_channels = 3
        fill_value = 255
    diff = target_width - img.shape[0]
    pad = np.ones((img.shape[0], diff // 2, n_channels)) * fill_value
    img = np.concatenate([pad, img, pad],
                         1)
    if is_uv:
        img = img.astype(np.float)
    else:
        img = img.astype(np.uint8)
    return img


def uv_to_color(rendered_uv, output_format="np"):
    uv_color = np.concatenate([rendered_uv, np.zeros((*rendered_uv.shape[:2], 1))], -1)
    if output_format == "pil":
        from PIL import Image
        uv_color = (uv_color + 1 ) * 127.5
        uv_color = Image.fromarray(uv_color.astype(np.uint8))
    return uv_color


def center_crop(image, new_height, new_width):
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


def interpolation(n, latent_from, latent_to, gaussian_correction=True):
    """
    Linear interpolate for a Gaussian RV
    :param n:
    :param latent_from: tensor or numpy array with batch dimension
    :param latent_to:
    :param type:
    :return:
    """
    latent_from, latent_to = to_np(latent_from), to_np(latent_to)
    steps = np.linspace(0, 1, n).astype(np.float32)

    for _ in range(len(latent_to.shape)):
        steps = np.expand_dims(steps, -1)
    if gaussian_correction:
        steps = steps / np.sqrt(steps ** 2 + (1 - steps) ** 2)  # Variance correction
    all_latents = (1 - steps) * latent_from + steps * latent_to
    return all_latents
