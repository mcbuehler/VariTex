import torch
import numpy as np
import PIL.Image


def to_np(t):
    if isinstance(t, torch.Tensor):
        t = t.detach().cpu().numpy()
    if isinstance(t, PIL.Image.Image):
        t = np.array(t)
    return t