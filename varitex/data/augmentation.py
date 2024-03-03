"""
Random affine transforms used during training.
"""
import PIL.Image
import numpy
import torchvision.transforms.functional as F
from torchvision.transforms import RandomAffine


class CustomRandomAffine(RandomAffine):
    def __init__(self, img_size, flip_p=0, *args, **kwargs):
        if not "degrees" in kwargs:
            kwargs["degrees"] = 0
        super().__init__(*args, **kwargs)
        self.ret = self.get_params(self.degrees, self.translate, self.scale, self.shear, img_size)
        self.flip = numpy.random.rand(1) < flip_p

    def to_pil(self, img):
        return PIL.Image.fromarray(img)

    def __call__(self, img):
        """
            img (PIL Image or Tensor): Image to be transformed.

        Returns:
            PIL Image or Tensor: Affine transformed image.
        """
        is_uv = img.shape[-1] == 2
        if not isinstance(img, PIL.Image.Image):
            if is_uv:
                # We convert the UV to a PIL Image such that we can apply the affine transform.
                img = numpy.concatenate([img, numpy.zeros((*img.shape[:2], 1))], -1)
                img = (img + 1) * 127.5
                img = img.astype(numpy.uint8)
            img = self.to_pil(img)

        if self.flip:
            img = F.hflip(img)

        transformed = F.affine(img, *self.ret,
                               fill=self.fill)
        result = numpy.array(transformed)
        if is_uv:
            # Convert back to UV range.
            result = result[:, :, :2]
            result = result / 127.5 - 1
        return result
