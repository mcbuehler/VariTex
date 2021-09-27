from enum import Enum


class DataItemKey(Enum):
    IMAGE_IN = 1
    IMAGE_ENCODED = 2
    STYLE_LATENT = 3
    TEXTURE_PRIOR = 4
    TEXTURE_PERSON = 5
    FACE_FEATUREIMAGE = 6
    ADDITIVE_FEATUREIMAGE = 7
    IMAGE_OUT = 8
    UV_RENDERED = 9
    MASK_UV = 10
    STYLE_LATENT_MU = 12
    STYLE_LATENT_STD = 13
    FILENAME = 14
    COEFF_SHAPE = 15
    COEFF_EXPRESSION = 16
    SEGMENTATION_MASK = 17
    SEGMENTATION_PREDICTED = 18
    FULL_FEATUREIMAGE = 19
    MASK_FULL = 21
    IMAGE_IN_ENCODE = 26
    LATENT_INTERIOR = 27
    LATENT_EXTERIOR = 28
    R = 90
    T = 91
    SCALE = 92

    def __str__(self):
        return str(self._name_).lower()
