from torch.nn import Sigmoid

from varitex.data.keys_enum import DataItemKey as DIK
from varitex.modules.custom_module import CustomModule
from varitex.modules.unet import UNet


class Feature2ImageRenderer(CustomModule):
    def __init__(self, opt):
        super().__init__(opt)

        n_input_channels = opt.texture_nc * 2  # We have two feature images: face and additive
        self.unet = UNet(output_nc=4, input_channels=n_input_channels, features_start=opt.nc_feature2image,
                         num_layers=opt.feature2image_num_layers)
        self.probability = Sigmoid()

    def forward(self, batch, batch_idx):
        texture_enhanced = batch[DIK.FULL_FEATUREIMAGE]
        tensor_out = self.unet(texture_enhanced)
        mask_out = tensor_out[:, :1]  # First channel should be the foreground mask. Note that we keep the dimension
        image_out = tensor_out[:, 1:]  # RGB image

        # Should be close to 0 for the background
        mask_proba = self.probability(mask_out)
        batch[DIK.SEGMENTATION_PREDICTED] = mask_proba

        batch[DIK.IMAGE_OUT] = image_out
        return batch
