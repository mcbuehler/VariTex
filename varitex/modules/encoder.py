from pl_bolts.models.autoencoders.components import (
    resnet18_encoder
)

from varitex.data.keys_enum import DataItemKey as DIK
from varitex.modules.custom_module import CustomModule


class Encoder(CustomModule):
    def __init__(self, opt):
        super().__init__(opt)
        self.encoder = resnet18_encoder(False, False)

    def forward(self, batch, batch_idx):
        image = batch[DIK.IMAGE_IN_ENCODE]
        encoded = self.encoder(image)
        batch[DIK.IMAGE_ENCODED] = encoded
        return batch
