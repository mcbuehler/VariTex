import torch
from pl_bolts.models.autoencoders.components import (
    DecoderBlock, Interpolate, resize_conv1x1, resize_conv3x3
)
from torch import nn

from varitex.data.keys_enum import DataItemKey as DIK
from varitex.modules.custom_module import CustomModule


class Decoder(CustomModule):
    """
    Decoder for the neural face texture.
    """

    def __init__(self, opt):
        super().__init__(opt)

        latent_dim_face = opt.latent_dim // 2

        input_height = opt.texture_dim
        # Resnet-18 variant
        self.decoder = SimpleResNetDecoder(DecoderBlock, [2, 2, 2, 2], latent_dim_face, input_height,
                                           nc_texture=opt.texture_nc)

    def forward(self, batch, batch_idx):
        style = batch[DIK.LATENT_INTERIOR]
        texture = self.decoder(style)
        batch[DIK.TEXTURE_PERSON] = texture
        return batch


class AdditiveDecoder(CustomModule):
    """
    Additive decoder. Produces the additive feature image.
    """

    def __init__(self, opt):
        super().__init__(opt)
        latent_dim_additive = opt.latent_dim // 2

        input_height = opt.texture_dim
        out_dim = opt.texture_nc
        condition_dim = opt.texture_nc  # We condition on a neural texture for the face interior
        # Resnet-18 variant
        self.decoder = EarlyConditionedSimpleRestNetDecocer(ConditionedDecoderBlock, [2, 2, 2, 2], latent_dim_additive,
                                                            input_height, nc_texture=out_dim,
                                                            condition_dim=condition_dim)

    def forward(self, batch, batch_idx):
        latent = batch[DIK.LATENT_EXTERIOR]

        # The sampled texture should already have been masked to the face interior
        # Errors should not back-propagate through the sampled face interior texture
        condition = batch[DIK.FACE_FEATUREIMAGE].detach()
        additive_featureimage = self.decoder(latent, condition)

        batch[DIK.ADDITIVE_FEATUREIMAGE] = additive_featureimage  # This also has features for the interior
        return batch


class SimpleResNetDecoder(nn.Module):
    """
    Resnet in reverse order.
    Most code from pl_bolts.models.autoencoders.components.
    """

    def __init__(self, block, layers, latent_dim, input_height, nc_texture, first_conv=False, maxpool1=False):
        super().__init__()

        self.expansion = block.expansion
        self.inplanes = 512 * block.expansion
        self.first_conv = first_conv
        self.maxpool1 = maxpool1
        self.input_height = input_height

        self.upscale_factor = 8

        self.linear = nn.Linear(latent_dim, self.inplanes * 4 * 4)

        self.initial = self._make_layer(block, 256, layers[0], scale=2)

        self.layer0 = self._make_layer(block, 256, layers[0], scale=2)
        self.layer1 = self._make_layer(block, 256, layers[0], scale=2)
        self.layer2 = self._make_layer(block, 128, layers[1], scale=2)
        self.layer3 = self._make_layer(block, 64, layers[2], scale=2)

        if self.input_height == 128:
            self.layer4 = self._make_layer(block, 64, layers[3])
        elif self.input_height == 256:
            self.layer4 = self._make_layer(block, 64, layers[3], scale=2)
        else:
            raise Warning("Invalid input height: '{}".format(self.input_height))

        if self.first_conv:
            self.upscale = Interpolate(scale_factor=2)
            self.upscale_factor *= 2
        else:
            self.upscale = Interpolate(scale_factor=1)

        # interpolate after linear layer using scale factor
        self.upscale1 = Interpolate(size=input_height // self.upscale_factor)

        self.conv1 = nn.Conv2d(
            64 * block.expansion, nc_texture, kernel_size=3, stride=1, padding=1, bias=False
        )

    def _make_layer(self, block, planes, blocks, scale=1):
        """

        Args:
            block:
            planes: int number of channels
            blocks: int number of blocks (e.g. 2)
            scale:

        Returns:

        """
        upsample = None
        if scale != 1 or self.inplanes != planes * block.expansion:
            upsample = nn.Sequential(
                resize_conv1x1(self.inplanes, planes * block.expansion, scale),
                nn.BatchNorm2d(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, scale, upsample))
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.linear(x)

        x = x.view(x.size(0), 512 * self.expansion, 4, 4)
        x = self.initial(x)

        x = self.layer0(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.conv1(x)
        return x


class EarlyConditionedSimpleRestNetDecocer(nn.Module):
    """
    Modified from pl_bolts.models.autoencoders.components.
    """

    def __init__(self, block, layers, latent_dim, input_height, nc_texture, first_conv=False, maxpool1=False,
                 condition_dim=16, **kwargs):
        # TODO: refactor duplicated code
        super().__init__()

        self.condition_dim = condition_dim

        self.expansion = block.expansion
        self.inplanes = 512 * block.expansion
        self.first_conv = first_conv
        self.maxpool1 = maxpool1
        self.input_height = input_height

        self.linear = nn.Linear(latent_dim, self.inplanes)

        self.initial = self._make_layer(block, 512, layers[0], scale=2, condition_dim=self.condition_dim)

        self.layer0 = self._make_layer(block, 256, layers[0], scale=2, condition_dim=self.condition_dim)
        self.layer1 = self._make_layer(block, 256, layers[0], scale=2, condition_dim=self.condition_dim)
        self.layer2 = self._make_layer(block, 128, layers[1], scale=2, condition_dim=self.condition_dim)
        self.layer3 = self._make_layer(block, 64, layers[2], scale=2, condition_dim=self.condition_dim)

        if self.input_height == 128:
            self.layer4 = self._make_layer(block, 64, layers[3], condition_dim=self.condition_dim)
        elif self.input_height == 256:
            self.layer4 = self._make_layer(block, 64, layers[3], scale=2, condition_dim=self.condition_dim)
        else:
            raise Warning("Invalid input height: '{}".format(self.input_height))

        if self.first_conv:
            self.upscale = Interpolate(scale_factor=2)
            self.upscale_factor *= 2

        self.conv1 = nn.Conv2d(
            64 * block.expansion, nc_texture, kernel_size=3, stride=1, padding=1, bias=False
        )

    def _make_layer(self, block, planes, blocks, scale=1, condition_dim=16):
        """

        Args:
            block:
            planes: int number of channels
            blocks: int number of blocks (e.g. 2)
            scale:

        Returns:

        """
        upsample = None
        if scale != 1 or self.inplanes != planes * block.expansion:
            upsample = nn.Sequential(
                resize_conv1x1(self.inplanes, planes * block.expansion, scale),
                nn.BatchNorm2d(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, scale, upsample, condition_dim=condition_dim))
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes, condition_dim=condition_dim))

        return nn.Sequential(*layers)

    def forward(self, x, condition):
        x = self.linear(x)
        # We now have 512 feature maps with the same value in all spatial locations
        # self.inplanes changes when creating blocks
        x = x.view(x.shape[0], 512, 1, 1).expand(-1, -1, 4, 4)

        in_dict = {"features": x, "condition": condition}
        out_dict = self.initial(in_dict)

        out_dict = self.layer0(out_dict)
        out_dict = self.layer1(out_dict)
        out_dict = self.layer2(out_dict)
        out_dict = self.layer3(out_dict)
        out_dict = self.layer4(out_dict)
        x = out_dict['features']
        x = self.conv1(x)
        return x


class ConditionedDecoderBlock(nn.Module):
    """
    ResNet block, but convs replaced with resize convs, and channel increase is in
    second conv, not first.
    Also heavily borrowed from pl_bolts.models.autoencoders.components.
    """

    expansion = 1

    def __init__(self, inplanes, planes, scale=1, upsample=None, condition_dim=16):
        super().__init__()
        self.conv1 = resize_conv3x3(inplanes + condition_dim,
                                    inplanes)  # 2 is the feature dimension for the conditioning
        self.bn1 = nn.BatchNorm2d(inplanes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = resize_conv3x3(inplanes, planes, scale)
        self.bn2 = nn.BatchNorm2d(planes)
        self.upsample = upsample

        self.interpolation_mode = "bilinear"

    def forward(self, data):
        x = data["features"]
        condition = data["condition"]

        condition_scaled = torch.nn.functional.interpolate(condition, x.shape[-2:], mode=self.interpolation_mode)
        identity = x

        out = torch.cat([x, condition_scaled], 1)  # Along the channel dimension
        out = self.conv1(out)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.upsample is not None:
            identity = self.upsample(x)

        out += identity
        out = self.relu(out)

        out_dict = {"features": out, "condition": condition}
        return out_dict
