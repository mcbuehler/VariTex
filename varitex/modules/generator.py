import torch

from varitex.data.keys_enum import DataItemKey as DIK
from varitex.modules.decoder import Decoder, AdditiveDecoder
from varitex.modules.encoder import Encoder
from varitex.modules.feature2image import Feature2ImageRenderer
from varitex.modules.custom_module import CustomModule


class Generator(CustomModule):
    def __init__(self, opt):
        super().__init__(opt)

        self.MASK_VALUE = opt.uv_mask_value if hasattr(opt, "uv_mask_value") else 0

        enc_out_dim = 512
        latent_dim = opt.latent_dim

        self.fc_mu = torch.nn.Linear(enc_out_dim, latent_dim)
        self.fc_var = torch.nn.Linear(enc_out_dim, latent_dim)
        self.encoder = Encoder(opt)
        self.decoder = Decoder(opt)

        self.decoder_exterior = AdditiveDecoder(opt)
        self.texture2image = Feature2ImageRenderer(opt)

    def forward(self, batch, batch_idx, std_multiplier=1):
        batch = self.forward_encode(batch, batch_idx)  # Only encoding, not yet a distribution
        batch = self.forward_encoded2latent_distribution(batch)  # Compute mu and std
        batch = self.forward_sample_style(batch, batch_idx, std_multiplier=std_multiplier)  # Sample a latent code
        batch = self.forward_latent2image(batch, batch_idx)  # Decoders for face and exterior, followed by rendering
        return batch

    def forward_encode(self, batch, batch_idx):
        # Only encode the image, adds DIK.IMAGE_ENCODED.
        # This is not yet the latent distribution.
        batch = self.encoder(batch, batch_idx)
        return batch

    def forward_encoded2latent_distribution(self, batch):
        # Computes the latent distribution from the encoded image.
        mu, log_var = self.fc_mu(batch[DIK.IMAGE_ENCODED]), self.fc_var(batch[DIK.IMAGE_ENCODED])
        std = torch.exp(log_var / 2)

        batch[DIK.STYLE_LATENT_MU] = mu
        batch[DIK.STYLE_LATENT_STD] = std
        return batch

    def forward_sample_style(self, batch, batch_idx, std_multiplier=1):
        # Sample the latent code z from the given distribution.
        mu, std = batch[DIK.STYLE_LATENT_MU], batch[DIK.STYLE_LATENT_STD]

        q = torch.distributions.Normal(mu, std * std_multiplier)
        z = q.rsample()
        batch[DIK.STYLE_LATENT] = z
        return batch

    def forward_latent2image(self, batch, batch_idx):
        # Given a latent code, render an image.
        batch = self.forward_latent2featureimage(batch, batch_idx)
        # Neural rendering
        batch = self.texture2image(batch, batch_idx)
        # Note that we do not mask the output. The network should learn to produce 0 values for the background.
        return batch

    def forward_latent2featureimage(self, batch, batch_idx):
        """
        Given the full latent code DIK.STYLE_LATENT, process both interior and exterior region and generate
         both feature images: the face feature image and the additive feature image.
        """
        n = self.opt.latent_dim // 2
        z_interior = batch[DIK.STYLE_LATENT][:, :n]
        z_exterior = batch[DIK.STYLE_LATENT][:, n:]

        batch[DIK.LATENT_EXTERIOR] = z_exterior
        batch[DIK.LATENT_INTERIOR] = z_interior

        batch = self.forward_latent2texture_interior(batch, batch_idx)
        batch = self.forward_latent2additive_featureimage(batch, batch_idx)
        batch = self.forward_merge_textures(batch, batch_idx)
        return batch

    def forward_latent2texture_interior(self, batch, batch_idx):
        batch = self.forward_decoder_interior(batch, batch_idx)
        # Sampling texture using the UV map.
        batch = self.sample_texture(batch)
        return batch

    def forward_latent2additive_featureimage(self, batch, batch_idx):
        batch = self.decoder_exterior(batch, batch_idx)
        return batch

    def forward_merge_textures(self, batch, batch_idx):
        batch[DIK.FULL_FEATUREIMAGE] = torch.cat(
            [batch[DIK.FACE_FEATUREIMAGE], batch[DIK.ADDITIVE_FEATUREIMAGE]], 1)
        return batch

    def forward_decoder_interior(self, batch, batch_idx):
        batch = self.decoder(batch, batch_idx)
        return batch

    def sample_texture(self, batch):
        uv_texture = batch[DIK.TEXTURE_PERSON]
        uv_map = batch[DIK.UV_RENDERED]

        texture_sampled = torch.nn.functional.grid_sample(uv_texture, uv_map, mode='bilinear',
                                                          padding_mode='border', align_corners=False)

        # Grid sample yields the same value for not rendered region
        # We mask that region
        batch[DIK.MASK_UV] = torch.logical_or(batch[DIK.UV_RENDERED][:, :, :, 1] != -1,
                                              batch[DIK.UV_RENDERED][:, :, :, 0] != -1).unsqueeze(1)
        mask = batch[DIK.MASK_UV].expand_as(texture_sampled)
        texture_sampled[~mask] = self.MASK_VALUE

        batch[DIK.FACE_FEATUREIMAGE] = texture_sampled
        return batch
