import torch
import torchvision.models
from mutil.pytorch_utils import ImageNetNormalizeTransformInverse
from torch.nn import functional as F


def kl_divergence(mu, std):
    p = torch.distributions.Normal(torch.zeros_like(mu), torch.ones_like(std))
    q = torch.distributions.Normal(mu, std)
    kl = torch.distributions.kl.kl_divergence(p, q).mean(1)
    return kl


def reconstruction_loss(image_fake, image_real):
    # Simple L1 for now
    return torch.mean(torch.abs(image_fake - image_real))


def l2_loss(image_fake, image_real):
    return torch.mean((image_fake - image_real) ** 2)


class VGG16(torch.nn.Module):
    def __init__(self, path_weights):
        super().__init__()
        vgg16 = torchvision.models.vgg16(pretrained=False)
        import h5py
        with h5py.File(path_weights, 'r') as f:
            state_dict = self.get_keras_mapping(f)
            vgg16.load_state_dict(state_dict, strict=False)  # Ignore the missing keys for the classifier
        vgg_pretrained_features = vgg16.features

        # 'D': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512, 'M'],
        self.slice1 = torch.nn.Sequential()
        self.slice2 = torch.nn.Sequential()
        self.slice3 = torch.nn.Sequential()
        self.slice4 = torch.nn.Sequential()

        self.slice1.add_module(str(0), vgg_pretrained_features[0])
        for i in range(1, 3):
            self.slice2.add_module(str(i), vgg_pretrained_features[i])
        for i in range(3, 22):
            self.slice3.add_module(str(i), vgg_pretrained_features[i])
        for i in range(22, 31):
            self.slice4.add_module(str(i), vgg_pretrained_features[i])

    def get_keras_mapping(self, f):
        # VGG 16 config D
        state_dict = {
            "features.0.weight": f["conv1_1"]["conv1_1_1"]["kernel:0"],
            "features.0.bias": f["conv1_1"]["conv1_1_1"]["bias:0"],

            "features.2.weight": f["conv1_2"]["conv1_2_1"]["kernel:0"],
            "features.2.bias": f["conv1_2"]["conv1_2_1"]["bias:0"],

            "features.5.weight": f["conv2_1"]["conv2_1_1"]["kernel:0"],
            "features.5.bias": f["conv2_1"]["conv2_1_1"]["bias:0"],

            "features.7.weight": f["conv2_2"]["conv2_2_1"]["kernel:0"],
            "features.7.bias": f["conv2_2"]["conv2_2_1"]["bias:0"],

            "features.10.weight": f["conv3_1"]["conv3_1_1"]["kernel:0"],
            "features.10.bias": f["conv3_1"]["conv3_1_1"]["bias:0"],

            "features.12.weight": f["conv3_2"]["conv3_2_1"]["kernel:0"],
            "features.12.bias": f["conv3_2"]["conv3_2_1"]["bias:0"],

            "features.14.weight": f["conv3_3"]["conv3_3_1"]["kernel:0"],
            "features.14.bias": f["conv3_3"]["conv3_3_1"]["bias:0"],

            "features.17.weight": f["conv4_1"]["conv4_1_1"]["kernel:0"],
            "features.17.bias": f["conv4_1"]["conv4_1_1"]["bias:0"],

            "features.19.weight": f["conv4_2"]["conv4_2_1"]["kernel:0"],
            "features.19.bias": f["conv4_2"]["conv4_2_1"]["bias:0"],

            "features.21.weight": f["conv4_3"]["conv4_3_1"]["kernel:0"],
            "features.21.bias": f["conv4_3"]["conv4_3_1"]["bias:0"],

            "features.24.weight": f["conv5_1"]["conv5_1_1"]["kernel:0"],
            "features.24.bias": f["conv5_1"]["conv5_1_1"]["bias:0"],

            "features.26.weight": f["conv5_2"]["conv5_2_1"]["kernel:0"],
            "features.26.bias": f["conv5_2"]["conv5_2_1"]["bias:0"],

            "features.28.weight": f["conv5_3"]["conv5_3_1"]["kernel:0"],
            "features.28.bias": f["conv5_3"]["conv5_3_1"]["bias:0"],
        }
        #  keras: [3, 3, 3, 64])
        #  pytorch: torch.Size([64, 3, 3, 3]).
        # Keras stores weights in the order (kernel_size, kernel_size, input_dim, output_dim),
        # but pytorch expects (output_dim, input_dim, kernel_size, kernel_size)
        # We need to transpose them https://discuss.pytorch.org/t/how-to-convert-keras-model-to-pytorch-and-run-inference-in-c-correctly/93451/3
        state_dict = {k: torch.Tensor(v[:].transpose()) for k, v in state_dict.items()}
        return state_dict

    def forward(self, x):
        x = F.interpolate(x, size=224, mode="bilinear")
        h_conv1_1 = self.slice1(x)
        h_conv1_2 = self.slice2(h_conv1_1)
        h_conv3_2 = self.slice3(h_conv1_2)
        h_conv4_2 = self.slice4(h_conv3_2)
        out = [h_conv1_1, h_conv1_2, h_conv3_2, h_conv4_2]
        return out


class VGG19(torch.nn.Module):
    def __init__(self):
        super().__init__()
        vgg_pretrained_features = torchvision.models.vgg19(pretrained=True).features
        self.slice1 = torch.nn.Sequential()
        self.slice2 = torch.nn.Sequential()
        self.slice3 = torch.nn.Sequential()
        self.slice4 = torch.nn.Sequential()
        self.slice5 = torch.nn.Sequential()
        for x in range(2):
            self.slice1.add_module(str(x), vgg_pretrained_features[x])
        for x in range(2, 7):
            self.slice2.add_module(str(x), vgg_pretrained_features[x])
        for x in range(7, 12):
            self.slice3.add_module(str(x), vgg_pretrained_features[x])
        for x in range(12, 21):
            self.slice4.add_module(str(x), vgg_pretrained_features[x])
        for x in range(21, 30):
            self.slice5.add_module(str(x), vgg_pretrained_features[x])

    def forward(self, X):
        h_relu1 = self.slice1(X)
        h_relu2 = self.slice2(h_relu1)
        h_relu3 = self.slice3(h_relu2)
        h_relu4 = self.slice4(h_relu3)
        h_relu5 = self.slice5(h_relu4)
        out = [h_relu1, h_relu2, h_relu3, h_relu4, h_relu5]
        return out


# Perceptual loss that uses a pretrained VGG network
class ImageNetVGG19Loss(torch.nn.Module):
    def __init__(self):
        super(ImageNetVGG19Loss, self).__init__()
        self.vgg = VGG19()
        # if gpu_ids:
        #     self.vgg.cuda()
        self.criterion = torch.nn.L1Loss()
        self.weights = [1.0 / 32, 1.0 / 16, 1.0 / 8, 1.0 / 4, 1.0]

        for param in self.parameters():
            param.requires_grad = False

    def forward(self, fake, real):
        x_vgg, y_vgg = self.vgg(fake), self.vgg(real)
        loss = 0
        for i in range(len(x_vgg)):
            loss += self.weights[i] * self.criterion(x_vgg[i], y_vgg[i].detach())
            # loss = self.criterion(x_vgg[i], y_vgg[i].detach())
            # print(list(x_vgg[i].shape), loss.data, (self.weights[i] * loss).data, self.weights[i])
        return loss


# Perceptual loss that uses a pretrained VGG network
class FaceRecognitionVGG16Loss(torch.nn.Module):
    def __init__(self, path_weights):
        super().__init__()
        self.vgg = VGG16(path_weights)
        self.criterion = torch.nn.MSELoss()
        self.weights = [0.25, 0.25, 0.25, 0.25]

        self.unnormalize = ImageNetNormalizeTransformInverse()
        # Mean values of face images from the VGGFace paper
        self.normalize_values = torch.Tensor((93.5940, 104.7624, 129.1863)).unsqueeze(0).unsqueeze(2).unsqueeze(2)

        for param in self.parameters():
            param.requires_grad = False

    def preprocess(self, tensor):
        tensor = self.unnormalize(tensor)  # Is now in the range [0, 1] (if not under-/ overshooting)
        self.normalize_values = self.normalize_values.to(tensor.device)
        tensor = ((tensor * 255) - self.normalize_values.expand_as(tensor))
        # tensor = tensor - self.normalize_values
        # tensor =
        return tensor / 127.5  # Should we do this?

    def forward(self, fake, real):
        fake = self.preprocess(fake)
        real = self.preprocess(real)
        x_vgg, y_vgg = self.vgg(fake), self.vgg(real)

        loss = 0
        for i in range(len(x_vgg)):
            loss += self.weights[i] * self.criterion(x_vgg[i], y_vgg[i].detach())
            # loss = self.criterion(x_vgg[i], y_vgg[i].detach())
            # print(list(x_vgg[i].shape), loss.data, (self.weights[i] * loss).data, self.weights[i])
        return loss


"""
Code below:
Copyright (C) 2019 NVIDIA Corporation.  All rights reserved.
Licensed under the CC BY-NC-SA 4.0 license (https://creativecommons.org/licenses/by-nc-sa/4.0/legalcode).
"""


# Defines the GAN loss which uses either LSGAN or the regular GAN.
# When LSGAN is used, it is basically same as MSELoss,
# but it abstracts away the need to create the target label tensor
# that has the same size as the input
class GANLoss(torch.nn.Module):
    def __init__(self, gan_mode, target_real_label=1.0, target_fake_label=0.0,
                 tensor=torch.FloatTensor, opt=None):
        super(GANLoss, self).__init__()
        self.real_label = target_real_label
        self.fake_label = target_fake_label
        self.real_label_tensor = None
        self.fake_label_tensor = None
        self.zero_tensor = None
        self.Tensor = tensor
        self.gan_mode = gan_mode
        self.opt = opt
        if gan_mode == 'ls':
            pass
        elif gan_mode == 'original':
            pass
        elif gan_mode == 'w':
            pass
        elif gan_mode == 'hinge':
            pass
        else:
            raise ValueError('Unexpected gan_mode {}'.format(gan_mode))

    def get_target_tensor(self, input, target_is_real):
        if target_is_real:
            if self.real_label_tensor is None:
                self.real_label_tensor = self.Tensor(1).to(input.device).fill_(self.real_label)
                self.real_label_tensor.requires_grad_(False)
            return self.real_label_tensor.expand_as(input)
        else:
            if self.fake_label_tensor is None:
                self.fake_label_tensor = self.Tensor(1).to(input.device).fill_(self.fake_label)
                self.fake_label_tensor.requires_grad_(False)
            return self.fake_label_tensor.expand_as(input)

    def get_zero_tensor(self, input):
        if self.zero_tensor is None:
            self.zero_tensor = self.Tensor(1).fill_(0)
            self.zero_tensor.requires_grad_(False)
        return self.zero_tensor.expand_as(input).to(input.device)

    def loss(self, input, target_is_real, for_discriminator=True):
        if self.gan_mode == 'original':  # cross entropy loss
            target_tensor = self.get_target_tensor(input, target_is_real)
            loss = F.binary_cross_entropy_with_logits(input, target_tensor)
            return loss
        elif self.gan_mode == 'ls':
            target_tensor = self.get_target_tensor(input, target_is_real)
            return F.mse_loss(input, target_tensor)
        elif self.gan_mode == 'hinge':
            if for_discriminator:
                if target_is_real:
                    minval = torch.min(input - 1, self.get_zero_tensor(input))
                    loss = -torch.mean(minval)
                else:
                    minval = torch.min(-input - 1, self.get_zero_tensor(input))
                    loss = -torch.mean(minval)
            else:
                assert target_is_real, "The generator's hinge loss must be aiming for real"
                loss = -torch.mean(input)
            return loss
        else:
            # wgan
            if target_is_real:
                return -input.mean()
            else:
                return input.mean()

    def __call__(self, input, target_is_real, for_discriminator=True):
        # computing loss is a bit complicated because |input| may not be
        # a tensor, but list of tensors in case of multiscale discriminator
        if isinstance(input, list):
            loss = 0
            for pred_i in input:
                if isinstance(pred_i, list):
                    pred_i = pred_i[-1]
                loss_tensor = self.loss(pred_i, target_is_real, for_discriminator)
                bs = 1 if len(loss_tensor.size()) == 0 else loss_tensor.size(0)
                new_loss = torch.mean(loss_tensor.view(bs, -1), dim=1)
                loss += new_loss
            return loss / len(input)
        else:
            return self.loss(input, target_is_real, for_discriminator)
