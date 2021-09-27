from varitex.options.base_options import BaseOptions


class TrainOptions(BaseOptions):

    def initialize(self, parser):
        parser = super().initialize(parser)
        parser.add_argument('--lr', type=float, default=1e-3, help="Learning rate for generator.")
        # Loss term weights:
        parser.add_argument('--lambda_l2', type=float, default=1.0)
        parser.add_argument('--lambda_kl', type=float, default=0.1)
        parser.add_argument('--lambda_vgg', type=float, default=2.0)
        parser.add_argument('--lambda_segmentation', type=float, default=1.0)
        parser.add_argument('--lambda_rgb_texture', type=float, default=0.0)

        parser.add_argument('--gan_mode', type=str, default='ls')
        parser.add_argument('--lambda_gan', type=float, default=1)
        parser.add_argument('--lambda_discriminator_features', type=float, default=1)
        parser.add_argument('--nc_discriminator', type=int, default=64)
        parser.add_argument('--num_discriminator', type=int, default=2,
                            help='number of discriminators to be used in multiscale')
        parser.add_argument('--n_layers_discriminator', type=int, default=4,
                            help='# layers in each discriminator')
        parser.add_argument('--norm_discriminator', type=str, default='spectralinstance')
        parser.add_argument('--lr_discriminator', type=float, default=1e-3)

        parser.add_argument('--num_workers', type=int, default=1, help='Number of workers in dataloader')
        parser.add_argument('--display_freq', type=int, default=1000, help='Display images every X iterations')
        parser.add_argument('--max_epochs', type=int, default=44, help='Should converge in 44 epochs.')
        return parser
