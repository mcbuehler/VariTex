import argparse
import json
import os
from types import SimpleNamespace


class BaseOptions:
    initialized = False

    def initialize(self, parser):
        parser.add_argument('--dataroot_npy', default=os.path.join(os.getenv("DP"), 'FFHQ/preprocessed_dataset'),
                            help='Path to the folder with the preprocessed datasets. Should contain .npy files "R", "t", "s", "sp", "ep", "segmentation", "uv", "filename", and a .npz file "dataset_splits".')
        parser.add_argument('--image_folder', default=os.path.join(os.getenv("DP"), 'FFHQ/images'),
                            help='Path to the folder that contains *.png images.')
        parser.add_argument('--path_out', default=os.path.join(os.getenv("OP", ""), 'varitex'),
                            help='Path to the folder where the outputs should be saved.')
        parser.add_argument('--path_bfm',
                            default=os.path.join(os.getenv("FP"), "basel_facemodel/model2017-1_face12_nomouth.h5"),
                            help='Basel face model (face only). Please use "model2017-1_face12_nomouth.h5"')
        parser.add_argument('--path_uv', default=os.path.join(os.getenv("FP"), "basel_facemodel/face12.json"),
                            help='UV parameterization. Download from "https://github.com/unibas-gravis/parametric-face-image-generator/blob/master/data/regions/face12.json".')
        parser.add_argument('--device', default='cuda', help='')
        parser.add_argument('--dataset', default='FFHQ', help='')
        parser.add_argument('--keep_background', action='store_true',
                            help="If True the dataloader won't remove the background and the model will also generate the background.")
        parser.add_argument('--bg_color', type=str, default='black',
                            help="Defines how to fill the masked regions. Only if keep_background=False. Check dataloader for details.")
        parser.add_argument('--transform_mode', type=str, default='all',
                            help='string with letters in {s, t, s, f}. d: rotate, t: translate, s: scale, f: flip')
        parser.add_argument('--logger', type=str, default='wandb', help='tensorboard | wandb')

        parser.add_argument('--checkpoint', type=str, default=None, help="Path to checkpoint file.")
        parser.add_argument('--dataset_split', type=str, default='train', help='all | train | val')
        parser.add_argument('--texture_nc', type=int, default=16, help='# features in neural texture')
        parser.add_argument('--texture_dim', type=int, default=256, help='Height and width of square neural texture')
        parser.add_argument('--image_h', type=int, default=256, help='image height')
        parser.add_argument('--image_w', type=int, default=256, help='image width')
        parser.add_argument('--batch_size', type=int, default=7)
        parser.add_argument('--uv_mask_value', type=float, default=0.0,
                            help='What values to use for invalid uv regions')

        parser.add_argument('--latent_dim', type=int, default=256,
                            help='Dimension of the full latent code z before splitting into z_face and z_additive.')
        parser.add_argument('--nc_feature2image', type=int, default=64,
                            help="# feature channels in the Feature2Image renderer.")
        parser.add_argument('--feature2image_num_layers', type=int, default=5,
                            help="Number of leves in the Feature2Image renderer.")

        parser.add_argument('--nc_decoder', type=int, default=32, help="# feature channels in texture decoder.")
        parser.add_argument('--semantic_regions', type=int, nargs='+', default=list(range(1, 16)),
                            help="Defines region indices that should be considered as foreground. You can find a label list here: https://github.com/switchablenorms/CelebAMask-HQ/blob/master/face_parsing/Data_preprocessing/g_mask.py")

        parser.add_argument('--experiment_name', default='default', help='Experiment name for logger')
        parser.add_argument('--project', default="varitex", help="Project name for wandb logger")
        parser.add_argument('--debug', action="store_true",
                            help="Enable debug mode, i.e., running a fast_dev_run in the Trainer.")
        self.initialized = True
        return parser

    def parse(self):
        # initialize parser with basic options
        if not self.initialized:
            self.parser = argparse.ArgumentParser()
            self.parser = self.initialize(self.parser)

        opt = self.parser.parse_args()
        self.print_options(opt)
        return opt

    def print_options(self, opt):
        message = ''
        message += '----------------- Options ---------------\n'
        for k, v in sorted(vars(opt).items()):
            comment = ''
            default = self.parser.get_default(k)
            if v != default:
                comment = '\t[default: %s]' % str(default)
            message += '{:>25}: {:<30}{}\n'.format(str(k), str(v), comment)
        message += '----------------- End -------------------'
        print(message)

    @staticmethod
    def load_from_json(path_opt):
        if not os.path.exists(path_opt):
            raise Warning("opt.json not found: '{}'".format(path_opt))

        with open(path_opt, 'r') as f:
            # Load and overwrite from a stored opt file
            opt = json.load(f, object_hook=lambda d: SimpleNamespace(**d))
        print("Loaded options from '{}'".format(path_opt))
        return opt

    @classmethod
    def update_from_json(cls, opt, path_opt, keep_keys=('checkpoint',)):
        # We load from a checkpoint, so let's load the opt as well
        to_keep = {k: getattr(opt, k) for k in keep_keys}
        opt_new = cls.load_from_json(path_opt)
        for k in opt.__dict__:
            if getattr(opt_new, k, None) is None:
                setattr(opt_new, k, getattr(opt, k))
        # We need to set this again
        for k, v in to_keep.items():
            setattr(opt_new, k, v)
        return opt_new
