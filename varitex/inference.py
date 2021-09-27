"""
CP=PATH/TO/CHECKPOINT.ckpt
CP=$OP/own/LDS/pretrained/checkpoints/ep44.ckpt
CUDA_VISIBLE_DEVICES=0 python varitex/inference.py --checkpoint $CP --dataset_split val
"""
import os

import pytorch_lightning as pl
from mutil.object_dict import ObjectDict

try:
    from varitex.options import varitex_default_options
    from varitex.evaluation import inference
    from varitex.options.eval_options import EvalOptions
except ModuleNotFoundError:
    print("Have you added VariTex to your pythonpath?")
    print('To fix this error, go to the root path of the repository ".../VariTex/" \n '
          'and run \n'
          "export PYTHONPATH=$PYTHONPATH:$(pwd)")
    exit()

if __name__ == "__main__":
    pl.seed_everything(1234)
    opt = EvalOptions().parse().__dict__

    default_opt = varitex_default_options()
    default_opt.update(opt)
    opt = ObjectDict(default_opt)
    assert opt.checkpoint is not None, "Please specify a checkpoint file."

    checkpoint_folder = os.path.dirname(opt.checkpoint)

    # Runs inference on FFHQ
    inference.inference_ffhq(opt, n=30, results_folder=os.path.join(opt.path_out, 'inference_ffhq'))

    # # Runs different poses for each sample in FFHQ (shape and expressions extracted from FFHQ)
    inference.inference_posed_ffhq(opt, n=30, results_folder=os.path.join(opt.path_out, 'inference_posed_ffhq'))
    #
    # # Runs different poses for each sample in FFHQ (random shape and expressions).
    # latents.npy should contain the latent distribuations predicted from the holdout set, see inference.inference_ffhq(...).
    inference.inference_posed(opt, n=30, results_folder=os.path.join(opt.path_out, 'inference_posed_random'),
                              path_latent=os.path.join(opt.path_out, 'inference_ffhq/latents.npy'))
