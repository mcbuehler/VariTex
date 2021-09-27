import json
import os

import pytorch_lightning as pl
import torch
import wandb
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import WandbLogger, TensorBoardLogger
from torch.utils.data import DataLoader

try:
    from varitex.custom_callbacks.callbacks import ImageLogCallback
    from varitex.data.npy_dataset import NPYDataset
    from varitex.modules.pipeline import PipelineModule
    from varitex.options.train_options import TrainOptions
    from mutil.files import copy_src, mkdir
except ModuleNotFoundError:
    print("Have you added VariTex to your pythonpath?")
    print('To fix this error, go to the root path of the repository ".../VariTex/" \n '
          'and run \n'
          "export PYTHONPATH=$PYTHONPATH:$(pwd)")
    exit()


if __name__ == "__main__":
    pl.seed_everything(1234)

    opt = TrainOptions().parse()
    if opt.checkpoint is not None:
        # We load from a checkpoint, so let's load the opt as well
        path_checkpoint = opt.checkpoint
        opt_new = TrainOptions.load_from_json(os.path.join(os.path.dirname(opt.checkpoint), "../opt.json"))
        for k in opt.__dict__:
            # Overwrite options from the json with current options
            if getattr(opt_new, k, None) is None:
                setattr(opt_new, k, getattr(opt, k))
        # We need to set this again
        opt_new.checkpoint = path_checkpoint
        opt = opt_new

    if opt.dataset_split == "all":
        # The dataset has no splits or we want to use the full dataset.
        dataset = NPYDataset(opt, split="all", augmentation=True)
        dataloader = DataLoader(dataset, batch_size=opt.batch_size, num_workers=opt.num_workers, shuffle=True)
        do_validation = False
    else:
        # Separate dataloaders for train and validation.
        train_dataset, val_dataset = NPYDataset(opt, split="train", augmentation=True), NPYDataset(opt, split="val",
                                                                                                   augmentation=False)
        train_dataloader = DataLoader(train_dataset, batch_size=opt.batch_size, num_workers=opt.num_workers,
                                      shuffle=True)
        val_dataloader = DataLoader(val_dataset, batch_size=opt.batch_size, num_workers=opt.num_workers, shuffle=False)
        do_validation = True

    pipeline = PipelineModule(opt)
    gpus = torch.cuda.device_count()
    print("Using {} GPU".format(gpus))
    print("Writing results to {}".format(opt.path_out))
    mkdir(opt.path_out)

    if opt.logger == "wandb":
        wandb.login()
        logger = pl.loggers.WandbLogger(save_dir=opt.path_out, name=opt.experiment_name, project=opt.project)
        logger.log_hyperparams(opt)
        logger.watch(pipeline)
    elif opt.logger == "tensorboard":
        logger = TensorBoardLogger(
            save_dir=opt.path_out, name=opt.experiment_name
        )

    trainer = pl.Trainer(logger, gpus=gpus, max_epochs=opt.max_epochs, default_root_dir=opt.path_out,
                         terminate_on_nan=False,  # Terminate on nan is expensive
                         limit_val_batches=0.25, callbacks=[ImageLogCallback(opt), ModelCheckpoint()],
                         fast_dev_run=opt.debug,
                         resume_from_checkpoint=opt.checkpoint, weights_summary='top')

    if not opt.debug:
        # We keep a copy of the current source code and opt config
        src_path = os.path.dirname(os.path.realpath(__file__))
        copy_src(path_from=src_path,
                 path_to=opt.path_out)
        with open(os.path.join(opt.path_out, "opt.json"), 'w') as f:
            json.dump(opt.__dict__, f)

    if do_validation:
        trainer.fit(pipeline, train_dataloader, val_dataloader)
    else:
        trainer.fit(pipeline, dataloader)
