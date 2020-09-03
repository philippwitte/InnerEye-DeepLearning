"""
Defines a system for training an encoder to extract useful representations
from images in an unsupervised manner using Constrastive Predictive Coding.
"""

import logging

from pytorch_lightning import Trainer
from InnerEye.Research.cpc.dataloaders.stl10 import get_unlabelled_data_loader
from InnerEye.Research.cpc.utils.logging_utils import get_default_logger
from InnerEye.Research.cpc.utils.callbacks import (get_early_stopping_cb,
                                                   get_model_checkpoint_cb)
from InnerEye.Research.cpc.pl_systems.bases.cpc_base import ContrastivePredictiveCoding


class STL10ContrastivePredictiveCoding(ContrastivePredictiveCoding):
    """
    Implements the prepare_data method for STL10
    See base class
    """
    def __init__(self, hparams):
        """
        :param hparams: Namespace
            Namespace object storing all hyperparameters that this PL system accepts according to its argparser func
            'cpc_stl10_argparser' at InnerEye.Research.cpc.argparsers
        """
        super(STL10ContrastivePredictiveCoding, self).__init__(
            hparams=hparams
        )

    def prepare_data(self):
        self.train_loader, self.val_loader = get_unlabelled_data_loader(
            base_folder=self.hparams.data_folder,
            download_dataset=self.hparams.download_data,
            batch_size=self.hparams.batch_size,
            input_image_size=self.hparams.input_image_size,
            patch_sizes=self.hparams.patch_sizes,
            patch_strides=self.hparams.patch_strides,
            num_workers=self.hparams.num_workers,
            augment_patches=not self.hparams.no_patch_augmentation
        )
        logging.info("N training batches:   {}".format(len(self.train_loader)))
        logging.info("N validation batches: {}".format(len(self.val_loader)))


def entry_func(args, logger=None):
    """
    Init the PL module with passed args and launch training.

    :param args: Namespace, hyperparameters to pass to the PL system
    :param logger: Not used
    """
    # Get module
    cpc = STL10ContrastivePredictiveCoding(hparams=args)

    # Configure the trainer
    monitor = "val_loss" if getattr(args, "train_split_fraction", 0) != 1 else "train_loss"
    checkpoint_cb = get_model_checkpoint_cb(monitor=monitor, save_top_k=15)
    trainer = Trainer(default_save_path="outputs",
                      logger=get_default_logger(args),
                      gpus=1,
                      checkpoint_callback=checkpoint_cb,
                      num_sanity_val_steps=3,
                      row_log_interval=50,
                      resume_from_checkpoint=args.resume_from or None,
                      min_epochs=args.num_epochs,
                      max_epochs=args.num_epochs)
    trainer.fit(cpc)
