import logging
import torch

from argparse import Namespace

from pytorch_lightning import Trainer
from InnerEye.Research.cpc.pl_systems.bases.cpc_base import ContrastivePredictiveCoding
from InnerEye.Research.cpc.pl_systems.bases.classifier_base import DownstreamLinearClassifier
from InnerEye.Research.cpc.dataloaders.stl10 import (get_training_data_loder,
                                                     get_eval_data_loader)
from InnerEye.Research.cpc.utils.callbacks import (get_early_stopping_cb,
                                                   get_model_checkpoint_cb,
                                                   get_best_model_from_checkpoint_cb)
from InnerEye.Research.cpc.utils.system import get_device


class CPCClassifier(DownstreamLinearClassifier):
    def __init__(self, device, hparams):
        """
        :param device: torch.device
            Device to place the encoder for inference
        :param hparams: Namespace
            Namespace object storing all hyperparameters that this PL system accepts according to its argparser func
            'cpc_downstream_clf_argparser' at InnerEye.Research.cpc.argparsers
        """
        # Init the CPC trained encoder model (we do not use the aggregator)
        ckpt = torch.load(hparams.cpc_model_ckpt, map_location=device)
        enc_hparams = Namespace(**ckpt["hparams"])
        enc_hparams.use_aggregator = False
        encoder = ContrastivePredictiveCoding(hparams=enc_hparams).encoder
        if not hparams.random_init:
            encoder.load_state_dict(ckpt["state_dict"], strict=False)
        super(CPCClassifier, self).__init__(
            encoder=encoder,
            in_channels=getattr(hparams, "num_features") or self.encoder.encoding_dim,
            n_classes=hparams.n_classes,
            hparams=hparams
        )

    def set_train_and_val_dataloaders(self):
        self.train_loader, self.val_loader = get_training_data_loder(
            base_folder=self.hparams.data_folder,
            download_dataset=self.hparams.download_data,
            batch_size=self.hparams.batch_size,
            input_image_size=self.encoder.hparams.input_image_size,
            patch_sizes=self.encoder.hparams.patch_sizes,
            patch_strides=self.encoder.hparams.patch_strides,
            train_data_fraction=self.hparams.train_data_fraction,
            num_workers=self.hparams.num_workers
        )
        logging.info("N training batches:   {}".format(len(self.train_loader)))
        logging.info("N validation batches: {}".format(len(self.val_loader)))

    def test_dataloader(self):
        return get_eval_data_loader(
            base_folder=self.hparams.data_folder,
            download_dataset=self.hparams.download_data,
            batch_size=self.hparams.batch_size,
            input_image_size=self.encoder.hparams.input_image_size,
            patch_sizes=self.encoder.hparams.patch_sizes,
            patch_strides=self.encoder.hparams.patch_strides,
            num_workers=self.hparams.num_workers
        )


def entry_func(args, logger=None):
    """
    Init the PL module with passed args and launch training.

    :param args: Namespace, hyperparameters to pass to the PL system
    :param logger: AMLTensorBoardLogger, logger to use for AML and TensorBoard logging
    """
    # Get module
    device = get_device(args.use_gpu)

    # Get module
    linear_classifier = CPCClassifier(device=device, hparams=args)

    # Configure the trainer
    checkpoint_cb = get_model_checkpoint_cb(monitor="avg_val_loss")
    early_stop_cb = get_early_stopping_cb(monitor="avg_val_loss")
    trainer = Trainer(default_save_path="outputs",
                      logger=logger,
                      gpus=int(args.use_gpu),
                      checkpoint_callback=checkpoint_cb,
                      num_sanity_val_steps=3,
                      print_nan_grads=True,
                      row_log_interval=25,
                      early_stop_callback=early_stop_cb,
                      resume_from_checkpoint=args.resume_from or None,
                      min_epochs=args.num_epochs,
                      max_epochs=args.num_epochs*2)
    try:
        trainer.fit(linear_classifier)
    except KeyboardInterrupt:
        pass
    finally:
        ckpt_path = get_best_model_from_checkpoint_cb(checkpoint_cb)
        trainer.restore(ckpt_path, on_gpu=args.use_gpu)
        trainer.test()
