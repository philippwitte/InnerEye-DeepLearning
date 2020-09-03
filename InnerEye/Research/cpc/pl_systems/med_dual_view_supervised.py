from pytorch_lightning import Trainer
from argparse import Namespace

from InnerEye.Research.cpc.pl_systems.cpc_dual_view_med_train import MEDDualViewContrastivePredictiveCoding
from InnerEye.Research.cpc.pl_systems.bases.classifier_base import DownstreamLinearClassifier
from InnerEye.Research.cpc.utils.callbacks import get_best_model_from_checkpoint_cb
from InnerEye.Research.cpc.utils.logging_utils import get_default_logger
from InnerEye.Research.cpc.utils.callbacks import (get_early_stopping_cb,
                                                   get_model_checkpoint_cb)


class MEDDualViewSupervised(DownstreamLinearClassifier):
    def __init__(self, hparams, metrics=None):
        """
        TODO
        """
        # Init an MEDDualViewContrastivePredictiveCoding instance for leveraging its model and data loading
        from InnerEye.Research.cpc.argparsers import cpc_dual_view_med_argparser
        default_args = vars(cpc_dual_view_med_argparser().parse_args([]))
        default_args.update(vars(hparams))
        self.dual_view_encoder_pl = MEDDualViewContrastivePredictiveCoding(
            hparams=Namespace(**hparams)
        )
        self.dual_view_encoder_pl.prepare_data()
        super().__init__(
            encoder=self.dual_view_encoder_pl.get_encoder(),
            in_channels=getattr(hparams, "num_features") or self.encoder.encoding_dim,
            n_classes=hparams.n_classes,
            hparams=hparams,
            metrics=metrics
        )

    def train_dataloader(self):
        return self.dual_view_encoder_pl.train_loader

    def val_dataloader(self):
        return self.dual_view_encoder_pl.val_loader

    def test_dataloader(self):
        raise NotImplementedError()

    def forward(self, batch, transform_key):
        """ Encode a batch of images, linearly classify each encoding """
        # First returned value is the raw encoding, second is None
        v1_enc, v2_enc = self.dual_view_encoder_pl.forward(batch, transform_key)
        # pred = self.linear_clf(x_enc)
        # return pred
        raise NotImplementedError()


def entry_func(args, logger=None):
    """
    Init the PL module with passed args and launch training.

    :param args: Namespace, hyperparameters to pass to the PL system
    :param logger: AMLTensorBoardLogger, logger to use for AML and TensorBoard logging
    """
    # Get module
    linear_classifier = MEDDualViewSupervised(hparams=args)

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
