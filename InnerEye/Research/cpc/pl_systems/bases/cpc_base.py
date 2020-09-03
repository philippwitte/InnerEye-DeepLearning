"""
Defines a system for training an encoder to extract useful representations
from images in an unsupervised manner using Constrastive Predictive Coding.

None of the classes implemented here are to be directly initialized. They
define abstract base classes that are not implemented, such as self.prepare_data.
"""

import logging
import torch
import pytorch_lightning as pl
from abc import abstractmethod

from InnerEye.Research.cpc.utils.training import compute_average_metrics
from InnerEye.Research.cpc.models import (ResnetEncoder2d, ResnetEncoder3d,
                                          ConvAggregator2d, ConvAggregator3d)
from InnerEye.Research.cpc.transforms.patch_transforms import Patchifier


class _ConstrastivePredictiveCodingPLModule(pl.LightningModule):
    """
    Base CPC PL module class, defines required attributes and methods but is
    normally not initialized directly.
    """

    def __init__(self, hparams):
        super(_ConstrastivePredictiveCodingPLModule, self).__init__()

        # Store the hyperparameters, must be named "hparams" for auto storage
        self.hparams = hparams

        # Transformation attributes set in self.set_transforms
        self.patchifier = self.get_patchifier()
        self.train_transform, self.val_transform = self.get_transforms()

        # Dataloaders will be set in self.prepare_data
        self.train_loader, self.val_loader, self.test_loader = None, None, None

    def forward(self, images, transform_key=None):
        """
        Encode a batch of images.
        This method is overloaded by most PL systems (all but STl10 trainer) to supports their particular inputs and
        encoders.

        :param images: torch.Tensor, patched image of shape [N, g1, g2, C, d1, d2]
        :param transform_key: Unused argument defining an interface for sub-classes
        :return: torch.Tensor, encoded patches of shape [N, g1, g2, C_env]
        """
        if isinstance(images, (tuple, list)):
            # We do not use potentially passed labels (e.g. STL10 dataloader passes empty labels array)
            images = images[0]
        encoded = self.encoder(images)
        return encoded

    def training_step(self, batch, batch_idx):
        """
        Perform 1 step of training on a batch as output by self.train_dataloader.

        Note: This implementation is overloaded by most PL systems (all but STl10 trainer).

        :param batch: tuple, (X, y) of batched data as output by self.train_dataloader
        :param batch_idx: int, batch index
        :return: dict
            Must store a key "loss" mapping to the loss tensor from which backprop. is initiated. May also store a key
            "log" mapping to dict of metrics to log (e.g. to TensorBoard).
        """
        encoded = self.forward(batch)
        loss = self.loss(encoded, self.aggregators)
        log = {"train_loss": loss}
        return {"loss": loss, "log": log}

    def validation_step(self, batch, batch_idx):
        """
        Perform 1 step of validation on a batch as output by self.val_dataloader.

        Note: This implementation is overloaded by most PL systems (all but STl10 trainer).

        :param batch: tuple, (X, y) of batched data as output by self.val_dataloader
        :param batch_idx: int, batch index
        :return: dict
            Must store a key "val_loss" mapping to the loss tensor. We add nothing to the log here, logging of metrics
            is handled in self.validation_epoch_end.
        """
        encoded = self.forward(batch)
        val_loss = self.loss(encoded, self.aggregators)
        # Note: we add nothing to logs here, only interested in mean val scores
        return {"val_loss": val_loss}

    def validation_epoch_end(self, outputs):
        """
        This method gets called once per epoch after all training and validation batches have been processed.
        The method receives a list of outputs from all calls to self.validation_step and may serve to aggregate results
        etc.

        :param outputs: list, list of outputs from all validation steps.
        :return: dict, storing at least "val_loss" mapping, optionally logs and other items.
        """
        if len(self.val_loader) != 0:
            val_outputs = compute_average_metrics(outputs)
            return {"val_loss": val_outputs["avg_val_loss"], "log": val_outputs}
        else:
            # No validation set
            return {"log": {}}

    def configure_optimizers(self):
        """
        Configure and return an optimizer to use for training
        :return: torch.optim Optimizer object
        """
        params = list(self.encoder.parameters()) + list(self.loss.parameters())
        if self.aggregators is not None:
            params += list(self.aggregators.parameters())
        return torch.optim.Adam(
            params=params,
            lr=self.hparams.learning_rate
        )

    @abstractmethod
    def prepare_data(self):
        """
        Should at least set the train_loader and val_loader attributes with iterable dataloader objects.
        For some PL systems the concrete class may also implement setting transforms on the encoder module.
        See e.g. MEDDualViewContrastivePredictiveCoding
        :return: None
        """
        raise NotImplemented

    def get_transforms(self):
        """
        Return a transform object for the training and validation datasets
        Normally, we want to at least apply the stored Patchifier, but this
        method can be overwritten to introduce e.g. augmentation.
        :return: Callable, Callable
        """
        return self.patchifier, self.patchifier

    def get_patchifier(self):
        """
        Returns a Patchifier object as specified in hparams
        A Patchifier patches an input [C, D1, D2, D3] --> [g1, g2, g3, C, d1, d2, d3] where g* are patch grid dimensions
        and d* are spatial patch dimensions. See InnerEye.Research.cpc.transforms.patch_transforms.
        :return: Patchifier, Callable
        """
        return Patchifier(
            patch_sizes=self.hparams.patch_sizes,
            patch_strides=self.hparams.patch_strides
        )

    def get_encoder(self):
        """
        Returns the base Encoder Module
        :return: Encoder model, torch.nn.Module
        """
        is_dp_module = isinstance(self.encoder, torch.nn.DataParallel)
        return self.encoder.module if is_dp_module else self.encoder

    def train_dataloader(self):
        """ Handle needed by PyTorch Lightning """
        return self.train_loader

    def val_dataloader(self):
        """ Handle needed by PyTorch Lightning """
        return self.val_loader

    def test_dataloader(self):
        """ Handle needed by PyTorch Lightning. Normally, we do not use a test_loader with this module """
        return self.test_loader


class ContrastivePredictiveCoding(_ConstrastivePredictiveCodingPLModule):
    """
    Basic CPC module
    Defines a system for training an encoder to extract useful representations
    from images in an unsupervised manner using Constrastive Predictive Coding.

    Implements the typical encoder + aggregator + InfoNCELoss setup, but
    does not implement all abstract methods of _ConstrastivePredictiveCodingPLModule
    and thus must be subclassed.
    """

    def __init__(self, hparams):
        """
        TODO
        """
        super(ContrastivePredictiveCoding, self).__init__(hparams)
        if len(self.hparams.patch_sizes) == 2:
            resnet_encoder = ResnetEncoder2d
            conv_aggregator = ConvAggregator2d
        elif len(self.hparams.patch_sizes) == 3:
            resnet_encoder = ResnetEncoder3d
            conv_aggregator = ConvAggregator3d
        else:
            raise NotImplementedError("Only implemented for 2d/3d data.")

        # Init encoder and aggregator model (if specified)
        self.encoder = resnet_encoder(
            num_channels=self.hparams.input_channels,
            init_dim=self.hparams.encoder_init_dim,
            encoding_dim=self.hparams.encoder_out_dim,
            res_block_depth=self.hparams.encoder_res_block_depth,
            input_patch_size=self.hparams.patch_sizes,
            use_norm=self.hparams.encoder_use_norm,
        )
        # Define InfoNCE (CPC) loss
        from InnerEye.Research.cpc.losses.info_nce import InfoNCELoss
        self.loss = InfoNCELoss(in_channels=self.hparams.encoder_out_dim,
                                negative_samples=self.hparams.negative_samples,
                                score_model_depth=self.hparams.score_model_depth,
                                k_prediction_steps=self.hparams.k_prediction_steps,
                                num_skip_steps=self.hparams.num_skip_steps,
                                directions=self.hparams.directions,
                                apply_unit_sphere_normalization=not self.hparams.no_unit_sphere_norm)
        if self.hparams.use_aggregator:
            # Init an aggregation model for each CPC direction
            self.aggregators = torch.nn.ModuleDict({})
            for direction in self.loss.directions:
                self.aggregators[direction] = conv_aggregator(
                    in_filters=self.hparams.encoder_out_dim,
                    hidden_filters=self.hparams.aggregator_hidden_dim,
                    out_filters=self.hparams.encoder_out_dim,
                    depth=self.hparams.aggregator_depth,
                    kernel_size=self.hparams.aggregator_kernel_size,
                    use_norm=self.hparams.aggregator_use_norm,
                )
            logging.info('Aggregator(s) parameters: {:.4f} million'.format(
                sum(s.numel() for s in self.aggregators.parameters()) / 10**6)
            )
        else:
            self.aggregators = None


class DualViewContrastivePredictiveCoding(_ConstrastivePredictiveCodingPLModule):
    def __init__(self, hparams):
        """
        Implements basic setup for dual-view multi-view CPC

        Implements the typical encoder + segmentation encoder + DualViewInfoNCELoss setup, but
        does not implement all abstract methods of _ConstrastivePredictiveCodingPLModule
        and thus must be subclassed.
        """
        super(DualViewContrastivePredictiveCoding, self).__init__(hparams)
        if len(self.hparams.patch_sizes) == 2:
            resnet_encoder = ResnetEncoder2d
        elif len(self.hparams.patch_sizes) == 3:
            resnet_encoder = ResnetEncoder3d
        else:
            raise NotImplementedError("Only implemented for 2d/3d data.")
        from InnerEye.Research.cpc.models.multi_view_encoder import DualViewEncoder
        from InnerEye.Research.cpc.losses.multi_view_info_nce import DualViewInfoNCELoss
        image_encoder = resnet_encoder(
            num_channels=self.hparams.input_channels,
            init_dim=self.hparams.encoder_init_dim,
            encoding_dim=self.hparams.encoder_out_dim,
            res_block_depth=self.hparams.encoder_res_block_depth,
            input_patch_size=self.hparams.patch_sizes,
            use_norm=self.hparams.encoder_use_norm
        )
        segmentation_encoder = resnet_encoder(
            num_channels=self.hparams.segmentation_encoder_input_channels,
            init_dim=self.hparams.segmentation_encoder_init_dim,
            encoding_dim=self.hparams.segmentation_encoder_out_dim,
            res_block_depth=self.hparams.segmentation_encoder_res_block_depth,
            input_patch_size=self.hparams.patch_sizes,
            use_norm=self.hparams.segmentation_encoder_use_norm
        )
        self.encoder = DualViewEncoder(
            image_encoder=image_encoder,
            segmentation_encoder=segmentation_encoder
        )
        # Init dual view loss
        self.loss = DualViewInfoNCELoss(
            negative_samples=self.hparams.negative_samples,
            tau=self.hparams.tau,
            keep_excluded_with_tau=self.hparams.tau_for_excluded_negatives,
            bidirectional=True,
            negative_sampling_mode=self.hparams.negative_sampling_mode,
            negatives_exclude_weeks_within_dist=self.hparams.negatives_exclude_weeks_within_dist,
            negatives_exclude_spatial_within_dist=self.hparams.negatives_exclude_spatial_within_dist
        )
