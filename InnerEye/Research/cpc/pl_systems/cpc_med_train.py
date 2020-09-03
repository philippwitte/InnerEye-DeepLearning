"""
Defines a system for training an encoder to extract useful representations
from images in an unsupervised manner using Constrastive Predictive Coding.
"""

import logging
import torch
from pytorch_lightning import Trainer
from apex import amp

from InnerEye.Research.cpc.utils.callbacks import get_model_checkpoint_cb
from InnerEye.Research.cpc.utils.training import get_random_train_val_split
from InnerEye.Research.cpc.dataloaders.med import (get_train_and_val_transforms,
                                                   get_csv_dataset,
                                                   get_h5_dataset,
                                                   create_data_loader)
from InnerEye.Research.cpc.transforms.default_med_transforms import (get_default_patch_transforms,
                                                                     get_default_slice_patch_transforms)
from InnerEye.Research.cpc.transforms.image_transforms import RandomCrop3d, CenterCrop3d
from InnerEye.Research.cpc.pl_systems.bases.cpc_base import ContrastivePredictiveCoding
from InnerEye.Research.cpc.callbacks import PlotEncodingSpace
from InnerEye.Research.cpc.utils.system import default_gpu_distribution
from InnerEye.Research.cpc.pl_systems.subsystems.rf_classifier_routine import run_classifier_routine_on_cache
from InnerEye.Research.cpc.pl_systems.subsystems.caching import EncodingsCache


class MEDContrastivePredictiveCoding(ContrastivePredictiveCoding):
    """
    Defines a PL system for patch-based CPC on 3D MEDs images + a time-dimension.

    See InnerEye.Research.cpc.pl_systems.cpc_base.ContrastivePredictiveCoding
    """

    def __init__(self, hparams):
        """
        :param hparams: Namespace
            Namespace object storing all hyperparameters that this PL system accepts according to its argparser func
            'cpc_med_argparser' at InnerEye.Research.cpc.argparsers
        """
        self.csv_dataset = None
        self.cache = EncodingsCache()
        super(MEDContrastivePredictiveCoding, self).__init__(
            hparams=hparams
        )
        self.encoder, self.loss, self.aggregators = default_gpu_distribution(self.encoder, self.loss, self.aggregators)

    def forward(self, images, transform_key=None):
        """
        Encoded all patches extracted from images in a batch

        Currently we work without the time-dimension.
        Combine batch and time dimension.

        Returned encodings are of shape [N*T, C_enc, g1, g2, g3] where g1, g2, g3 and the patch-grid dimensions.
        Images are patches within the encoder according to 'transform_key' to a [N*T, g1, g2, g3, C, d1, d2, d3] shape.
        See encoder.forward() for details.

        :param images: torch.Tensor, shape [N, T, D1, D2, D3]
        :param transform_key: str, a key indicating which transform object (stored in encoder) to apply to the input
                                   before encoding it, e.g. "train" or "val", which may alter augmentations etc.
        :return: torch.Tensor, encoded images shape [N*T, C_enc, g1, g2, g3]
        """
        if images.ndim == 5:
            # Flatten time-dimension into batch dim
            images = images.view(-1, *images.shape[2:])
        return self.encoder(images, transform_key=transform_key)

    def cache_batch(self, batch, encodings, tags):
        if "features" in batch:
            non_image_features = self.csv_dataset.select_features(
                features=batch["features"],
                select_columns=self.hparams.classifier_train_on_additional_features
            )
        else:
            non_image_features = None
        self.cache.cache_results(encodings=encodings,
                                 subjects=batch["subjects"],
                                 labels=batch["labels"],
                                 weeks=batch["weeks"],
                                 non_image_features=non_image_features,
                                 tags=tags)

    def training_step(self, batch, batch_idx):
        """
        Performs 1 step of training on a batch.

        :param batch: dict, a batch as output by CSVDataset or H5Dataset, storing at least "images", "segmentations",
                            "subjects", "weeks" and "labels".
        :param batch_idx: int, the index of the batch
        :return: dict, a dictionary of "loss": loss tensor and "log": dict of metrics to log.
        """
        encoded = self.forward(batch["images"], transform_key="train")
        if self.hparams.classifier_train_every:
            self.cache_batch(batch, encodings={"enc": encoded}, tags={"split": "train"})
        loss = self.loss(encoded, self.aggregators)
        log = {"train_loss": loss}
        return {"loss": loss, "log": log}

    def validation_step(self, batch, batch_idx):
        """
        Performs 1 step of validation on a batch.

        :param batch: dict, a batch as output by CSVDataset or H5Dataset, storing at least "images", "segmentations",
                            "subjects", "weeks" and "labels".
        :param batch_idx: int, the index of the batch
        :return: dict, a dictionary of "loss": loss tensor and "log": dict of metrics to log.
        """
        encoded = self.forward(batch["images"], transform_key="val")
        if self.hparams.classifier_train_every:
            self.cache_batch(batch, encodings={"enc": encoded}, tags={"split": "val"})
        val_loss = self.loss(encoded, self.aggregators)
        # Note: we add nothing to logs here, only interested in mean val scores
        return {"val_loss": val_loss}

    def validation_epoch_end(self, outputs):
        """
        Called after the last batch of validation data has been processed in self.validation_step. This method recives
        all outputs from self.validation_step as a list (list of dicts) and calls it's super implementation to aggregate
        these multiple outputs into a single output for the epoch.

        This method also calls run_classifier_routine_on_cache if self.hparams.classifier_train_every is >= 1.

        :param outputs: List of dicts of outputs from all validation steps.
        :return: dict, a combined output for the epoch, storing at least "val_loss" and "log".
        """
        val_outputs = super().validation_epoch_end(outputs)
        if self.hparams.classifier_train_every and not (self.current_epoch % self.hparams.classifier_train_every):
            with torch.enable_grad():
                # Train a downstream classifier with the current encoder
                classifier_results = run_classifier_routine_on_cache(
                    cache=self.cache,
                    cached_encodings_keys=["enc"],
                    csv_dataset=self.csv_dataset,
                    current_epoch=self.current_epoch,
                    classifier_train_on_additional_features=self.hparams.classifier_train_on_additional_features,
                    weeks=[0, 1]  # TODO consider alternative to hard-coding
                )
            val_outputs["log"].update(classifier_results)
        return val_outputs

    def clear_cache(self):
        """ Calls clear_cache on the stored EncodingsCache """
        logging.info("Clearing cache of {} elements".format(len(self.cache)))
        self.cache.clear_cache()

    def get_transforms(self):
        """
        Returns two transforms to apply to the training- and validation images respectively.

        The returned callables are of type MEDTransformer (see InnerEye.Research.cpc.dataloaders.med) and take a batch
        of images as input (torch.Tensor of shape [N, C, D1, D2, D3]) and returns a cropped, normalized, patched and
        augmented output of shape [N, g1, g2, g3, C, d1, d2, d3] where g* are grid dimensions and d* are patch-dimensions

        Uses the Patchifier (see InnerEye.Research.cpc.transforms.patch_transforms) object at self.patchifier to
        transform a batch of images into patches.

        :return: callable, callable
            Two callables to apply to training and validation batches respectively
        """
        # Init image transforms
        train_image_transform = RandomCrop3d(self.hparams.input_image_size)
        val_image_transform = CenterCrop3d(self.hparams.input_image_size)
        # Init patch transforms
        if not self.hparams.no_patch_augmentation:
            if self.hparams.patch_strides[1] == 1 and self.hparams.patch_strides[2] == 1:
                # Slice only inputs
                patch_transform = get_default_slice_patch_transforms()
            else:
                patch_transform = get_default_patch_transforms()
        else:
            patch_transform = None
        # Get MEDTransformer objects for training and validation subsets
        train_transform, val_transform = get_train_and_val_transforms(self.patchifier,
                                                                      train_image_transform=train_image_transform,
                                                                      train_patch_transform=patch_transform,
                                                                      val_image_transform=val_image_transform,
                                                                      val_patch_transform=None)
        return train_transform, val_transform

    def prepare_data(self, train_val_inds=None, train_split_fraction=None):
        """
        Overwrites the  dataloader of
        InnerEye.Research.cpc.pl_systems.cpc_base.ContrastivePredictiveCoding

        Performs the following actions:
            1) Initializes a CSVData or H5Dataset as per self.hparams, sets self.csv_dataset attribute
            2) Initializes a training- and validation batch loader from self.csv_dataset and sets the self.train_loader
               and self.val_loader attributes.
            3) Sets self.train_transform and self.val_transform on the encoder for GPU augmentation

        :param train_val_inds: None or list of len 2
            List of two sets of inds to use for training and validation respectively. If None, generates random split
            according to train_split_fraction
        :param train_split_fraction: None or float
            Fraction of data to use for training. Will be overwritten by self.hparams.train_split_fraction if None.
        """
        # Mount, download or get path to dataset
        from InnerEye.Research.cpc.utils.azure_utils import mount_or_download_dataset
        hdf5_path = mount_or_download_dataset(self.hparams.hdf5_files_path, getattr(self.hparams, "azure_dataset_id"))
        logging.info("Using dataset at path: {} (exists={})".format(hdf5_path, hdf5_path.exists()))

        # Get dataset objects
        dataset_class = get_csv_dataset if not self.hparams.non_sequential_inputs else get_h5_dataset
        self.csv_dataset = dataset_class(
            csv_file_path=hdf5_path/self.hparams.csv_file_name,
            path_to_hdf5_files=hdf5_path,
            use_hdf5_files=True,
            load_segmentations=False,
            input_weeks=self.hparams.input_weeks,
            target_weeks=self.hparams.target_weeks
        )
        if train_split_fraction is None:
            train_split_fraction = self.hparams.train_split_fraction
        if train_val_inds is None:
            train_inds, val_inds = get_random_train_val_split(num_inds=len(self.csv_dataset),
                                                              train_split_fraction=train_split_fraction)
        else:
            if len(train_val_inds) != 2:
                raise ValueError("'train_val_inds' must be a list of length 2 of training and validation inds.")
            train_inds, val_inds = train_val_inds
        self.train_loader = create_data_loader(self.csv_dataset,
                                               inds=train_inds,
                                               batch_size=self.hparams.batch_size,
                                               num_workers=self.hparams.num_workers,
                                               random_sampling=True)
        self.val_loader = create_data_loader(self.csv_dataset,
                                             inds=val_inds,
                                             batch_size=self.hparams.batch_size,
                                             num_workers=self.hparams.num_workers,
                                             random_sampling=False)
        logging.info("N training batches:   {}".format(len(self.train_loader)))
        logging.info("N validation batches: {}".format(len(self.val_loader)))

        # Set transforms on encoder object: This is to do DP augmentation across GPUs
        self.get_encoder().set_transforms({"train": self.train_transform, "val": self.val_transform})

    def configure_optimizers(self):
        """
        Configure the optimizer + AMP + encoder DataParallel
        """
        encoder_params = list(self.get_encoder().parameters())
        loss_params = list(self.loss.parameters())
        if self.aggregators is not None:
            loss_params += list(self.aggregators.parameters())
        optimizer = torch.optim.Adam(params=encoder_params+loss_params, lr=self.hparams.learning_rate)

        # Prepare modules for AMP
        if self.hparams.amp_training:
            (self.encoder,), optimizer = amp.initialize([self.encoder], optimizer, opt_level="O1")
        self.encoder = torch.nn.DataParallel(self.encoder, device_ids=[0, 1, 2, 3], output_device=3)
        return optimizer

    def backward(self, trainer, loss, optimizer, optimizer_idx):
        """ Overwrites the default backward method to perform AMP scaled loss even with PL was init with no GPUs """
        if self.hparams.amp_training:
            # Use scaled loss with AMP
            with amp.scale_loss(loss, optimizer) as scaled_loss:
                scaled_loss.backward()
        else:
            # Pass to default backward method
            super(MEDContrastivePredictiveCoding, self).backward(trainer, loss, optimizer, optimizer_idx)


def entry_func(args, logger):
    """
    Init the PL module with passed args and launch training.

    :param args: Namespace, hyperparameters to pass to the PL system
    :param logger: AMLTensorBoardLogger, logger to use for AML and TensorBoard logging
    """
    cpc = MEDContrastivePredictiveCoding(hparams=args)

    # Configure the trainer
    monitor = "val_loss" if getattr(args, "train_split_fraction", 0) != 1 else "train_loss"
    checkpoint_cb = get_model_checkpoint_cb(monitor=monitor, save_top_k=15)
    trainer = Trainer(default_save_path="outputs",
                      logger=logger,
                      gpus=None,  # Currently we manually place modules on the GPUs as needed
                      checkpoint_callback=checkpoint_cb,
                      num_sanity_val_steps=0,
                      row_log_interval=10,
                      fast_dev_run=False,
                      print_nan_grads=False,
                      resume_from_checkpoint=args.resume_from or None,
                      progress_bar_refresh_rate=1,
                      show_progress_bar=not bool(logger.aml_run),
                      callbacks=[PlotEncodingSpace(out_dir="outputs/plots",
                                                   cache_encoding_keys=["enc"],
                                                   weeks=[0, 1])],  # TODO: consider alternative to hard-coding
                      min_epochs=args.num_epochs,
                      max_epochs=args.num_epochs)
    trainer.fit(cpc)
