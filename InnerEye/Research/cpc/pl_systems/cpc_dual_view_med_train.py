import logging
import torch

from pytorch_lightning import Trainer
from apex import amp
from torchvision.transforms import Compose

from InnerEye.Research.cpc.utils.callbacks import get_model_checkpoint_cb
from InnerEye.Research.cpc.utils.training import get_random_train_val_split
from InnerEye.Research.cpc.dataloaders.med import (get_train_and_val_transforms,
                                                   get_csv_dataset,
                                                   get_h5_dataset,
                                                   create_data_loader)
from InnerEye.Research.cpc.transforms.default_med_transforms import get_default_slice_patch_transforms
from InnerEye.Research.cpc.transforms.image_transforms import (RandomCrop3d, CenterCrop3d, OneHotEncode,
                                                               SliceCropper, SliceDistanceTransform)
from InnerEye.Research.cpc.pl_systems.bases.cpc_base import DualViewContrastivePredictiveCoding
from InnerEye.Research.cpc.callbacks import PlotEncodingSpace
from InnerEye.Research.cpc.utils.system import default_gpu_distribution
from InnerEye.Research.cpc.pl_systems.subsystems.caching import EncodingsCache


class MEDDualViewContrastivePredictiveCoding(DualViewContrastivePredictiveCoding):
    """
    Defines a PL system for dual-view (MED images + MED segmentations) CPC on 3D MEDs images + a time-dimension.

    See InnerEye.Research.cpc.pl_systems.cpc_dual_view_med_train DualViewContrastivePredictiveCoding
    """

    def __init__(self, hparams):
        """
        :param hparams: Namespace
            Namespace object storing all hyperparameters that this PL system accepts according to its argparser func
            'cpc_dual_view_med_argparser' at InnerEye.Research.cpc.argparsers
        """
        self.csv_dataset = None
        self.cache = EncodingsCache()
        super(MEDDualViewContrastivePredictiveCoding, self).__init__(
            hparams=hparams
        )
        self.encoder, self.loss, _ = default_gpu_distribution(self.encoder, self.loss, aggregators=None)

    def forward(self, batch, transform_key=None):
        """
        Encoded all patches extracted from images and segmentations in a batch

        Currently we work without the time-dimension.
        Combine batch and time dimension.

        Returned encodings are of shape [N, T, C, g1, g2, g3] where g1, g2, g3 and the patch-grid dimensions.

        :param batch: dict, containing at least keys "images" and "segmentations". Images and segmentation should be
                            torch.Tensor objects of shape [N, T, d1, d2, d3]
        :param transform_key: str, a key indicating which transform object (stored in encoder) to apply to the input
                                   before encoding it, e.g. "train" or "val", which may alter augmentations etc.
        :return: torch.Tensor of images encoded (shape [N, T, C, g1, g2, g3]),
                 torch.Tensor of segmentations encoded (shape [N, T, C, g1, g2, g3])
        """
        images, segmentations = batch["images"], batch["segmentations"]
        n_subjects, n_weeks = images.shape[:2]
        if images.ndim == 5:
            # Flatten time-dimension into batch dim
            images = images.view(-1, *images.shape[2:])
        if segmentations.ndim == 5:
            # Flatten time-dimension into batch dim
            segmentations = segmentations.view(-1, *segmentations.shape[2:])
        # Encode images and segmentations
        v1_encoded, v2_encoded = self.encoder(images=images,
                                              segmentations=segmentations,
                                              transform_key=transform_key)
        # Reshape to [N, T, C, d1, d2, d3] view
        return v1_encoded.view(n_subjects, n_weeks, *v1_encoded.shape[1:]), \
               v2_encoded.view(n_subjects, n_weeks, *v2_encoded.shape[1:])

    def compute_losses(self, v1_encoded, v2_encoded, log_prefix):
        """
        Compute L(V1, V2) with or without a memory bank as per self.hparams.use_memory_bank
        :param v1_encoded: torch.Tensor, encoded images (shape [N, T, C, g1, g2, g3]),
        :param v2_encoded: torch.Tensor, encoded segmentations (shape [N, T, C, g1, g2, g3]),
        :param log_prefix: str, prefix for logging, e.g. with prefix="train" a metric "loss" logs as "train_loss"
        :return:
        """
        if self.hparams.use_memory_bank:
            memory_bank = self.cache.get_encodings_from_cache(["v1_enc", "v2_enc"],
                                                              flatten_time_dim=False,
                                                              pool_to_vector=False,
                                                              as_memory_bank=True)
        else:
            memory_bank = None
        loss, v1_v2_loss, v2_v1_loss = self.loss(v1_enc=v1_encoded,
                                                 v2_enc=v2_encoded,
                                                 v1_memory_bank=memory_bank,
                                                 v2_memory_bank=memory_bank)
        log = {f"{log_prefix}_loss": loss,
               f"{log_prefix}_v1_v2_loss": v1_v2_loss,
               f"{log_prefix}_v2_v1_loss": v2_v1_loss}
        return loss, log

    def cache_batch(self, batch, encodings, tags):
        """
        Wrapper around self.cache.cache_results

        :param batch: dict, a batch as output by CSVDataset or H5Dataset
        :param encodings: dict, mapping between a key, e.g. "v1_enc" to a torch.Tensor of encodings, typically of shape
                          [N, T, C, ...], where ... are possible grid dimensions.
        :param tags: dict, additional information to cache for each subject. Values in tags must be either a single elem
                     or a list-like of len(batch["subjects'])
        """
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
        v1_encoded, v2_encoded = self.forward(batch, transform_key="train")
        if self.hparams.classifier_train_every:
            self.cache_batch(batch, encodings={"v1_enc": v1_encoded, "v2_enc": v2_encoded}, tags={"split": "train"})
        loss, log = self.compute_losses(v1_encoded=v1_encoded,
                                        v2_encoded=v2_encoded,
                                        log_prefix="train")
        return {"loss": loss, "log": log}

    def validation_step(self, batch, batch_idx):
        """
        Performs 1 step of validation on a batch.

        :param batch: dict, a batch as output by CSVDataset or H5Dataset, storing at least "images", "segmentations",
                            "subjects", "weeks" and "labels".
        :param batch_idx: int, the index of the batch
        :return: dict, a dictionary of "loss": loss tensor and "log": dict of metrics to log.
        """
        v1_encoded, v2_encoded = self.forward(batch, transform_key="val")
        if self.hparams.classifier_train_every:
            self._cache_batch(batch, encodings={"v1_enc": v1_encoded, "v2_enc": v2_encoded}, tags={"split": "val"})
        _, log = self.compute_losses(v1_encoded=v1_encoded,
                                     v2_encoded=v2_encoded,
                                     log_prefix="val")
        return log  # val loss is contained in log, we actually dont log here because we only want the averages

    def run_classifier_routine_on_cache(self, cached_encodings_keys=("v1_enc", "v2_enc")):
        """
        Calls run_classifier_routine_on_cache with argument cached_encodings_keys. Allows derived classes to
        overwrite the encoding keys.
        :param cached_encodings_keys: list-like of encoding keys in the EncodingsCache to retrieve for classification.
        :return: Classifier results
        """
        return run_classifier_routine_on_cache(
            cache=self.cache,
            cached_encodings_keys=cached_encodings_keys,
            csv_dataset=self.csv_dataset,
            current_epoch=self.current_epoch,
            classifier_train_on_additional_features=self.hparams.classifier_train_on_additional_features,
            weeks=[0, 1]  # consider alternative to hard-coding
        )

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
        if self.hparams.classifier_train_every and not (self.current_epoch %
                                                        self.hparams.classifier_train_every):
            classifier_results = self.run_classifier_routine_on_cache()
            val_outputs["log"].update(classifier_results)
        return val_outputs

    def clear_cache(self):
        """ Calls clear_cache on the stored EncodingsCache """
        logging.info("Clearing cache of {} elements".format(len(self.cache)))
        self.cache.clear_cache()

    def get_image_transforms(self):
        """
        Init image transforms.

        :return: callable, callable
                 Two callable transforms to apply to training- and validation images respectively
        """
        slice_cropper = SliceCropper(*self.hparams.input_slice_slice_range)
        train_image_transform = [slice_cropper, RandomCrop3d(self.hparams.input_image_size)]
        val_image_transform = [slice_cropper, CenterCrop3d(self.hparams.input_image_size)]
        if self.hparams.add_slice_distance_transform:
            dist_transform = SliceDistanceTransform(slice_shape=[320, 320])  # TODO: Fix hard-coding
            train_image_transform.insert(1, dist_transform)
            val_image_transform.insert(1, dist_transform)
        if not self.hparams.no_patch_augmentation:
            # Slice only inputs
            patch_transform = get_default_slice_patch_transforms()
        else:
            patch_transform = None
        # Get MEDTransformer objects for training and validation subsets
        train_transform, val_transform = get_train_and_val_transforms(self.patchifier,
                                                                      train_image_transform=Compose(train_image_transform),
                                                                      train_patch_transform=patch_transform,
                                                                      val_image_transform=Compose(val_image_transform),
                                                                      val_patch_transform=None,
                                                                      normalization_level=self.hparams.normalization_level)
        return train_transform, val_transform

    def get_segmentation_transforms(self):
        """
        Init segmentation transforms.

        :return: callable, callable
                 Two callable transforms to apply to training- and validation segmentations respectively
        """
        # Get image crop transforms and one-hot encoder
        slice_cropper = SliceCropper(*self.hparams.input_slice_slice_range)
        train_image_transform = Compose([slice_cropper, RandomCrop3d(self.hparams.input_image_size)])
        val_image_transform = Compose([slice_cropper, CenterCrop3d(self.hparams.input_image_size)])

        if not self.hparams.no_patch_augmentation:
            # Slice only inputs
            patch_transform = get_default_slice_patch_transforms(is_segmentation=True)
        else:
            patch_transform = None
        # Get MEDTransformer objects for training and validation subsets
        one_hot_encoder = OneHotEncode(num_classes=self.hparams.segmentation_encoder_input_channels, dim=4)
        train_transform, val_transform = get_train_and_val_transforms(self.patchifier,
                                                                      train_image_transform=train_image_transform,
                                                                      train_patch_transform=patch_transform,
                                                                      val_image_transform=val_image_transform,
                                                                      val_patch_transform=None,
                                                                      one_hot_encoder=one_hot_encoder,
                                                                      normalization_level=None)  # no norm
        return train_transform, val_transform

    def get_transforms(self):
        """
        TODO

        Defines the set of transforms to apply to the training- and validation datasets.

        :return:
        """
        if self.hparams.patch_strides[1] != 1 or self.hparams.patch_strides[2] != 1:
            raise NotImplementedError("DualViewCPC is only implemented for Slice level encodings.")
        im_train, im_val = self.get_image_transforms()
        seg_train, seg_val = self.get_segmentation_transforms()
        return {"image": im_train, "segmentation": seg_train}, \
               {"image": im_val, "segmentation": seg_val}

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
            load_segmentations=True,
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
                                             num_workers=1,
                                             random_sampling=False)
        logging.info("N training batches:   {}".format(len(self.train_loader)))
        logging.info("N validation batches: {}".format(len(self.val_loader)))

        # Set transforms on image and segmentation encoder objects
        self.get_encoder().image_encoder.set_transforms({"train": self.train_transform["image"],
                                                         "val": self.val_transform["image"]})
        self.get_encoder().segmentation_encoder.set_transforms({"train": self.train_transform["segmentation"],
                                                                "val": self.val_transform["segmentation"]})

    def configure_optimizers(self, additional_loss_parameters=None):
        """
        Configure the optimizer + AMP + encoder DataParallel
        """
        encoder_params = list(self.get_encoder().parameters())
        loss_params = list(self.loss.parameters())
        if additional_loss_parameters is not None:
            loss_params += list(additional_loss_parameters)
        # Experimental 10x multiplier on loss lr (magnitude of gradients is typically much lower for those modules)
        optimizer = torch.optim.Adam(params=encoder_params+loss_params, lr=self.hparams.learning_rate)

        # Prepare modules for AMP. We manually do this here due to a bug in PL (apex really) with AMP + DP.
        if self.hparams.amp_training:
            (self.encoder,), optimizer = amp.initialize([self.encoder], optimizer, opt_level="O1")
        self.encoder = torch.nn.DataParallel(self.encoder, device_ids=[0, 1, 2, 3], output_device=3)
        return optimizer

    def backward(self, trainer, loss, optimizer, optimizer_idx):
        """
        Overwrites the default backward method to perform AMP scaled loss even with PL was init with no GPUs.
        This is needed because we manually control the number of GPUs and device placement.
        """
        if self.hparams.amp_training:
            # Use scaled loss with AMP
            with amp.scale_loss(loss, optimizer) as scaled_loss:
                scaled_loss.backward()
        else:
            # Pass to default backward method
            super(MEDDualViewContrastivePredictiveCoding, self).backward(trainer, loss, optimizer, optimizer_idx)


def entry_func(args, logger):
    """
    Init the PL module with passed args and launch training.

    :param args: Namespace, hyperparameters to pass to the PL system
    :param logger: AMLTensorBoardLogger, logger to use for AML and TensorBoard logging
    """
    # Get module
    cpc = MEDDualViewContrastivePredictiveCoding(hparams=args)

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
                                                   cache_encoding_keys=["v1_enc", "v2_enc"],
                                                   weeks=[0, 1])],
                      min_epochs=args.num_epochs,
                      max_epochs=args.num_epochs)
    trainer.fit(cpc)
