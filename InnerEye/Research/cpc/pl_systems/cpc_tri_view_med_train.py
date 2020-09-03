"""
Defines a system for training an encoder to extract useful representations
from images in an unsupervised manner using Constrastive Predictive Coding.
"""

from pytorch_lightning import Trainer
from torchvision.transforms import Compose

from InnerEye.Research.cpc.utils.callbacks import (get_early_stopping_cb,
                                                   get_model_checkpoint_cb)
from InnerEye.Research.cpc.dataloaders.med import get_train_and_val_transforms
from InnerEye.Research.cpc.transforms.default_med_transforms import get_default_slice_patch_transforms
from InnerEye.Research.cpc.transforms.image_transforms import (RandomCrop3d, CenterCrop3d, OneHotEncode,
                                                               SliceCropper, SliceDistanceTransform)
from InnerEye.Research.cpc.pl_systems.cpc_dual_view_med_train import MEDDualViewContrastivePredictiveCoding
from InnerEye.Research.cpc.callbacks import PlotEncodingSpace


class MEDTriViewContrastivePredictiveCoding(MEDDualViewContrastivePredictiveCoding):
    """
    Defines a PL system for tri-view (MED images <-> MED segmentations and image <-> aug image) CPC.

    See InnerEye.Research.cpc.pl_systems.cpc_base.ContrastivePredictiveCoding
    """

    def __init__(self, hparams):
        """
        :param hparams: Namespace
            Namespace object storing all hyperparameters that this PL system accepts according to its argparser func
            'cpc_tri_view_med_argparser' at InnerEye.Research.cpc.argparsers
        """
        super(MEDTriViewContrastivePredictiveCoding, self).__init__(hparams)
        # Ensure enabled bidirectional loss computation
        self.loss.bidirectional = True
        # Do not return individual view losses (handled in self.compute_losses)
        self.loss.return_view_losses = False

    def configure_optimizers(self, additional_loss_parameters=None):
        """
        Overwrites the base configure_optimizers method in order to pass parameters from the image <-> aug(image)
        loss specified in this derived class to the optimizers.

        Note: This currently has no effect as the DualViewINfoNCELoss has no parameters, but is implemented in
              case the loss implementation changes in the future to include learnable parameters.

        :param additional_loss_parameters: Unused argument
        :return: A torch optimizer object
        """
        assert additional_loss_parameters is None, "Not implemented"
        return super().configure_optimizers()

    def forward(self, batch, transform_key, aug_transform_key):
        """
        Encoded all patches extracted from images and segmentations in a batch with and without augmentations applied.
        Thus, this method returns 4 sets of encodings.

        Currently we work without the time-dimension.
        Combine batch and time dimension.

        Returned encodings are of shape [N, T, C, g1, g2, g3] where g1, g2, g3 and the patch-grid dimensions.

        :param batch: dict, containing at least keys "images" and "segmentations". Images and segmentation should be
                            torch.Tensor objects of shape [N, T, d1, d2, d3]
        :param transform_key: str, a key indicating which transform object (stored in encoder) to apply to the input
                                   before encoding it, e.g. "train" or "val", which may alter augmentations etc.
        :param aug_transform_key: Same as transform_key for but the augmented images/segmentations.
        :return: torch.Tensor of images encoded (shape [N, T, C, g1, g2, g3]),
                 torch.Tensor of augmented images encoded (shape [N, T, C, g1, g2, g3]),
                 torch.Tensor of segmentations encoded (shape [N, T, C, g1, g2, g3]),
                 torch.Tensor of augmented segmentations encoded (shape [N, T, C, g1, g2, g3])
        """
        images, segmentations = batch["images"], batch["segmentations"]
        if images.ndim == 5:
            # Flatten time-dimension into batch dim
            images = images.view(-1, *images.shape[2:])
        if segmentations.ndim == 5:
            # Flatten time-dimension into batch dim
            segmentations = segmentations.view(-1, *segmentations.shape[2:])

        # Forward pass without augmentation
        im_enc, seg_enc = self.encoder(images=images,
                                       segmentations=segmentations,
                                       transform_key=transform_key)

        # Forward pass with augmentation
        im_aug_enc, seg_aug_enc = self.encoder(images=images,
                                               segmentations=segmentations,
                                               transform_key=aug_transform_key)

        return im_enc, im_aug_enc, seg_enc, seg_aug_enc

    def compute_losses(self, im_enc, im_aug_enc, seg_enc, seg_aug_enc, log_prefix):
        """
        Computes three view losses:
            V(images, augmented images) - SimCLR inspired
            V(segmentations, augmented images)
            V(images, augmented segmentations)

        Negatives are sampled from the second view, or a memory bank of image and segmentation encodings
        respectively across both augmented and non-augmented samples.

        :param im_enc:      torch.Tensor of shape [N, C, (d1, ...)] of encodings of images
        :param im_aug_enc:  torch.Tensor of shape [N, C, (d1, ...)] of encodings of augmented image
        :param seg_enc:     torch.Tensor of shape [N, C, (d1, ...)] of encodings of segmentations
        :param seg_aug_enc: torch.Tensor of shape [N, C, (d1, ...)] of encodings of augmented segmentations
        :param log_prefix:  str, prefix to metric names for logging (e. g. 'train' or 'val')
        :return:
        """
        if self.hparams.use_memory_bank:
            memory_bank = self.cache.get_encodings_from_cache(encoding_keys=["im_enc", "im_aug_enc", "seg_enc", "seg_aug_enc"],
                                                              flatten_time_dim=True,
                                                              pool_to_vector=False,
                                                              as_memory_bank=True)
        else:
            memory_bank = None

        # Compute individual view losses, with or without memory banks as per --use_memory_bank flag
        im_seg_loss = self.loss(im_enc, seg_enc, memory_bank, memory_bank)
        im_seg_aug_loss = self.loss(im_enc, seg_aug_enc, memory_bank, memory_bank)
        seg_im_aug_loss = self.loss(seg_enc, im_aug_enc, memory_bank, memory_bank)
        seg_seg_aug_loss = self.loss(seg_enc, seg_aug_enc, memory_bank, memory_bank)
        im_im_aug_loss = self.loss(im_enc, im_aug_enc, memory_bank, memory_bank)
        im_aug_seg_aug = self.loss(im_aug_enc, seg_aug_enc, memory_bank, memory_bank)

        # Combined loss
        loss = (im_seg_loss + im_seg_aug_loss + seg_im_aug_loss + seg_seg_aug_loss + im_im_aug_loss + im_aug_seg_aug)
        loss = loss / 6

        # Add each loss term and combined loss to logs
        log = {f"{log_prefix}_loss": loss,
               f"{log_prefix}_im_seg_loss": im_seg_loss,
               f"{log_prefix}_im_seg_aug_loss": im_seg_aug_loss,
               f"{log_prefix}_seg_im_aug_loss": seg_im_aug_loss,
               f"{log_prefix}_seg_seg_aug_loss": seg_seg_aug_loss,
               f"{log_prefix}_im_im_aug_loss": im_im_aug_loss,
               f"{log_prefix}_im_aug_seg_aug": im_aug_seg_aug}
        return loss, log

    def training_step(self, batch, batch_idx):
        """
        Performs 1 step of training on a batch.

        :param batch: dict, a batch as output by CSVDataset or H5Dataset, storing at least "images", "segmentations",
                            "subjects", "weeks" and "labels".
        :param batch_idx: int, the index of the batch
        :return: dict, a dictionary of "loss": loss tensor and "log": dict of metrics to log.
        """
        im_enc, im_aug_enc, seg_enc, seg_aug_enc = self.forward(batch,
                                                                transform_key="train",
                                                                aug_transform_key="train_aug")
        if self.hparams.classifier_train_every:
            self.cache_batch(batch,
                             encodings={"im_enc": im_enc, "im_aug_enc": im_aug_enc,
                                        "seg_enc": seg_enc, "seg_aug_enc": seg_aug_enc},
                             tags={"split": "train"})
        loss, log = self.compute_losses(im_enc, im_aug_enc, seg_enc, seg_aug_enc, log_prefix="train")
        return {"loss": loss, "log": log}

    def validation_step(self, batch, batch_idx):
        """
        Performs 1 step of validation on a batch.

        :param batch: dict, a batch as output by CSVDataset or H5Dataset, storing at least "images", "segmentations",
                            "subjects", "weeks" and "labels".
        :param batch_idx: int, the index of the batch
        :return: dict, a dictionary of "loss": loss tensor and "log": dict of metrics to log.
        """
        im_enc, im_aug_enc, seg_enc, seg_aug_enc = self.forward(batch,
                                                                transform_key="val",
                                                                aug_transform_key="val_aug")
        if self.hparams.classifier_train_every:
            self.cache_batch(batch,
                             encodings={"im_enc": im_enc, "im_aug_enc": im_aug_enc,
                                        "seg_enc": seg_enc, "seg_aug_enc": seg_aug_enc},
                             tags={"split": "val"})
        _, log = self.compute_losses(im_enc, im_aug_enc, seg_enc, seg_aug_enc, log_prefix="val")
        return log  # val loss is contained in log, we actually dont log here because we only want the averages

    def run_classifier_routine_on_cache(self, cached_encodings_keys=None):
        """
        Overwrite MEDDualViewContrastivePredictiveCoding implementation to pass correct encoding keys
        :param cached_encodings_keys: Unused argument
        :return: Classifier results
        """
        return super().run_classifier_routine_on_cache(cached_encodings_keys=("im_enc", "seg_enc"))

    def get_image_transforms(self):
        """ Init image transforms """
        slice_cropper = SliceCropper(*self.hparams.input_slice_slice_range)
        train_image_transforms = [slice_cropper, RandomCrop3d(self.hparams.input_image_size)]
        val_image_transforms = [slice_cropper, CenterCrop3d(self.hparams.input_image_size)]
        if self.hparams.add_slice_distance_transform:
            dist_transform = SliceDistanceTransform(slice_shape=[320, 320])  # TODO: Fix hard-coding
            train_image_transforms.insert(1, dist_transform)
            val_image_transforms.insert(1, dist_transform)

        # Get patch-wise augmentations. Note that we apply each transform with probability 1 and modify individually
        assert not self.hparams.no_patch_augmentation, "Cannot specify --no_patch_augmentation with this module."
        patch_transform = get_default_slice_patch_transforms(apply_prob=1.0,
                                                              flip_probability=0.5,
                                                              elastic_probability=0.80)

        # Get MEDTransformer objects for training and validation subsets, w. and w.o patch augmentations
        aug_train_transform, aug_val_transform = get_train_and_val_transforms(self.patchifier,
                                                                              train_image_transform=Compose(train_image_transforms),
                                                                              train_patch_transform=patch_transform,
                                                                              val_image_transform=Compose(val_image_transforms),
                                                                              val_patch_transform=patch_transform,
                                                                              normalization_level=self.hparams.normalization_level)
        train_transform, val_transform = get_train_and_val_transforms(self.patchifier,
                                                                      train_image_transform=Compose(train_image_transforms),
                                                                      train_patch_transform=None,
                                                                      val_image_transform=Compose(val_image_transforms),
                                                                      val_patch_transform=None,
                                                                      normalization_level=self.hparams.normalization_level)

        return train_transform, aug_train_transform, val_transform, aug_val_transform

    def get_segmentation_transforms(self):
        """ Init segmentation transforms """
        # Get image crop transforms and one-hot encoder
        slice_cropper = SliceCropper(*self.hparams.input_slice_slice_range)
        train_image_transform = Compose([slice_cropper, RandomCrop3d(self.hparams.input_image_size)])
        val_image_transform = Compose([slice_cropper, CenterCrop3d(self.hparams.input_image_size)])

        # Get patch-wise augmentations. Note that we apply each transform with probability 1 and modify individually
        assert not self.hparams.no_patch_augmentation, "Cannot specify --no_patch_augmentation with this module."
        patch_transform = get_default_slice_patch_transforms(apply_prob=1.0,
                                                              flip_probability=0.5,
                                                              elastic_probability=0.80,
                                                              is_segmentation=True)
        # Get MEDTransformer objects for training and validation subsets, w. and w.o patch augmentations
        one_hot_encoder = OneHotEncode(num_classes=self.hparams.segmentation_encoder_input_channels, dim=4)
        aug_train_transform, aug_val_transform = get_train_and_val_transforms(self.patchifier,
                                                                              train_image_transform=train_image_transform,
                                                                              train_patch_transform=patch_transform,
                                                                              val_image_transform=val_image_transform,
                                                                              val_patch_transform=patch_transform,
                                                                              one_hot_encoder=one_hot_encoder,
                                                                              normalization_level=None)  # no norm
        train_transform, val_transform = get_train_and_val_transforms(self.patchifier,
                                                                      train_image_transform=train_image_transform,
                                                                      train_patch_transform=None,
                                                                      val_image_transform=val_image_transform,
                                                                      val_patch_transform=None,
                                                                      one_hot_encoder=one_hot_encoder,
                                                                      normalization_level=None)  # no norm

        return train_transform, aug_train_transform, val_transform, aug_val_transform

    def get_transforms(self):
        """
        TODO

        Defines the set of transforms to apply to the training- and validation datasets.

        :return:
        """
        if self.hparams.patch_strides[1] != 1 or self.hparams.patch_strides[2] != 1:
            raise NotImplementedError("DualViewCPC is only implemented for Slice level encodings.")
        im_train, im_aug_train, im_val, im_aug_val = self.get_image_transforms()
        seg_train, seg_aug_train, seg_val, seg_aug_val = self.get_segmentation_transforms()
        train_transforms = {"image": im_train,
                            "image_aug": im_aug_train,
                            "segmentation": seg_train,
                            "segmentation_aug": seg_aug_train}
        val_transforms = {"image": im_val,
                          "image_aug": im_aug_val,
                          "segmentation": seg_val,
                          "segmentation_aug": seg_aug_val}
        return train_transforms, val_transforms

    def prepare_data(self, train_val_inds=None, train_split_fraction=None):
        """
        Overwrites the  dataloader of MEDDualViewContrastivePredictiveCoding to set proper transforms on the encoder.
        See MEDDualViewContrastivePredictiveCoding.prepare_data for details.
        """
        super().prepare_data(train_val_inds, train_split_fraction)
        # Overwrite transforms on image and segmentation encoder objects
        self.get_encoder().image_encoder.set_transforms({"train": self.train_transform["image"],
                                                         "train_aug": self.train_transform["image_aug"],
                                                         "val": self.val_transform["image"],
                                                         "val_aug": self.val_transform["image_aug"]})
        self.get_encoder().segmentation_encoder.set_transforms({"train": self.train_transform["segmentation"],
                                                                "train_aug": self.train_transform["segmentation_aug"],
                                                                "val": self.val_transform["segmentation"],
                                                                "val_aug": self.val_transform["segmentation_aug"]})


def entry_func(args, logger):
    """
    TODO
    :param args:
    :param logger:
    :return:
    """
    # Get module
    cpc = MEDTriViewContrastivePredictiveCoding(hparams=args)

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
                                                   cache_encoding_keys=["im_enc", "seg_enc"],
                                                   weeks=[0, 1])],
                      min_epochs=args.num_epochs,
                      max_epochs=args.num_epochs)
    trainer.fit(cpc)
