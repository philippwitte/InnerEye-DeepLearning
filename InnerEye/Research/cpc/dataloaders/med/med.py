import torch
import numpy as np
from torch.utils.data import DataLoader, Subset
from torch.utils.data.sampler import SubsetRandomSampler


class MEDTransformer:
    """
    A callable defining the full pre-processing pipeline for MED images or segmentations.

    MEDTransformer is a callable which takes a torch.Tensor batch of images or segmentations of shape [N, C, D1, D2, D3]
    where N is the number of images, C the number of channels (can be omitted, [N, D1, D2, D3]) and D* are spatial dims.

    D1 is considered the Z-axis along which image slices of shape [D2, D3] may be sampled. A typical input shape is:
        [5, (1), 32, 320, 320]

    See self.transform_images for additional details.
    """

    def __init__(self,
                 patchifier,
                 image_wise_transform=None,
                 patch_wise_transform=None,
                 one_hot_encoder=None,
                 normalization_level="Volume"):
        """
        :param patchifier: Patchifier, callable
            A callable that transforms an image of shape [C, D1, D2, D3] --> [g1, g2, g3, C, d1, d2, d3].
            See InnerEye.Research.cpc.transforms.patch_transforms.
        :param image_wise_transform: None or callable
            An optional callable that is to be applied to the input before patching.
            Is applied to the full batch of shape [N, C, D1, D2, D3].
        :param patch_wise_transform: None or callable
            An optional callable that is to be applied to a patched input.
            Is applied to the patching of a single input of shape [g1, g2, g3, C, d1, d2, d3]
        :param one_hot_encoder: None or callable
            An optional callable to transform segmentation map inputs to one-hot encoding format
            Is applied to the full patched batched of shape [N, g1, g2, g3, 1, d1, d2, d3], e.g. should transform dim 4
        :param normalization_level: None, False or a string in [Slice, Volume]
            If bool(normalization_level) must be a string in [Slice, Volume] indicating whether to compute the mean
            and std constants along the last 2 or last 3 spatial dimensions of the input [N, C, D1, D2, D3].
        """
        self.patchifier = patchifier
        self.image_transform = image_wise_transform
        self.patch_transform = patch_wise_transform
        self.apply_normalization = bool(normalization_level)
        if self.apply_normalization and normalization_level not in ("Slice", "Volume"):
            raise ValueError(f"Normalization level must be a string in [Slice, Volume], got {normalization_level}")
        self.normalize_slices_individually = True if normalization_level == "Slice" else False
        self.one_hot_encoder = one_hot_encoder
        if self.one_hot_encoder and self.apply_normalization:
            raise ValueError("Should not apply both one-hot encoding and normalization to inputs.")

    def transform_images(self, images):
        """
        Takes a torch.Tensor of shape [N, (C), D1, D2, D3] and outputs a normalized, patched and augmented tensor of
        shape [N, g1, g2, g3, C, p1, p2, p3].

        Handles the following:
        1) applies callable self.image_transform to the full batch of input images.
           self.image_transform may affect the spatial dimensions (crops etc.) but the output should still be of ndim 5
        2) Computes normalization constants if bool(normalization_level) is True (e.g. set to False to disable).
           Computes the mean and std along the last 3 dims with normalization_level = "Volume" and last 2 with "Slice".
        3) Loops over all elements along the first axis of the input (each of shape [C, D1, D2, D3]) and
            3a) applies self.patchifier (argument: patchifier) to each. The patchifier outputs a patched image of shape
                [g1, g2, g3, C, d1, d2, d3] where g* are grid dims and d* are spatial patch dims.
            3b) transforms patches according to self.patch_transform
        4) Applies normalization to all elements in the patched tensor if specified
        5) Applies self.one_hot_encoder if set (e.g. for segmentation map inputs)

        :param images: torch.Tensor of shape [N, C, D1, D2, D3] or [N, D1, D2, D3]
        :return: torch.Tensor, patched and augmented images, shape [N, g1, g2, g3, C, p1, p2, p3]
        """
        if images.ndim == 4:
            # Expand to channel dim
            images = torch.unsqueeze(images, dim=1)  # [N, D1, D2, D3] --> [N, 1, D1, D2, D3], e.g. (5, 1, 32, 320, 320)
        elif images.ndim != 5:
            raise ValueError("Only implemented for images of shape [N, C, D1, D2, D3], "
                             "got shape {}".format(images.shape))
        images = images.to(torch.float32)  # We do all normalization & augmentation in FP32, AMP will cast to FP16 later

        # Apply image-wise transform if specified
        if self.image_transform is not None:
            images = self.image_transform(images)

        if self.apply_normalization:
            # Normalize the images
            start = 3 if self.normalize_slices_individually else 2
            means = torch.mean(images, dim=tuple(range(start, images.ndim)), keepdim=True).permute(0, 2, 3, 4, 1)
            stds = torch.std(images, dim=tuple(range(start, images.ndim)), keepdim=True).permute(0, 2, 3, 4, 1)
            # OBS: we apply normalization below after transformations to keep zero-point background the lowest value
        else:
            means, stds = None, None

        # Check numerics
        err = "One or more images in the batch contains NaN or inf numbers."
        assert not (torch.isnan(images).any() or torch.isinf(images).any()), err
        # Process each image
        out_images = None
        for i, image in enumerate(images):  # 'image' is of shape [C, D1, D2, D3], e.g. [1, 32, 320, 320]
            patched_image = self.patchifier(image)  # 'patched_image' is of shape [g1, g2, g3, C, d1, d2, d3]
            if self.patch_transform is not None:
                # Transform the individual patches with a set of augmentations
                patched_image = self.patch_transform(patched_image)
            if out_images is None:
                # Init a new tensor to store the patched outputs
                size = [images.shape[0]] + list(patched_image.size())
                out_images = torch.empty(size=size, dtype=patched_image.dtype, device=patched_image.device)
            out_images[i] = patched_image

        if means is not None:
            # Apply normalization
            for i in range(out_images.ndim - means.ndim):
                # Expand to same number of dims
                means = means.unsqueeze(-1)
                stds = stds.unsqueeze(-1)
            out_images.sub_(means)
            out_images.div_(stds)
        if self.one_hot_encoder:
            # Inputs are segmentation maps
            out_images = self.one_hot_encoder(out_images)
        return out_images.detach()

    def __call__(self, images):
        """
        See self.transform_images for docstring.
        """
        with torch.no_grad():
            return self.transform_images(images)


def create_data_loader(dataset, batch_size, inds, num_workers=0, random_sampling=False):
    """
    """
    if random_sampling:
        return DataLoader(dataset,
                          batch_size=batch_size,
                          num_workers=num_workers,
                          sampler=SubsetRandomSampler(inds))
    else:
        return DataLoader(Subset(dataset, inds),
                          batch_size=batch_size,
                          num_workers=num_workers)


def get_h5_dataset():
    raise NotImplementedError


def get_csv_dataset():
    raise NotImplementedError


def get_train_and_val_h5_datasets():
    h5_dataset_train = get_h5_dataset()
    h5_dataset_val = get_h5_dataset()
    return h5_dataset_train, h5_dataset_val


def get_train_and_val_csv_datasets():
    csv_dataset_train = get_csv_dataset()
    csv_dataset_val = get_csv_dataset()
    return csv_dataset_train, csv_dataset_val


def get_train_and_val_transforms(patchifier,
                                 train_image_transform=None,
                                 train_patch_transform=None,
                                 val_image_transform=None,
                                 val_patch_transform=None,
                                 one_hot_encoder=None,
                                 normalization_level="Slice"):
    """
    TODO
    """
    # Init transformer objects for train and val subsets
    train_transform = MEDTransformer(patchifier,
                                     image_wise_transform=train_image_transform,
                                     patch_wise_transform=train_patch_transform,
                                     one_hot_encoder=one_hot_encoder,
                                     normalization_level=normalization_level)
    val_transform = MEDTransformer(patchifier,
                                   image_wise_transform=val_image_transform,
                                   patch_wise_transform=val_patch_transform,
                                   one_hot_encoder=one_hot_encoder,
                                   normalization_level=normalization_level)
    return train_transform, val_transform
