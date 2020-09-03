import logging
from torchvision.transforms import transforms
from torchio import transforms as iotransforms
from InnerEye.Research.cpc.transforms import (RandomFlip2d, RandomAffine2d, ElasticTransform2d,
                                              GainBiasTransform, GaussianBlur2d)
from InnerEye.Research.cpc.transforms.wrappers import (TensorToTorchIOFormat,
                                                       TorchIOFormatToTensor,
                                                       ApplyToAllSlices)
from InnerEye.Research.cpc.transforms.patch_transforms import ApplyToPatches


def get_default_slice_patch_transforms(is_segmentation=False, apply_prob=0.33,
                                        flip_probability=1.0, elastic_probability=0.5):
    """
    :param is_segmentation:        If True, indicates that the transform is to be applied to segmentation maps
    :param apply_prob:             The probability that each patch transform is applied to each patch.
    :param flip_probability:       The probability that the horizontal Slice axis is flipped when the flip transform
                                   is selected. Note that the actual probability is apply_prob * flip_probability.
    :param elastic_probability:    The probability that the random elastic deformations transform is applied
                                   when it's transform is selected. Note that the actual probability is
                                   apply_prob * elastic_probability.
    :return: A transform, Callable
    """
    # Compose a set of transforms that is to be applied to each patch
    interpolation = "bilinear" if not is_segmentation else "nearest"
    patch_transforms = [  # Each is applied with some probability
        RandomFlip2d(dims=[1], flip_probability=flip_probability),
        RandomAffine2d(degrees=[-25, 25], translate=[0.05, 0.05],
                       scale=[0.8, 1.2], shear=[0.025, 0.025],
                       interpolation=interpolation),
        # We apply elastic deformations half as often by wrapping it as follows:
        transforms.RandomApply(transforms=[ElasticTransform2d(alpha_range=[2, 5],
                                                              sigma_range=[15, 35],
                                                              interpolation=interpolation)], p=elastic_probability)
    ]
    if not is_segmentation:
        # OBS: Consider whether the input is normalized or not at this stage!
        patch_transforms.append(GainBiasTransform(gain_range=[0.33, 1.8], bias_range=[-0.1, 0.1], channel_index=0))
        patch_transforms.append(GaussianBlur2d(sigma_range=[0.1, 5]))
    patch_wise_transform = transforms.RandomApply(p=apply_prob, transforms=patch_transforms)
    # Wrap transforms to apply transform to each Slice in a stack
    patch_wise_transform = ApplyToAllSlices(patch_wise_transform)
    # Wrap the patch_wise_transforms in an ApplyToPatches object
    # This object iterates all patches and applies its transform
    patch_wise_transform = ApplyToPatches(transform=patch_wise_transform, convert_to_pil=False)
    logging.info("Patch-wise transforms:\n{}".format(patch_wise_transform))
    return patch_wise_transform


def get_default_patch_transforms():
    """
    TODO
    :return:
    """
    # Compose a set of transforms that is to be applied to each patch
    patch_wise_transform = transforms.Compose([
        TensorToTorchIOFormat(),
        transforms.RandomApply(p=0.33, transforms=[
            iotransforms.RandomFlip(axes=(0, 1, 2), flip_probability=1.0),
            iotransforms.RandomAffine(scales=(0.9, 1.1), degrees=(-30, 30)),
            iotransforms.RandomElasticDeformation(num_control_points=(5, 7, 8),
                                                  max_displacement=(2, 9, 11),
                                                  proportion_to_augment=0.33)
        ]),
        TorchIOFormatToTensor()
    ])
    # Wrap the patch_wise_transforms in an ApplyToPatches object
    # This object iterates all patches and applies its transform
    patch_wise_transform = ApplyToPatches(transform=patch_wise_transform,
                                          convert_to_pil=False)
    logging.info("Patch-wise transforms:\n{}".format(patch_wise_transform))
    return patch_wise_transform
