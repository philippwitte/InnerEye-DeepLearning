from torchvision.transforms import transforms
from InnerEye.Research.cpc.transforms.patch_transforms import (Patchifier,
                                                               ApplyToPatches,
                                                               NormalizePatches)


def get_eval_transforms(input_image_size,
                        patch_sizes,
                        patch_strides):
    """
    Returns a transforms.Compose object of transforms to use for STL10
    evaluation data.

    TODO

    Parameters
    ----------
    input_image_size
    patch_sizes
    patch_strides

    Returns
    -------

    """
    eval_transforms = [
        transforms.CenterCrop(size=input_image_size),
        transforms.Grayscale(),
        transforms.ToTensor(),
        Patchifier(patch_sizes=patch_sizes,
                   patch_strides=patch_strides),
        NormalizePatches(mean=[0.4120], std=[0.2570])
    ]
    return transforms.Compose(eval_transforms)


def _get_train_transforms(input_image_size,
                          patch_sizes,
                          patch_strides,
                          augment_patches=True):
    """
    Returns a list of transforms that is used during training on both
    unlabeled and labeled data.

    If augment_patches is True this also applies augmentation on individual
    patches (normally used only for CPC training on unlabelled data)

    TODO

    Parameters
    ----------
    input_image_size,
    patch_sizes,
    patch_strides,
    augment_patches

    Returns
    -------

    """
    list_of_transforms = [
        transforms.RandomCrop(size=input_image_size),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.Grayscale(),
        transforms.ToTensor(),
        Patchifier(patch_sizes=patch_sizes, patch_strides=patch_strides)
    ]
    if augment_patches:
        patch_transforms = ApplyToPatches(
            transforms.RandomApply(p=0.1, transforms=[
                transforms.RandomHorizontalFlip(1.0),
                transforms.RandomAffine(degrees=30,
                                        translate=(0.1, 0.1),
                                        scale=(0.9, 1.1),
                                        shear=5),
                transforms.ColorJitter(brightness=0.5,
                                       contrast=0.1,
                                       saturation=0.1,
                                       hue=0.1)
            ])
        )
        list_of_transforms.append(patch_transforms)
    normalizer = NormalizePatches(mean=[0.4120], std=[0.2570])
    return transforms.Compose(list_of_transforms + [normalizer])


def get_unlabelled_transforms(input_image_size,
                              patch_sizes,
                              patch_strides,
                              augment_patches):
    """
    Returns a transforms.Compose object of transforms to use for STL10
    unlabelled training data.

    If augment_patches is True:
        Apart from the default augmentation applied on the whole images (see
        _get_train_transforms) this also applies augmentation on
        individual patches.

    TODO

    Parameters
    ----------
    input_image_size
    patch_sizes
    patch_strides
    augment_patches

    Returns
    -------

    """
    return _get_train_transforms(input_image_size,
                                 patch_sizes,
                                 patch_strides,
                                 augment_patches)


def get_training_transforms(input_image_size,
                            patch_sizes,
                            patch_strides):
    """
    Returns a transforms.Compose object of transforms to use for STL10
    labeled training data.

    Applies augmentation only on the whole images, not on patches.

    TODO

    Parameters
    ----------
    input_image_size
    patch_sizes
    patch_strides

    Returns
    -------

    """
    return _get_train_transforms(input_image_size,
                                 patch_sizes,
                                 patch_strides,
                                 augment_patches=False)
