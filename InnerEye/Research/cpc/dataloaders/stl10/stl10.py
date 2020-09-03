import torch
import torchvision
import numpy as np

# Import default augmentations
from InnerEye.Research.cpc.dataloaders.stl10 import (get_eval_transforms,
                                                     get_training_transforms,
                                                     get_unlabelled_transforms)


def create_train_val_samplers(dataset_size, 
                              train_data_fraction=1.0, 
                              validation_split=0.2):
    """
    TODO
    """
    # Creating data indices for training and validation splits:
    indices = list(range(dataset_size))
    split = int(np.floor(validation_split * dataset_size))
    np.random.shuffle(indices)
    train_indices, val_indices = indices[split:], indices[:split]

    # Potentially discard some training inds. according to 'train_data_fraction'
    if train_data_fraction <= 0 or train_data_fraction > 1:
        raise ValueError("0 < 'train_data_fraction' <= 1 constraint violated.")
    n_train_keep = int(train_data_fraction * len(train_indices))
    train_indices = train_indices[:n_train_keep]

    # Creating data samplers and loaders:
    train_sampler = torch.utils.data.sampler.SubsetRandomSampler(train_indices)
    valid_sampler = torch.utils.data.sampler.SubsetRandomSampler(val_indices)

    return train_sampler, valid_sampler


def get_unlabelled_data_loader(base_folder,
                               download_dataset,
                               batch_size,
                               input_image_size,
                               patch_sizes,
                               patch_strides,
                               num_workers,
                               augment_patches=True,
                               train_data_fraction=1.0):
    """
    TODO
    """
    # Fetch the unlabeled datasets, make train and validation splits
    unsupervised_dataset_train = torchvision.datasets.STL10(
        base_folder,
        split="unlabeled",
        transform=get_unlabelled_transforms(input_image_size,
                                            patch_sizes,
                                            patch_strides,
                                            augment_patches),
        download=download_dataset,
    )
    unsupervised_dataset_val = torchvision.datasets.STL10(
        base_folder,
        split="unlabeled",
        transform=get_eval_transforms(input_image_size,
                                      patch_sizes,
                                      patch_strides),
        download=False,
    )
    train_sampler, val_sampler = create_train_val_samplers(len(unsupervised_dataset_train),
                                                           train_data_fraction)

    # Get train and val data loaders
    unsupervised_loader_train = torch.utils.data.DataLoader(
        unsupervised_dataset_train,
        batch_size=batch_size,
        sampler=train_sampler,
        num_workers=num_workers,
    )
    unsupervised_loader_val = torch.utils.data.DataLoader(
        unsupervised_dataset_val,
        batch_size=batch_size,
        sampler=val_sampler,
        num_workers=num_workers,
    )
    return unsupervised_loader_train, unsupervised_loader_val


def get_training_data_loder(base_folder,
                            download_dataset,
                            batch_size,
                            num_workers,
                            input_image_size,
                            patch_sizes,
                            patch_strides,
                            train_data_fraction=1.0):
    """
    TODO
    """
    # Fetch the training and validation sets
    dataset_train = torchvision.datasets.STL10(
        base_folder,
        split="train",
        transform=get_training_transforms(input_image_size,
                                          patch_sizes,
                                          patch_strides),
        download=download_dataset,
    )
    dataset_val = torchvision.datasets.STL10(
        base_folder,
        split="train",
        transform=get_eval_transforms(input_image_size,
                                      patch_sizes,
                                      patch_strides),
        download=False,
    )
    train_sampler, val_sampler = create_train_val_samplers(len(dataset_train),
                                                           train_data_fraction)

    # Create training and validation split data loaders
    train_loader = torch.utils.data.DataLoader(
        dataset_train,
        batch_size=batch_size,
        sampler=train_sampler,
        num_workers=num_workers,
    )
    val_loader = torch.utils.data.DataLoader(
        dataset_val,
        batch_size=batch_size,
        sampler=val_sampler,
        num_workers=num_workers,
    )
    return train_loader, val_loader


def get_eval_data_loader(base_folder,
                         download_dataset,
                         batch_size,
                         input_image_size,
                         patch_sizes,
                         patch_strides,
                         num_workers):
    """
    TODO
    """
    # Fetch the testing set
    dataset_test = torchvision.datasets.STL10(
        base_folder,
        split="test",
        transform=get_eval_transforms(input_image_size,
                                      patch_sizes,
                                      patch_strides),
        download=download_dataset,
    )
    loader_test = torch.utils.data.DataLoader(
        dataset_test,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
    )
    return loader_test
