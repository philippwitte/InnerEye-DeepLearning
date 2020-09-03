import logging
import torch
import os


def get_device(use_gpu):
    """
    Return a single GPU device if use_gpu is True, otherwise return CPU device
    :param use_gpu: bool
    :return: torch.device
    """
    device = torch.device("cuda" if torch.cuda.is_available()
                                 and use_gpu else "cpu")
    logging.info("Using device: {}".format(device))
    return device


def default_gpu_distribution(encoder, loss, aggregators=None):
    """
    Distribute encoder, loss and optionally aggregators nn.Modules as used in typical CPC experiments to either CPU
    (if CUDA is not available) else to the first GPU for encoder and (optionally) aggregators and last GPU for loss.
    The encoder and/or aggregators are typically distributed across GPUs later using torch.nn.DataParallel.

    :param encoder: torch.nn.Module
    :param loss: torch.nn.Module
    :param aggregators: torch.nn.Module
    :return: torch.nn.Module, torch.nn.Module, torch.nn.Module or None
    """
    num_gpus = torch.cuda.device_count()
    if num_gpus == 0:
        first_device = torch.device("cpu")
        last_device = first_device
    else:
        first_device = torch.device(type="cuda", index=0)
        last_device = torch.device(type="cuda", index=num_gpus-1)
    encoder = encoder.to(first_device)
    loss = loss.to(last_device)
    if aggregators is not None:
        aggregators = aggregators.to(first_device)
    return encoder, loss, aggregators


def make_dirs(dirs, exist_ok=True):
    """ Small helper, creates all dirs in a list of (str) paths """
    for dir_ in dirs:
        os.makedirs(dir_, exist_ok=exist_ok)
