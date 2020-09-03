import torch
import logging
import numpy as np
from collections import defaultdict
from sklearn.decomposition import PCA


def compute_average_metrics(outputs):
    """
    Takes a list of PL step dict. outputs (or a single dict) and returns a
    new dictionary of mean metric values computed over all the steps.
    The returned metrics are named as 'avg_{org_metric_name}'.

    Each entry in outputs (or the single entry 'outputs') is expected to
    either store a sub-dictionary under key 'log' itself storing all metrics to
    consider, or to store only scalar values, each to be considered a metric.

    Parameters
    ----------
    outputs : list of dicts or dict
        A single or list of metrics dictionaries

    Returns
    -------
    A dictionary of mean metric values
    """
    avg_metrics = defaultdict(list)
    if isinstance(outputs, dict):
        outputs = [outputs]
    for output in outputs:
        output = output.get('log') or output  # If no 'log' entry, all scalars
        for metric, value in output.items():
            avg_metrics[metric].append(value)
    return {
        "avg_" + metric: torch.stack(values).mean() for
        metric, values in avg_metrics.items()
    }


def concatenate_outputs(outputs):
    """
    TODO
    """
    grouped_metrics = defaultdict(list)
    if isinstance(outputs, dict):
        outputs = [outputs]
    for output in outputs:
        for metric, value in output.items():
            if isinstance(value, dict):
                raise NotImplementedError("All entries should be scalars")
            grouped_metrics[metric].append(value)
    return {
        name: torch.cat(values, dim=0) for
        name, values in grouped_metrics.items()
    }


def gather_metrics(metrics, split_key):
    """
    TODO
    """
    torch_metrics, sklearn_metrics = {}, {}
    for metric, (splits, type_, func) in metrics.items():
        if split_key in splits:
            if type_ == "torch":
                torch_metrics[metric] = func
            elif type_ == "sklearn":
                sklearn_metrics[metric] = func
            else:
                raise ValueError("Unknown metrics type {}".format(type_))
    return torch_metrics, sklearn_metrics


def normalized_numpy(predictions, targets):
    """
    TODO
    Input shapes: torch.Size([N, C]) torch.Size([N, 1])
    """
    assert predictions.shape[-1] == 2, "Only implemented for 2-class/binary problems"
    predictions = torch.softmax(predictions, dim=1)[:, 1]
    return predictions.detach().cpu().numpy(), targets.detach().cpu().numpy()


def compute_sklearn_metrics(sklearn_metrics, predictions, targets, log_prefix=""):
    """
    TODO
    """
    assert predictions.shape[-1] == 2, "Only implemented for 2-class/binary problems"
    predictions, targets = normalized_numpy(predictions, targets)
    return compute_metrics(sklearn_metrics, predictions, targets, log_prefix=log_prefix)


def compute_metrics(metrics, predictions, targets, log_prefix=""):
    """
    Takes a dictionary of {metric_name: metric_func_callable} of metrics,
    a prediction tensor and corresponding targets values tensor and
    computes metric_func_callable(predictions, targets) for all entries
    in 'metrics'

    Parameters
    ----------
    metrics : dict
        A dictionary of {metric_name: metric_func_callable} of metrics
    predictions : torch.Tensor
        A tensor of prediction values
    targets : torch.Tensor
        A tensor of target values
    log_prefix : str, optional
        A string to pre-pend to all metric names in the output dict

    Returns
    --------
    A dictionary of computed metrics, {[log_prefix_]metric_name: metric_val}
    """
    metrics_to_return = {}
    for metric_name, func in metrics.items():
        if log_prefix:
            metric_name = log_prefix.rstrip("_") + "_" + metric_name
        metric_val = func(predictions, targets)
        if isinstance(metric_val, torch.Tensor):
            metric_val = metric_val.detach().cpu().numpy()
        metrics_to_return[metric_name] = float(metric_val)
    return metrics_to_return


def pca_reduce_features(features, keep_n_components, pca_obj=None):
    """
    TODO
    :param features:
    :param keep_n_components:
    :param pca_obj:
    :return:
    """
    keep_n_components = keep_n_components or pca_obj.n_components
    if pca_obj is None and min(features.shape[0], features.shape[1]) < keep_n_components:
        logging.error("Cannot perform PCA (too few samples or features)")
    else:
        # Reduce dimensionality of features with PCA
        dtype, device = features.dtype, features.device
        features = features.detach().cpu().numpy()
        if pca_obj is None:
            logging.info("Performing PCA: Reducing encodings from dim {} -> {}".format(
                features.shape[-1], keep_n_components
            ))
            pca_obj = PCA(n_components=keep_n_components).fit(features)
        features = pca_obj.transform(features)
        features = torch.from_numpy(features).to(dtype).to(device)
    return features, pca_obj


def normalize_features(features, means=None, stds=None, dim=0):
    """
    TODO
    :param features:
    :param means:
    :param stds:
    :return:
    """
    features = features.to(torch.float32)
    if means is None:
        means = torch.mean(features, dim=dim, keepdim=True)
    if stds is None:
        stds = torch.std(features, dim=dim, keepdim=True)
    normed_features = (features - means) / stds
    num_mask = torch.isnan(normed_features) | torch.isinf(normed_features)
    if num_mask.any():
        logging.warning("{} feature entries are NaN or inf post normalization. "
                        "Setting those entries to 0.".format(num_mask.sum()))
    normed_features[num_mask] = 0.0
    return normed_features, means, stds


def stack_feature_sequences(features, sequence_length, not_nan_mask=None):
    """
    TODO
    """
    org_shape = features.shape
    features = torch.stack(list(map(torch.flatten, torch.split(features, sequence_length, dim=0))))
    if not_nan_mask is not None:
        features = features[not_nan_mask]
    # logging.info("Stacking features for every {} images in the batch. Features {} -> {}".format(
    #     sequence_length, org_shape, features.shape
    # ))
    return features


def get_random_train_val_split(num_inds, train_split_fraction=0.9):
    """
    TODO
    """
    inds = list(range(num_inds))
    np.random.shuffle(inds)
    train_split_ind = int(len(inds) * train_split_fraction)
    train_inds = inds[:train_split_ind]
    val_inds = inds[train_split_ind:]
    assert len(train_inds) + len(val_inds) == len(inds)
    return train_inds, val_inds
