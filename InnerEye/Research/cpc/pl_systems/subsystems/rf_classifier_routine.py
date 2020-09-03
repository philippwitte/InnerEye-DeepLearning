import logging
import numpy as np
import torch
from pprint import pformat
from pathlib import Path


def run_classifier_routine_on_cache(cache,
                                    cached_encodings_keys,
                                    csv_dataset,
                                    current_epoch,
                                    classifier_train_on_additional_features=None,
                                    weeks=None):
    """
    Run the 'run_classifier_routine' function on a passed set of encodings in cached_encodings or
    extract it from the cache.

    :param cached_encodings_keys:
    :return: Classifier results, dict of scalar matric values
    """

    raise NotImplementedError
