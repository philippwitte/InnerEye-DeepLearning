import numpy as np
import logging
from pytorch_lightning.callbacks import Callback
from pathlib import Path
from InnerEye.Research.cpc.utils.plotting import plot_encoding_space


class ClearEncodingsCache(Callback):
    """
    Calls clear_epoch() on the parent Lightning module
    Normally used with MEDContrastivePredictiveCoding that caches encodings for classifier training.
    """

    def on_epoch_end(self, trainer, pl_module):
        pl_module.clear_cache()


class PlotEncodingSpace(Callback):
    """
    On validation end:
        Plots 2d t-SNE representation plots of the encodings cached through training by a
        MEDContrastivePredictiveCoding module. If module does not have attribute 'cache',
        this callback has no effect.
    """

    def __init__(self, out_dir, cache_encoding_keys, weeks=None):
        """
        :param out_dir: Path to a folder (existing/non-existing) in which to store plots
        :param cache_encoding_keys: List, a list of one or more encoding keys to extract features for from the cache.
                                    See cpc.pl_systems.subsystems.caching -> EncodingsCache for details.
        :param weeks: (optional) list, list of week integers to consider, e.g. [0, 1]
        """
        self.out_dir = Path(out_dir)
        self.out_dir.mkdir(exist_ok=True, parents=True)
        self.encoding_keys = list(cache_encoding_keys)
        self.weeks = list(weeks) if weeks is not None else weeks

        # Default maps
        self.labels_map = np.vectorize({0: "Label 0", 1: "Label 1"}.get)

    def make_plot(self, features, epoch, week_inds=None, split=None, labels=None):
        out_path = self.out_dir / "encoding_space_epoch_{}.png".format(epoch)
        # Get encodings and other fields as numpy arrays
        if features.ndim == 3:
            if week_inds is not None:
                if max(week_inds) >= features.shape[1]:
                    logging.error("Cannot select week inds {} from features of shape {}. "
                                  "Skipping plotting...".format(week_inds, features.shape))
                    return
                features = features[:, week_inds]
                logging.info("Selecting features from week inds {} "
                             "for plotting. New shape: {}".format(week_inds,
                                                                  features.shape))
            features = features.reshape(len(features), -1)
        logging.info("Saving encodings space plot using features of shape {} -"
                     "OBS: For large arrays this may take long!".format(features.shape))
        if labels is not None and len(np.unique(labels)) == 2:
            labels = self.labels_map(labels)
        plot_encoding_space(
            encodings=features,
            color_by=labels.ravel(),
            marker_by=split.ravel(),
            title="2D (t-SNE) encoding space - Epoch {}".format(epoch),
            out_path=out_path
        )
        logging.info("Encoding space plot saved to path {}".format(str(out_path)))

    def on_validation_end(self, trainer, pl_module):
        cache = getattr(pl_module, "cache")
        if cache:
            subjects = cache.get_cached_subjects()
            cached_encodings, subjects = cache.get_encodings_from_cache(encoding_keys=self.encoding_keys,
                                                                        subjects=subjects,
                                                                        weeks=self.weeks,
                                                                        flatten_time_dim=False,
                                                                        pool_to_vector=True,
                                                                        as_memory_bank=False)
            logging.info("Creating encoding plot based on encoding keys {} for {} subjects, encodings: {}".format(
                self.encoding_keys, len(subjects), cached_encodings.shape
            ))
            self.make_plot(features=cached_encodings.detach().cpu().numpy(),
                           epoch=pl_module.current_epoch,
                           split=cache.get_from_cache("split", subjects)[0],
                           labels=cache.get_from_cache("labels", subjects)[0])
