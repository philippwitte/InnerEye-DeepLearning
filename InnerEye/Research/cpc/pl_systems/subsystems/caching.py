import torch
import numpy as np
from collections import defaultdict
from InnerEye.Research.cpc.models.linear_classifier import LinearClassifier


# Define a dictionary type to distinguish caches added per-week from other caching using dicts
class WeeksCacheDict(dict): pass


class EncodingsCache:
    """
    Defines a dict-like structure for storing cached encodings of MED time-series through training. This class is used
    by MED CPC PL systems such as MEDDualViewContrastivePredictiveCoding to collect the encoded version of a full
    dataset through an epoch of training, and then passes the encodings and other information on to downstream
    classifiers or plotting functions as needed.

    The EncodingsCache uses a dictionary of the following mapping:

        self.cache[subject_id_str] = dict()

    The sub-dict for each subject may store various information depending on the inputs to self.cache_results, but
    a usual dict may look like:

    self.cache[subject_id_str] : {
        "image_encodings": type WeeksCacheDict,
        "segmentation_encodings": type WeeksCacheDict,
        "labels": type torch.Tensor, scalar,
        "non_image_features": type WeeksCacheDict,
        "split": "train"
    }

    The WeeksCacheDict is another dictionary storing values across weeks. Please refer to self._add_weeks_cache_dict
    and self._unpack_weeks_cache_dict for details.
    """

    def __init__(self):
        self.cache = defaultdict(dict)

    def __len__(self):
        """ Returns the number of subjects currently referenced in the cache """
        return len(self.cache)

    def clear_cache(self):
        """ Remove all entries, re-initializing the stored defaultdict """
        self.cache = defaultdict(dict)

    def get_cached_subjects(self):
        """
        Returns the IDs of subjects currently in the cache.
        Note: order is not guaranteed to be preserved at the cache is updated with new entries.

        :return: list
            Returns a list of string subject identifiers currently in the cache.
        """
        return list(self.cache.keys())

    @staticmethod
    def _add_weeks_cache_dict(cache_dict, key, values, weeks):
        """
        Takes a torch.Tensor 'values' of 'week-shape' [T, *] where T == len(weeks) and a list of ints of
        weeks that the entries along the first dim of 'values' correspond to. Initializes a WeeksCacheDict dictionary
        at 'cache_dict[key]' and adds all week:value mappings in zip(values, weeks) to it.

        This method is used to populate encodings across weeks for a given subject over multiple calls to
        _add_weeks_cache_dict. This may be useful if not all weeks for a given subject were encoded within a single
        batch.

        :param cache_dict: dict, a subject-specific dictionary to update
        :param key: str, key in 'cache_dict' that should point to WeeksCacheDict storing values
        :param values: torch.Tensor, shape [T, *]
        :param weeks: list of ints, length T
        """
        if key not in cache_dict:
            cache_dict[key] = WeeksCacheDict()
        for value, week in zip(values, weeks):
            cache_dict[key][int(week)] = value

    @staticmethod
    def _unpack_weeks_cache_dict(cache_dict, weeks=None):
        """
        Takes a WeeksCacheDict input storing values in a int(week) : torch.Tensor([*]) mapping and returns an ordered
        torch.Tensor of shape [T, *] where T == len(cache_dict) and where elements along the first dimension is ordered
        so that the values for the lowest numeric week value comes first.

        :param cache_dict: WeeksCacheDict
        :param weeks: (optional) list of ints to fetch from cache_dict (in that order) instead of all weeks ordered
                                 low --> high.
        :return: torch.Tensor of shape [T, *] where T is the number of weeks
                 Returns None if any week in 'weeks' were not found in the passed cache_dict.
        """
        if weeks is None:
            weeks = sorted(cache_dict)
        weeks = list(map(int, weeks))
        if all([week in cache_dict for week in weeks]):
            outs = [cache_dict[week] for week in sorted(weeks)]
            return torch.stack(outs)
        else:
            return None

    def cache_results(self, encodings, subjects, weeks, labels=None, non_image_features=None, tags=None):
        """
        Add a batch of data to the cache.

        :param encodings: dictionary of encoding_name : torch.Tensor, each tensor of shape [N, T, *]
            A set of encodings to store for N subjects over T weeks. T may be 1, but should always have an axis.
            Each encoding in 'encodings' will be added to the cache under the same key name.
        :param subjects: list or ndarray of strings, length N
            List of subject-IDs that each sample along the first axis of 'encodings' correspond to.
        :param weeks: torch.Tensor or ndarray, ints or floats, shape [N, T]
            Array indicating what week each encoding in 'encodings' comes from for the given subject.
        :param labels: (optional) torch.Tensor, shape [N]
            Optional tensor of labels for each subject entry to also store. Added to cache under key 'labels'.
        :param non_image_features: (optional) torch.Tensor, shape [N, T, *]
            Optional set of other features to store for each subject at each week. Stored under key "non_image_features"
        :param tags: (optional) dict
            Optional dictionary of additional tags to store for all passed subjects. E.g. a tags dict {"my_tag": 2} will
            add a key 'my_tag' to dicts in self.cache[subject_id] for all subjects pointing to the value 2. If the value
            in 'tags' for a given key is a list, tuple, ndarray or Tensor, it is expected to have
            length == len(subjects) and store a value to tag each subject with specifically under that key. E.g.
            {"my_tag": [1, 2, 3]} would tag the 3 subjects in 'subjects' with 1, 2 and 3 respectively under key 'my_tag'
        """
        if labels is not None and isinstance(labels, torch.Tensor):
            # Cast labels to numpy arrays
            labels = labels.detach().cpu().numpy()
        for i, subject in enumerate(subjects):
            for key, enc in encodings.items():
                self._add_weeks_cache_dict(
                    cache_dict=self.cache[subject],
                    key=key,
                    values=enc[i].detach(),
                    weeks=weeks[i]
                )
            if non_image_features is not None:
                self._add_weeks_cache_dict(
                    cache_dict=self.cache[subject],
                    key="non_image_features",
                    values=non_image_features[i].detach(),
                    weeks=weeks[i]
                )
            if labels is not None:
                self.cache[subject]["labels"] = labels[i]
            for tag_name, value in tags.items():
                value = value if not isinstance(value, (list, tuple, np.ndarray, torch.Tensor)) else value[i]
                self.cache[subject][tag_name] = value

    def get_from_cache(self, key, subjects=None, as_numpy_array=True, weeks=None):
        """
        Return elements from the cache stored under a key for a set of subjects.

        :param key: str
            The key pointing to the relevant item to fetch from the encoding under each subject. E.g. 'image_encodings'
            or 'labels'.
        :param subjects: (optional) list-like of strings
            An optional list of subject-ID strings to fetch values (in that order) from in the cache.
        :param as_numpy_array: bool
            Return values from the cache as numpy arrays, otherwise return as entries in a list
        :param weeks: (optional) list of ints
            Optional list of weeks integers to fetch encodings for from the cache, e.g. [0, 1]
        :return: Returns the encodings either as a list or numpy array and the corresponding list of subjects
        """
        if subjects is None:
            subjects = self.get_cached_subjects()
        try:
            result = [self.cache[subject][key] for subject in subjects]
            # Unpack weeks if needed
            result = [(self._unpack_weeks_cache_dict(res, weeks=weeks) if isinstance(res, WeeksCacheDict) else res)
                      for res in result]
            # Get map of complete sequences (not None entries)
            is_complete = list(map(lambda x: x is not None, result))
            result = [r for r, complete in zip(result, is_complete) if complete]
            subjects = [s for s, complete in zip(subjects, is_complete) if complete]
        except KeyError:
            # If any entry does not have the key, return None
            return None, None
        if as_numpy_array:
            return np.asarray(result), subjects
        else:
            return result, subjects

    def get_encodings_from_cache(self,
                                 encoding_keys,
                                 subjects=None,
                                 flatten_time_dim=False,
                                 pool_to_vector=True,
                                 as_memory_bank=False,
                                 weeks=None):
        """
        Wrapper around self.get_from_cache for encoding fetches specifially.
        This method fetches one or more set of encodings from the cache under keys in 'encoding_keys' for a given set
        of subjects and first converts each to torch.Tensors of shape [N, T, C, *] where N is the number of subjects and
        T the number of weeks. Each entry in the cache must have cached encodings for all weeks or all weeks in 'weeks'
        if specified. Each tensor of encodings is the reshaped and processed as:

            if flatten_time_dim is True
                --> flatten the [N, T, C, *] tensor into [N*T, C, *] so that the first dim stores encodings across
                    subjects and weeks together
            if pool_to_vector is True
                --> pool all spatial dims * in a tensor [N, T, C, *] using mean pooling, returning [N, T, C] or [N*T, C]
                    depending on flatten_time_dim

            if as_memory_bank is True
                Stack any per-patch encodings along dims * in a tensor [N, T, C, *] or [N, C, *] onto the first dim.
                E.g. an input [N, T, C, 2, 4] --> [N*2*4, T, C]. Note that if pool_to_vector is True, this has no effect
                as all patches are pooled into a single vector.

        Then, outputs:

            if as_memory_bank is False:
                a tensor of shape [N, (T), C_all, (*)] depending on parameters described above, and where C_all is a
                concatenation over all encodings fetched as per 'encoding_keys'.
            if as_memory_bank is True:
                a tensor of shape [N_all, (T), C, (*)] depending on parameters described above, and where N_all is a
                concatenation over all encodings fetched as per 'encoding_keys'.

        :param encoding_keys: list of strings
        :param subjects: list of strings
        :param flatten_time_dim: bool
        :param pool_to_vector: bool
        :param as_memory_bank: bool
        :param weeks: (optional) list of ints
        :return: torch.Tensor, subjects
        """
        if subjects is None:
            subjects = self.get_cached_subjects()
        stacked_encodings = []
        for enc_key in encoding_keys:
            encodings, subjects = self.get_from_cache(enc_key, subjects, as_numpy_array=False, weeks=weeks)
            try:
                encodings = torch.stack(encodings)
            except RuntimeError as e:
                raise RuntimeError("Could not stack encodings, most likely due to non-equal shapes. "
                                   "See above traceback. This may occur if the cache is accessed when not all "
                                   "subjects have the same number of weeks cached.") from e
            n_subjects, time_steps = encodings.shape[:2]
            if flatten_time_dim or pool_to_vector:
                # [N, T, C, d...] --> [N*T, C, d...]
                encodings = encodings.reshape(-1, *encodings.shape[2:])
            if pool_to_vector:
                # Pool all encodings across grid to vector, [N, C, d...] -> [N, C]
                encodings = LinearClassifier.pool_to_vector(encodings)
                if not flatten_time_dim:
                    # Reshape to time-view
                    encodings = encodings.reshape(n_subjects, time_steps, encodings.shape[-1])
            if as_memory_bank:
                start_dim = 2 if flatten_time_dim else 3
                encodings = encodings.reshape(*encodings.shape[:start_dim], -1)
                # Stack encodings from all slices together
                encodings = torch.cat(torch.split(encodings, 1, dim=start_dim), dim=0).squeeze()
            stacked_encodings.append(encodings)
        if as_memory_bank:
            return torch.cat(stacked_encodings, dim=0), subjects
        else:
            dim = (1 if flatten_time_dim else 2)
            return torch.cat(stacked_encodings, dim=dim), subjects
