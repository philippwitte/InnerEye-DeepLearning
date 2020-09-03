"""
This code implements the InfoNCE loss for multi-view CPC described in:
https://arxiv.org/pdf/1906.05849.pdf

Extends the implementation to allow sampling nearest negatives, negatives from memory banks and excluding negatives
based on tempo-spatial distance to the reference within the patch-grid.
"""

import logging
import torch
import torch.nn as nn
from itertools import chain
from torch.nn.modules.loss import CrossEntropyLoss


class DualViewInfoNCELoss(nn.Module):
    """"
    This code implements the InfoNCE loss for 2-view CPC described in:
    https://arxiv.org/pdf/1906.05849.pdf
    This implementation extends the loss further by allowing various negative sampling modes, negative exclusions with
    distance etc.

    Example:
    --------
    >>> a = torch.randn(8, 5, 64, 35, 1, 1)  # 8 samples, 5 time-steps, 64 dim encodings, 35x1x1 patch-grid
    >>> b = torch.randn(8, 5, 64, 35, 1, 1)
    >>> loss = DualViewInfoNCELoss(negative_samples=512, tau=1, bidirectional=True, negative_sampling_mode="random",
    >>>                            negatives_exclude_weeks_within_dist=2, negatives_exclude_spatial_within_dist=2)
    >>> loss(a, b)
    (tensor(6.2471), tensor(6.2472), tensor(6.2471))  # mean loss, L(V1, V2), L(V2, V1)
    """
    def __init__(self,
                 negative_samples,
                 bidirectional=True,
                 return_view_losses=True,
                 tau=1.0,
                 keep_excluded_with_tau=False,
                 negative_sampling_mode="random",
                 negatives_exclude_weeks_within_dist=0,
                 negatives_exclude_spatial_within_dist=0):
        """
        :param negative_samples: int
            the number of negative samples to contrast against
        :param bidirectional: bool
            Given views V1 and V2 compute both L(V1, V2) and L(V2, V1) for combined
            L = L(V1, V2) + L(V2, V1)
        :param return_view_losses: bool
            Return the mean loss + L(V1, V2) + L(V2, V1) else only the mean loss
        :param tau: float
            Temperature scaling coefficient. Cosine similarities are scaled by a 'dynamic range' parameter 1/tau.
        :param keep_excluded_with_tau: False, float
            If False, exclude negatives according to negatives_exclude_weeks_within_dist and
            negatives_exclude_spatial_within_dist. Otherwise, keep those negatives but assign a differentiated tau value
            to those negative pairs.
        :param negative_sampling_mode: str in ("random", "nearest")
            Sample negatives randomly or select the 'negative_samples' nearest negatives to an anchor.
        :param negatives_exclude_weeks_within_dist: int
            Exclude patches that are <= negatives_exclude_weeks_within_dist weeks away from the reference.
            E.g. with 0 exclude same week, with 1 exclude same, previous and next week (if available).
            Set a negative value to exclude nothing.
        :param negatives_exclude_spatial_within_dist: int
            Same as negatives_exclude_weeks_within_dist but for spatial distance along all patch grid dimensions
        """
        assert negative_sampling_mode in ("random", "nearest"), "negative_sampling_mode must be 'random' or 'nearest'"
        super().__init__()
        self.negative_samples = negative_samples
        self.dynamic_range = torch.FloatTensor([1/float(tau)])
        self.dynamic_range_excluded = torch.FloatTensor([1/float(keep_excluded_with_tau)]) if keep_excluded_with_tau else False
        self.bidirectional = bool(bidirectional)
        self.return_view_losses = bool(return_view_losses)
        self.negatives_exclude_weeks_max_dist = int(negatives_exclude_weeks_within_dist)
        self.negatives_exclude_spatial_max_dist = int(negatives_exclude_spatial_within_dist)
        self.negative_sampling_mode = str(negative_sampling_mode)
        self.contrastive_loss = CrossEntropyLoss(reduction='mean')
        logging.info('InfoNCELoss parameters: {:.4f} million'.format(
            sum(s.numel() for s in self.parameters()) / 10 ** 6)
        )
        logging.info('Excluding negatives if week distance to anchor <= {}'.format(self.negatives_exclude_weeks_max_dist))
        logging.info('Excluding negatives if spatial distance to anchor <= {}'.format(self.negatives_exclude_spatial_max_dist))
        logging.info('Keeping excluded negatives but with separate tau: {}'.format(keep_excluded_with_tau))

    def _compute_pair_scores(self, v1_enc, v2_enc):
        """
        Computes h({v1, v2}) = \exp(\frac{v1_enc * v2_enc}{norm(v1_enc) * norm(v2_enc)} * 1/tau)
        :param v1_enc: torch.Tensor, shape [N, C]
        :param v2_enc: torch.Tensor, shape [N, C]
        :return: torch.Tensor, pair scores, shape [N]
        """
        assert v1_enc.ndim == v2_enc.ndim == 2, "Expected tensor of ndim=3, shape " \
                                                "[N, C], got {}".format(v1_enc.shape)
        cosine_similarities = torch.nn.functional.cosine_similarity(v1_enc, v2_enc, dim=1)
        return self._compute_pair_scores_from_cosine_similarities(cosine_similarities)

    def _compute_pair_scores_from_cosine_similarities(self, cosine_similarities, dynamic_ranges=None):
        """
        :param cosine_similarities: torch.Tensor, shape [N]
        :param dynamic_ranges: torch.Tensor, shape [N]
        :return: torch.Tensor, cosine similarities, shape [N]
        """
        ranges_ = dynamic_ranges if dynamic_ranges is not None else self.dynamic_range
        return torch.exp(cosine_similarities * ranges_.to(cosine_similarities))

    def _make_negatives_generator(self,
                                  num_negatives,
                                  v1,
                                  take_negatives_from,
                                  negative_sampling_mode,
                                  exclude_mask=None,
                                  dynamic_ranges=None):
        """
        For each encoding in v1, sample a negative from take_negatives_from and yield the pair-scores.
        The generator yields num_negatives times.

        OBS: Might yield fewer times if a number of negatives for any anchor is less than 'num_negative' after
        application of the exclusion mask.

        :param num_negatives: int, a number of times to yield 1 negative per sample in v1
        :param v1: torch.Tensor, shape [N, C], Reference/anchor encodings
        :param take_negatives_from: torch.Tensor, shape [-1, C]
        :param negative_sampling_mode: str in ['nearest', 'random'], take a random or next nearest negative to anchor
        :param exclude_mask:   torch.Tensor of shape [len(take_negatives_from), len(take_negatives_from)] and dtype
                               torch.bool masking out elements to consider as valid negative samples.
        :param dynamic_ranges: None or torch.Tensor of shape [len(take_negatives_from), len(vtake_negatives_from)]
                               and dtype double giving pair-specific dynamic range parameters (1/tau).
                               If None, self.dynamic_range is used for all pairs.
        :return: Generator which yields torch.Tensors of shape [len(v1)] of pair scores 'num_negative' times.
                 OBS: Might yield fewer times if a number of negatives for any anchor is less than 'num_negative' after
                 application of the exclusion mask.
        """
        if num_negatives == 0:
            yield from ()
        else:
            negatives = take_negatives_from.to(v1.device)
            if num_negatives >= len(negatives):
                raise ValueError("Too many negatives {} for negative tensor of length {}".format(
                    num_negatives, len(negatives)
                ))
            # Pre-compute full NxN cosine similarity matrix
            # TODO: Memory consuming (and slow) implementation, consider fix
            distance_matrix = torch.nn.functional.cosine_similarity(
                v1.unsqueeze(1),
                take_negatives_from.unsqueeze(0),
                dim=-1
            )
            dynamic_ranges = dynamic_ranges if dynamic_ranges is not None else \
                torch.empty_like(distance_matrix).fill_(self.dynamic_range)
            # Keep only valid negatives
            keep_masks = torch.logical_not(exclude_mask)
            valid_negatives = [(distances[valid], ranges[valid])
                               for distances, ranges, valid in zip(distance_matrix, dynamic_ranges, keep_masks)]
            n_available_negatives_per_positive = torch.LongTensor(list(map(lambda x: len(x[0]), valid_negatives)))
            min_available_negatives = torch.min(n_available_negatives_per_positive)
            # Take only the maximally available number of negatives from any sample for efficiency reasons
            # TODO: Consider if this approach may be optimized to not discard any negatives
            num_negatives = min(min_available_negatives, num_negatives)

            # Sort and select negatives
            selected_valid_negatives, selected_valid_dynamic_ranges = [], []
            for valid_negative, valid_dynamic_range in valid_negatives:
                selected_inds = torch.argsort(valid_negative, dim=0, descending=True)[:min_available_negatives]
                selected_valid_negatives.append(valid_negative[selected_inds])
                selected_valid_dynamic_ranges.append(valid_dynamic_range[selected_inds])
            selected_valid_negatives = torch.stack(selected_valid_negatives)
            selected_valid_dynamic_ranges = torch.stack(selected_valid_dynamic_ranges)

            for i in range(num_negatives):
                if negative_sampling_mode == "nearest":
                    similarities = selected_valid_negatives[:, i]
                    ranges = selected_valid_dynamic_ranges[:, i]
                elif negative_sampling_mode == "random":
                    rand_inds = torch.randint(high=selected_valid_negatives.shape[1],
                                              size=[len(v1), 1],
                                              dtype=torch.long,
                                              device=negatives.device)
                    similarities = torch.gather(selected_valid_negatives, dim=1, index=rand_inds).squeeze()
                    ranges = torch.gather(selected_valid_dynamic_ranges, dim=1, index=rand_inds).squeeze()
                else:
                    raise NotImplementedError("This should not happen")
                yield self._compute_pair_scores_from_cosine_similarities(similarities, dynamic_ranges=ranges)

    @staticmethod
    def _input_as_vectors(input):
        """
        Reshapes an input:
        [N, T, C, ...] -> [-1, C]
        Also returns the shape of the input excluding the channel dimension.
        :returns: torch.Tensor of shape [-1, C], torch.Size of the input excluding the channel dimension.
        """
        dim_order = [0, 1] + list(range(3, input.ndim)) + [2]
        input_reordered = input.permute(*dim_order)
        return input_reordered.reshape(-1, input.size(2)), input_reordered.shape[:-1]

    def get_excludes_mask_from_spatio_temporal_shape(self,
                                                     spatio_temporal_shape,
                                                     max_week_dist=0,
                                                     max_spatial_dist=0,
                                                     device=None):
        """
        Takes a torch.Size as output by self._input_as_vectors (2nd output) over an input of shape [N, T, C, ...]
        and builds a prod(*spatio_temporal_shape) times prod(*spatio_temporal_shape) exclude mask based on the distance
        from each entry to all other entries as per max_week_dist and max_spatial_dist.

        :param spatio_temporal_shape: torch.Size or list, the shape of the input positive encoding array excluding the
                                      channel dimension, e.g. as output by self._input_as_vectors.
        :param max_week_dist: int, max distance between a reference to another point along the 'week' (T) dim
        :param max_spatial_dist: int, max distance between a reference to another point along any spatial grid dim
        :param device: torch.device to place new tensors on
        :return: torch.Tensor boolean of shape [prod(*spatio_temporal_shape), prod(*spatio_temporal_shape)]
        """
        # Create meshgrid over the spatial dimensions
        meshgrid = torch.meshgrid(*list(map(lambda x: torch.arange(x, device=device or "cpu"), spatio_temporal_shape)))
        # Reshape the meshgrid to vector representation
        # OBS: Input encodings must be similarly reshaped to vectors so that indicies match
        origin_spec, _ = self._input_as_vectors(torch.stack(meshgrid, dim=2))
        # Compute if each entry in the origin_spec is too far away as per max_week_dist and max_spatial_dist from the
        # remaining entries
        n_spatial_dims = origin_spec.shape[-1] - 2
        max_diff = torch.LongTensor([0, max_week_dist] + [max_spatial_dist]*n_spatial_dims).to(origin_spec.device)
        return (torch.abs(origin_spec.unsqueeze(0)-origin_spec.unsqueeze(1)) <= max_diff).all(-1)

    def _2_view_cpc(self, v1_enc, v2_enc, memory_bank=None):
        """
        Computes contrastive loss L(V1, V2), optionally sampling negatives from a memory_bank in addition to V2

        :param v1_enc: torch.Tensor of shape [N, T, C, d1, d2, (d3)]
            Encoded patches from view 'v1', d* are patch-grid dimensions
        :param v2_enc: torch.Tensor of shape [N, T, C, d1, d2, (d3)]
            Encoded patches from view 'v2', d* are patch-grid dimensions
        :param memory_bank: torch.Tensor of shape [-1, T, C, d1, d2, (d3)]
        :returns: torch.Tensor, scalar loss
        """
        # Reshape inputs from [N, T, C, ...] -> [X, C]
        v1_enc, spatio_temporal_shape = self._input_as_vectors(v1_enc)
        v2_enc, _ = self._input_as_vectors(v2_enc)
        if memory_bank is not None:
            memory_bank = self._input_as_vectors(memory_bank)

        # Get [len(v1_enc), len(v1_enc)] mask of booleans indicating whether pairs are to be consider 'invalid' i.e.
        # too close to each other in time or space as per self.negatives_min_week_distance and
        # self.negatives_min_spatial_distance
        negatives_to_exclude = self.get_excludes_mask_from_spatio_temporal_shape(
            spatio_temporal_shape=spatio_temporal_shape,
            max_week_dist=self.negatives_exclude_weeks_max_dist,
            max_spatial_dist=self.negatives_exclude_spatial_max_dist,
            device=v1_enc.device
        )
        # Assign a dynamic range score (1/tau) to each pair as per self.dynamic_range and self.dynamic_range_excluded
        range_incl = self.dynamic_range.to(negatives_to_exclude.device)
        range_excl = self.dynamic_range_excluded.to(range_incl) if self.dynamic_range_excluded else range_incl
        dynamic_ranges = torch.where(negatives_to_exclude, range_excl, range_incl)
        if self.dynamic_range_excluded:
            # Do not actually exclude non-diagonal pairs as we assigned a different margin to those samples instead
            negatives_to_exclude = torch.eye(n=len(negatives_to_exclude),
                                             device=negatives_to_exclude.device,
                                             dtype=torch.bool)

        # Compute number batch negatives, memory bank negatives and sub-patch negatives
        n_batch_negatives = min(len(v2_enc)-1, self.negative_samples if bool(self.negative_samples) else len(v2_enc))
        n_mem_bank_negatives = 0 if memory_bank is None else n_batch_negatives  # TODO: Fix hard-coding
        n_total_negatives = n_batch_negatives + n_mem_bank_negatives

        # Prepare Tensor to store scores
        all_scores = torch.empty(size=[v1_enc.shape[0], n_total_negatives+1],  # +1 for positive pair
                                 dtype=v1_enc.dtype,
                                 device=v1_enc.device)

        # Compute positive pair scores
        all_scores[:, 0] = self._compute_pair_scores(v1_enc, v2_enc)

        # Compute negative scores within the batch, memory bank (if passed) and sub-patches (if passed)
        batch_negatives = self._make_negatives_generator(n_batch_negatives,
                                                         v1=v1_enc,
                                                         take_negatives_from=v2_enc,
                                                         exclude_mask=negatives_to_exclude,
                                                         negative_sampling_mode=self.negative_sampling_mode,
                                                         dynamic_ranges=dynamic_ranges)
        mem_bank_negatives = self._make_negatives_generator(n_mem_bank_negatives,
                                                            v1=v1_enc,
                                                            take_negatives_from=memory_bank,
                                                            negative_sampling_mode="random")
        actual_num_negatives_sampled = 0
        for i, negative_scores in enumerate(chain(batch_negatives, mem_bank_negatives)):
            all_scores[:, i+1] = negative_scores
            actual_num_negatives_sampled += 1
        all_scores = all_scores[:, :actual_num_negatives_sampled+1]

        # Construct target values for NLL computation
        # Index 0 is always the true class
        target = torch.zeros(
            size=(all_scores.shape[0],),
            dtype=torch.long,
            device=all_scores.device
        )  # shape:  [batch_size, S]
        return self.contrastive_loss(
            input=all_scores,
            target=target
        )

    def forward(self, v1_enc, v2_enc, v1_memory_bank=None, v2_memory_bank=None):
        """
        Takes two torch.Tensor objects of encodings from view V1 and view V2.
        v1_enc and v2_enc must have the same exact shape.

        v1_enc and v2_enc must be time-shaped batches, i.e. [N, T, C, ...]
        If no time-dimension is needed, make sure to pass [N, 1, C, ...] tensors.

        Supports batches of nD grid inputs.

        v1_enc example shape: 2D [16, 5, 1024, 7, 7] ([batch_size, time-steps, channels, H, W])
                              3D [16, 5, 1024, 7, 7, 7]

        Memory banks should be shaped similar to their v1_enc & v2_enc counterparts, except for the batch dim
        which may differ, e.g. [800, 5, 1024, 7, 7]

        :param v1_enc: torch.Tensor of shape [N, T, C, d1, d2, (d3)]
            Encoded patches from view 'v1', d* are patch-grid dimensions
        :param v2_enc: torch.Tensor of shape [N, T, C, d1, d2, (d3)]
            Encoded patches from view 'v2', d* are patch-grid dimensions
        :param v1_memory_bank: A torch.Tensor from which negative samples are taken for computation of L(V2, V1).
                               Only used with self.bidirectional = True
        :param v2_memory_bank: A torch.Tensor from which negative samples are taken for computation of L(V1, V2).
        """
        v1_enc = v1_enc.to(torch.float32)
        v2_enc = v2_enc.to(torch.float32)
        if v1_memory_bank is not None:
            v1_memory_bank = v1_memory_bank.to(torch.float32)
        if v2_memory_bank is not None:
            v2_memory_bank = v2_memory_bank.to(torch.float32)
        v1_v2_loss = self._2_view_cpc(v1_enc, v2_enc, memory_bank=v2_memory_bank)
        if self.bidirectional:
            v2_v1_loss = self._2_view_cpc(v2_enc, v1_enc, memory_bank=v1_memory_bank)
            total_loss = v1_v2_loss + v2_v1_loss
            total_loss = total_loss / 2
        else:
            v2_v1_loss = 0
            total_loss = v1_v2_loss
        if self.return_view_losses:
            return total_loss, v1_v2_loss, v2_v1_loss
        else:
            return total_loss
