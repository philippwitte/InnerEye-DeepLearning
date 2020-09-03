"""
This code implements the InfoNCE loss as described in:
https://arxiv.org/abs/1807.03748
"""

import logging
import torch
import torch.nn as nn
from torch.nn.modules.loss import CrossEntropyLoss
from InnerEye.Research.cpc.models import ScoreModel
from InnerEye.Research.cpc.losses.utils import apply_unit_sphere_normalization


def extract_directions(directions):
    """
    Takes a string of space-separated directions 
    or a list of directions and returns a tuple of 
    directions.
    """
    if isinstance(directions, str):
        directions = directions.split()
    directions = tuple(map(lambda x: x.lower(), directions))
    valid_directions = "down", "up", "left", "right", "ahead", "backward"
    for direction in directions:
        if direction not in valid_directions:
            raise ValueError("Invalid direction {}, must be one of "
                             "{}".format(direction, valid_directions))
    return directions


def view_by_direction(x_enc, direction):
    """
    Takes a batch of data of shape [batch_size, encoding_size, H, W, (D)]
    and returns a view of the same data according to 'direction'. Supports
    2D and 3D patches.

    :param x_enc: torch.Tensor
        Tensor of encoded patches, shape [N, C, d1, d2, (d3)]
    :param direction: str
        A direction in [down, up, left, right, (ahead), (backward)]
    :returns: torch.Tensor
        Permuted and potentially flipped x_enc of same ndim and number of elements
    """
    if direction == "down":
        # Do nothing, CPC routine is implemented for 
        # top-to-bottom predictions.
        return x_enc
    elif direction == "up":
        # Predict bottom-to-top
        # Flip the vertical axis
        return x_enc.flip(2)
    elif direction == "left":
        # Predict right-to-left
        dimorder = (0, 1, 3, 2, 4)[:x_enc.ndim]
        return x_enc.permute(dimorder).flip(2)
    elif direction == "right":
        # Predict left-to-right
        dimorder = (0, 1, 3, 2, 4)[:x_enc.ndim]
        return x_enc.permute(dimorder)
    elif direction == "ahead":
        # Predict back-to-front, only valid for 3D patches
        dimorder = (0, 1, 4, 2, 3)
        return x_enc.permute(dimorder).flip(2)
    elif direction == "backward":
        # Predict front-to-back, only valid for 3D patches
        dimorder = (0, 1, 4, 2, 3)
        return x_enc.permute(dimorder)
    else:
        raise ValueError("Invalid direction '{}'".format(direction))


def broadcast_to_directions(int_or_list, directions):
    """
    Given an integer, a list containing a single integer, or a list of length
    num_directions, returns a dictionary of mapping direction: integer of
    len(directions) elements, repeating the integer as needed.
    """
    num_directions = len(directions)
    if not isinstance(int_or_list, (list, tuple)):
        int_or_list = [int_or_list]
    list_of_ints = list(map(int, int_or_list))
    if len(list_of_ints) == 1:
        list_of_ints = list_of_ints * num_directions
    elif len(list_of_ints) != num_directions:
        raise ValueError("Expected either a single integer, list of 1 integer"
                         " or list of {} integers (matching  {}), "
                         "but got {}".format(num_directions, directions,
                                             int_or_list))
    return {dir_: val for dir_, val in zip(directions, list_of_ints)}


class InfoNCELoss(nn.Module):
    """"
    This code implements the InfoNCE loss as described in:
    https://arxiv.org/abs/1807.03748

    Within the InnerEye.Research.cpc project this class defines the main loss for patch-based CPC experiments.

    Example:
    --------
    >>> x_enc = torch.randn(16, 256, 7, 7, 7)  # 7x7x7 patch-grid of encodings each of depth 256, batch size 16
    >>> loss = InfoNCELoss(in_channels=256,
    >>>                    negative_samples=128,
    >>>                    score_model_depth=2,  # non-linear projection function
    >>>                    k_prediction_steps=5, # predict 5 rows ahead
    >>>                    num_skip_steps=1,     # ... but skipping the first row
    >>>                    directions=["up", "down", "left", "right", "backward", "ahead"],  # all directions in 3D
    >>>                    apply_unit_sphere_normalization=True)
    >>> loss(x_enc)
    tensor(4.8600, grad_fn=<DivBackward0>)
    """
    def __init__(self,
                 in_channels,
                 negative_samples,
                 score_model_depth,
                 k_prediction_steps,
                 num_skip_steps,
                 directions,
                 apply_unit_sphere_normalization=False):
        """
        Initialize a patch-based CPC objective.

        Parameters
        ----------
        in_channels: int
            the number of feature maps in the encoded patches
        negative_samples: int
            the number of negative samples to contrast against
        score_model_depth: int
            the number of layers in the projection/scoring models
            A score_model_depth of 1 gives a single, linear layer. A score model of 2 defines a simple, non-linear
            model with intervening non-linearities. See InnerEye.Research.cpc.models.score_models
        k_prediction_steps: int, list of ints
            the number of steps ahead to predict in each direction
        num_skip_steps: int, list of ints
            the number of steps to skip in each direction
        directions: list of str
            list of directions in ("down", "up", "left", "right", "ahead", "backward") to predict along
            Note that down/up corresponds to the 1. spatial dimension, left/right the 2. and ahead/backward the 3.
        apply_unit_sphere_normalization: bool
            Normalize each feature to unit norm before loss computation
        """
        super().__init__()
        self.negative_samples = negative_samples
        self.directions = extract_directions(directions)
        self.k_pred_steps = broadcast_to_directions(k_prediction_steps,
                                                    directions)
        self.skip_steps = broadcast_to_directions(num_skip_steps,
                                                  directions)
        self.apply_unit_sphere_norm = bool(apply_unit_sphere_normalization)
        logging.info("CPC running with {} directions: {}".format(
            len(self.directions), self.directions
        ))
        logging.info("Pred steps: {}".format(self.k_pred_steps))
        logging.info("Skip steps: {}".format(self.skip_steps))

        # Get models for inferring density ratios in each directions
        self.score_models = nn.ModuleDict()
        for direction in self.directions:
            self.score_models[direction] = nn.ModuleList([
                ScoreModel(in_channels, n_layers=score_model_depth)
                for _ in range(self.k_pred_steps[direction])
            ])
        self.contrastive_loss = CrossEntropyLoss(reduction='mean')
        logging.info('InfoNCELoss parameters: {:.4f} million'.format(
            sum(s.numel() for s in self.parameters()) / 10 ** 6)
        )

    def _one_directional_cpc(self, x_enc, x_enc_aggr, score_models, num_skip_steps):
        """
        Performs the CPC routine on x_enc and x_enc_aggr with scoring  functions in list score_models.
        Inputs are expected to be oriented for top-to-bottom predictions.

        :param x_enc: torch.Tensor
            Set of encoded patches, example shape: 2D [16, 1024, 7, 7] ([N, C, g1, g2]), 3D [N, C, g1, g2, g3]
        :param x_enc_aggr: torch.Tensor
            Set of aggregated encoded patches, same shape as x_enc
        :param score_models: torch.nn.ModuleList
            A list of torch.nn.Module objects to apply to rows of encodings to predict encodings k+num_skip_steps
            steps down from the context where k is the index of that module within score_models.
        :param num_skip_steps: int
            Number of rows to skip from the context and down (e.g. due to overlapping patches).
        """
        # Sum into total_loss in each of the K step-ahead steps
        total_loss = 0.0
        batch_size, n_channels = x_enc.size()[:2]

        # For each time-step k
        for k in range(1, len(score_models) + 1):
            # Get rows of encodings below k + the skip length
            rows_below = x_enc[:, :, (k + num_skip_steps):]

            # Get the context (the row of encodings from which we predict)
            # The context may be aggregated with an autoregressive like model
            # or not depending on the input 'x_enc_aggr'.
            context = x_enc_aggr[:, :, :-(k + num_skip_steps)]

            # Reshape to flat views, [batch_size, n_channels, S]
            # S is W * H (* D if 3D inputs)
            # OBS: Reshaping order must be identical for rows_below and context
            rows_below = rows_below.reshape(batch_size, n_channels, -1)
            context = context.reshape(batch_size, n_channels, -1)

            # Predict density ratios for all rows below
            density_ratios = score_models[k - 1].forward(rows_below)

            # Reorder from [batch_size, C, S] to [S, batch_size, C]
            density_ratios = density_ratios.permute(2, 0, 1)

            # Get a flat view of all densities
            flat_densities = density_ratios.reshape(-1, density_ratios.shape[-1])
            n_densities = len(flat_densities)

            # For each density, sample self.negative_samples random inds to
            # contrast against
            rand_inds = torch.randint(
                high=n_densities,
                size=(n_densities * self.negative_samples,),
                dtype=torch.long,
                device=flat_densities.device,
            )
            contrastive_densities = flat_densities[rand_inds, :]

            # Reshape to look like 'density_ratios' with negative samples axis
            contrastive_densities = contrastive_densities.view(
                density_ratios.shape[0],
                density_ratios.shape[1],
                self.negative_samples,
                density_ratios.shape[2],
            ).permute(
                0, 1, 3, 2
            )  # [S, batch_size, C, self.negative_examples]

            # Permute context in preparation for matmul with densities
            # [batch_size, n_channels, S] to [S, batch_size, n_channels]
            context = context.permute(
                2, 0, 1
            )

            # Compute density ratios for positive and negative samples
            log_density_positive = torch.matmul(
                context.unsqueeze(-2),
                density_ratios.unsqueeze(-1)
            ).squeeze(-2)  # output shape: [S, batch_size, 1]
            log_density_negative = torch.matmul(
                context.unsqueeze(-2),
                contrastive_densities
            ).squeeze(-2)  # output shape: [S, batch_size, negative_samples]

            # Concatenate positive and negative densities
            log_density = torch.cat((log_density_positive,
                                     log_density_negative), dim=-1).permute(
                1, 2, 0
            )  # shape: [batch_size, negative_samples+1, S]

            # Construct target values for NLL computation
            # Index 0 is always the true class
            target = torch.zeros(
                size=(log_density.shape[0],
                      log_density.shape[-1]),
                dtype=torch.long,
                device=log_density.device
            )  # shape:  [batch_size, S]
            total_loss += self.contrastive_loss(
                input=log_density.to(torch.float32),
                target=target
            )
        return total_loss / len(score_models)

    def forward(self, x_enc, aggregators=None, v1_memory_bank=None, v2_memory_bank=None):
        """
        Takes a set of patched encodings 'x_enc' and computes patch-based CPC along all directions as specified in
        self.directions. If aggregators is not None, expects a mapping-like object with keys in self.directions
        mapping to a torch.nn.Module to apply to x_enc as seen from that direction.

        v1_memory_bank and v2_memory_bank arguments are not yet supported.

        :param x_enc: torch.Tensor
            Set of encoded patches, example shape: 2D [16, 1024, 7, 7] ([N, C, g1, g2]), 3D [N, C, g1, g2, g3]
        :param aggregators: mapping or None
            Must be a dict mapping a direction to a Aggregator model to apply to x_enc as seen from that direction,
            or None, to apply no aggregation.
        :returns: torch.Tensor
            The scalar loss
        """
        if v1_memory_bank is not None or v2_memory_bank is not None:
            raise NotImplementedError("Using memory banks is only implemented for multi-view InfoNCE.")
        if self.apply_unit_sphere_norm:
            x_enc = apply_unit_sphere_normalization(x_enc)

        total_loss = 0.0
        for direction in self.directions:
            # View the batch according to the direction
            x_enc_dir = view_by_direction(x_enc, direction)

            # If specified, aggregate the encoding with the passed
            # direction specific aggregation model
            if aggregators is not None:
                x_enc_aggr = aggregators[direction].forward(x_enc_dir)
            else:
                x_enc_aggr = x_enc_dir

            # Do actual CPC in the specified direction
            total_loss += self._one_directional_cpc(x_enc_dir, 
                                                    x_enc_aggr,
                                                    self.score_models[direction],
                                                    self.skip_steps[direction])
        return total_loss / len(self.directions)
