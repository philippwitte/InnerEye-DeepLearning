import torch
from torch import nn
import torch.nn.functional as F


class LinearClassifier(nn.Module):
    """
    A single linear layer model
    This only purpose of this module is to implement the pool_to_vec staticmethod
    TODO: Consider moving pool_to_vec to a function and remove this module
    """
    def __init__(self, in_channels, num_classes=10, bias=True):
        super().__init__()
        self.linear = nn.Linear(in_features=in_channels,
                                out_features=num_classes,
                                bias=bias)

    @staticmethod
    def pool_to_vector(x_enc):
        """
        Takes a tensor of ndim 2, 3, 4 or 5 and returns a adaptively mean pooled
        tensor of ndim 2 (shape [N, C]). This can be used to pool a grid of encodings
        [N, C, (d1), (d2), (d3)]  -> [N, C]
        """
        dtype = x_enc.dtype
        x_enc = x_enc.to(torch.float32)
        batch_size, channels = x_enc.size()[:2]
        if x_enc.ndim == 2:
            return x_enc
        elif x_enc.ndim == 3:
            # 1D
            x_enc = F.adaptive_avg_pool1d(x_enc, 1)
        elif x_enc.ndim == 4:
            # 2D patches
            x_enc = F.adaptive_avg_pool2d(x_enc, [1, 1])
        elif x_enc.ndim == 5:
            # 3D patches
            x_enc = F.adaptive_avg_pool3d(x_enc, [1, 1, 1])
        else:
            raise NotImplementedError("Only implemented for 1D, 2D and 3D "
                                      "patches. Got input with {} dimensions, "
                                      "expected 2, 3, 4 or 5.".format(x_enc.ndim))
        x_enc = x_enc.squeeze()
        if batch_size == 1:
            x_enc = x_enc.unsqueeze(0)
        if channels == 1:
            x_enc = x_enc.unsqueeze(-1)
        return x_enc.to(dtype)

    def forward(self, x_enc):
        x_enc = self.pool_to_vector(x_enc)
        return self.linear(x_enc)
