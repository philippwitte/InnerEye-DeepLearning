import numpy as np
import torch


class RandomParameterRange:
    """
    Defines a range of parameters in [lower, upper] from which
    values may be sampled linearly og log-linearly using .sample()
    """
    def __init__(self, lower, upper, log_scale_sampling=False):
        """
        TODO
        """
        self.lower = float(lower)
        self.upper = float(upper)
        self.log_scale_sampling = bool(log_scale_sampling)

    @staticmethod
    def check_list_length_2(input_, name="input"):
        if not isinstance(input_, (list, tuple, np.ndarray)) or len(input_) != 2:
            raise ValueError("{} must be a length 2 list of "
                             "values to samples from.".format(name))

    def sample(self):
        if self.log_scale_sampling:
            range_ = map(np.log10, (self.lower, self.upper))
            return np.power(10, np.random.uniform(*range_, 1)[0])
        else:
            return np.random.uniform(self.lower, self.upper, 1)[0]


def broadcast_to_n_dims(entry, n_dims):
    if isinstance(entry, (list, tuple, torch.Tensor, np.ndarray)):
        if len(entry) != n_dims:
            raise ValueError("Must specify a value for each dimension ({} dims) "
                             "when passing list. Pass a single value to broadcast a "
                             "value to all dimensions/channels.")
        return entry
    elif isinstance(entry, (float, int)):
        return [float(entry)]*n_dims
    else:
        raise ValueError("Did not understand input type {} (input={}), "
                         "expected list, tuple, Tensor, float or int.".format(type(entry), entry))


def get_as_2d_batch(image):
    """
    Takes a Torch.Tensor and maps it to ndim 4 as:
        Input shape    Output shape
             [H, W]    [1, 1, H, W]
          [N, H, W]    [N, C, H, W]
       [N, C, H, W]    [N, C, H, W]
    [N, C, 1, H, W]    [N, C, H, W]
    :param image:
    :return:
    """
    org_shape = image.shape
    if image.ndim == 2:
        # Assumes [H, W]
        image = image.unsqueeze(0).unsqueeze(0)
    elif image.ndim == 3:
        # Assumes [N, H, W]
        image = image.unsqueeze(1)
    elif image.ndim == 4:
        # Assumes already [N, C, H, W]
        pass
    elif image.ndim == 5:
        image = image.squeeze(2)
    if image.ndim == 4:
        return image, org_shape
    else:
        raise ValueError("Only supports input tensors of shape ndim 2, 3, 4 or 5 (if 5, must be [N, C, 1, H, W])")
