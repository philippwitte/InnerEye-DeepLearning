import torch
from InnerEye.Research.cpc.transforms.utils import broadcast_to_n_dims, get_as_2d_batch


class RandomFlip2d:
    """
    Implementation over torchvision.transforms.RandomAffine which accepts
    tensors instead of PIL Images and can be run on GPU.
    """
    def __init__(self, dims, flip_probability=0.5):
        """
        OBS: Note that dims should be relative to the 2d image of shape [W, H], even if the input is e. g. [N, C, W, H]

        :param dims: Int or list of ints of dimensions in the image of shape [W, H] to flip along
        :param flip_probability: Int or list of ints of float values in [0, 1] giving the probability
                                 of flipping along the axis specified in 'dims' at the same index.
        """
        self.dims = dims if isinstance(dims, list) else [dims]
        assert not any([d not in (0, 1) for d in dims]), "Can only flip along dim 0 and/or 1 for 2d images"
        self.flip_probability = broadcast_to_n_dims(flip_probability, len(self.dims))

    def __call__(self, image):
        """
        image (torch.Tensor): Image to be flipped along self.dims with probability self.flip_probability.

        Returns:
            Torch.Tensor: Transformed image, same shape as input
        """
        image, in_shape = get_as_2d_batch(image)
        for dim, prob in zip(self.dims, self.flip_probability):
            if torch.rand(1) <= prob:
                image = torch.flip(image, [dim+2])
        return image.view(in_shape)
