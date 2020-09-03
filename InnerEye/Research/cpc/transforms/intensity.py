import torch
from InnerEye.Research.cpc.transforms.utils import RandomParameterRange
from scipy.ndimage.filters import _gaussian_kernel1d
from InnerEye.Research.cpc.transforms.utils import get_as_2d_batch


class GainBiasTransform:
    """
    Implementation over torchvision.transforms.RandomAffine which accepts
    tensors instead of PIL Images and can be run on GPU.
    """
    def __init__(self, gain_range, bias_range, channel_index=None):
        self.gain_param = RandomParameterRange(*gain_range, log_scale_sampling=True)
        self.bias_param = RandomParameterRange(*bias_range, log_scale_sampling=False)
        self.channel_index = channel_index

    def __call__(self, image):
        """
        image (torch.Tensor): Image to be flipped along its last dimension.

        Returns:
            Torch.Tensor: Transformed image, same shape as input
        """
        gain = self.gain_param.sample()
        bias_param = self.bias_param.sample()
        if self.channel_index is not None:
            # In-place update only the selected channel.
            # This does not affect autodiff as we detach the augmented images
            image[:, self.channel_index, ...] *= gain
            image[:, self.channel_index, ...] += bias_param
            return image
        else:
            return gain * image + bias_param


class GaussianBlur2d:
    """
    TODO
    """
    def __init__(self, sigma_range):
        RandomParameterRange.check_list_length_2(sigma_range, "sigma_range")
        self.sigma_range = RandomParameterRange(*sigma_range, log_scale_sampling=False)

    @staticmethod
    def get_kernel(in_channels, sigma):
        # Get 2d Gaussian kernel
        kernel = torch.from_numpy(_gaussian_kernel1d(sigma, 0, int(3 * sigma + 0.5)))
        kernel = torch.matmul(kernel.view(-1, 1), kernel.view(1, -1))
        kernel = kernel.view(1, 1, *kernel.size())
        kernel = kernel.repeat(in_channels, *[1] * (kernel.dim() - 1))

        # Prepare conv layer for smoothing in __call__
        smooth = torch.nn.Conv2d(in_channels=1, out_channels=1, kernel_size=1, padding=0, bias=False)
        smooth.weight = torch.nn.Parameter(kernel, requires_grad=False)
        return smooth

    def __call__(self, img):
        """
        Applies Gaussian smoothing to torch.Tensor image of shape [N, C, H, W])

        :param img: torch.Tensor
        :return: torch.Tensor
        """
        img, in_shape = get_as_2d_batch(img)
        kernel = self.get_kernel(img.size(1), self.sigma_range.sample()).to(img)
        s1, s2 = kernel.weight.shape[-2:]
        padding = [s1//2, s1//2, s2//2, s2//2]
        padded_img = torch.nn.functional.pad(img, padding, mode='reflect')
        return kernel(padded_img).view(in_shape)
