import torch
from scipy.ndimage.filters import _gaussian_kernel1d
from InnerEye.Research.cpc.transforms.utils import RandomParameterRange, broadcast_to_n_dims, get_as_2d_batch


def _elastic_transform_2d(image, alphas, sigmas, interpolation="bilinear"):
    """
    Applies random elastic deformations to batch of 2d images (torch.Tensor of shape [N, C, H, W])
    The same transform is applied to all elements N.

    :param image: torch.Tensor of shape [N, C, H, W]
    :param alphas: list of length 2 of augmentation amplifier parameters applied along each spatial dim H, W
    :param sigmas: list of length 2 of augmentation smoothing parameters applied along each spatial dim H, W
    :param interpolation: str, 'bilinear' | 'nearest'
    :return: Augmented torch.Tensor of shape [N, C, H, W]
    """
    # Get Gaussian kernels to apply along each dim + padding needed to maintain input size in output
    kernels = [torch.from_numpy(_gaussian_kernel1d(sigma, 0, int(3 * sigma + 0.5))).to(image.device).to(image.dtype) for sigma in sigmas]
    paddings = [(len(kernel)//2)*2 for kernel in kernels]
    image_shape = image.shape[2:]

    # Apply the Gaussian kernels along Gaussian displacement noise sampled for each pixel along each axis
    smooth_noise_field = []
    for i, (kernel, alpha, pad) in enumerate(zip(kernels, alphas, paddings)):
        padded_shape = [im_s + pad for im_s in image_shape]
        kernel = torch.matmul(kernel.view(1, -1, 1), kernel.view(1, 1, -1)).unsqueeze(0)
        noise = torch.rand(*padded_shape, device=image.device, dtype=image.dtype) * 2 - 1
        noise = noise.view(1, 1, *noise.shape)
        conv = torch.nn.Conv2d(in_channels=1, out_channels=1, kernel_size=1, padding=0, bias=False)
        conv.weight = torch.nn.Parameter(kernel, requires_grad=False)
        smooth_noise = conv(noise).squeeze()
        smooth_noise_field.append(smooth_noise / smooth_noise.std() * alpha)  # normalize field before scaling
    smooth_noise_field = torch.stack(smooth_noise_field)

    # Get in image coordinates
    r1, r2 = map(lambda s: torch.arange(s, device=image.device, dtype=torch.float32), image_shape)
    grid = torch.stack(torch.meshgrid([r1, r2]))
    displaced_grid = grid + smooth_noise_field

    # Grid must be normalized by image size to [-1, 1] values
    size_normalizer = torch.as_tensor(image_shape).view(-1, 1, 1)
    normed_displacement_grid = (displaced_grid / size_normalizer.to(displaced_grid.device)) * 2 - 1

    # Permute grid and image
    image = image.permute(0, 1, 3, 2)
    normed_displacement_grid = torch.stack(len(image) * [normed_displacement_grid.permute(1, 2, 0)], dim=0)

    # Sample the image on the displaced grid
    sampled = torch.nn.functional.grid_sample(
        image,
        normed_displacement_grid,
        mode=interpolation,
        padding_mode="zeros",
        align_corners=False
    )
    return sampled


def elastic_transform_2d(image, alphas, sigmas, interpolation="bilinear"):
    """
    Elastic deformation of images as described in [Simard2003]_.
    [Simard2003] Simard, Steinkraus and Platt, "Best Practices for
    Convolutional Neural Networks applied to Visual Document Analysis", in
    Proc. of the International Conference on Document Analysis and
    Recognition, 2003.

    :param image: Torch.Tensor of shape [N, C, H, W], [N, H, W] or [H, W]
    :param alphas: list of length 2 of augmentation amplifier parameters applied along each spatial dim H, W
    :param sigmas: list of length 2 of augmentation smoothing parameters applied along each spatial dim H, W
    :param interpolation: str in ("bilinear", "nearest")
    :returns: Torch.Tensor of shape [N, C, H, W], [N, H, W] or [H, W]
    """
    image, in_shape = get_as_2d_batch(image)
    # Setup parameters to span all spatial and channel dimensions
    alphas = broadcast_to_n_dims(alphas, image.ndim-2)
    sigmas = broadcast_to_n_dims(sigmas, image.ndim-2)
    return _elastic_transform_2d(image, alphas, sigmas, interpolation=interpolation).view(in_shape)


class ElasticTransform2d:
    """
    Defines a random elastic transform with random parameter value ranges for 2d images.
    See docstring of elastic_transform_2d
    """
    def __init__(self, alpha_range, sigma_range, interpolation="bilinear"):
        """
        :param alpha_range: list, tuple, ndarray of length 2
            Range (e.g. [1, 10]) in which to randomly sample alpha (deformation strength parameters) at each call
        :param sigma_range: list, tuple, ndarray of length 2
            Range (e.g. [1, 10]) in which to randomly sample sigma (deformation smoothing parameters) at each call
        :param interpolation: str in ("bilinear", "nearest")
        """
        RandomParameterRange.check_list_length_2(alpha_range, "alpha_range")
        RandomParameterRange.check_list_length_2(sigma_range, "sigma_range")
        self.alpha_range = RandomParameterRange(*alpha_range, log_scale_sampling=False)
        self.sigma_range = RandomParameterRange(*sigma_range, log_scale_sampling=False)
        assert interpolation in ("bilinear", "nearest")
        self.interpolation = interpolation

    def __call__(self, image):
        """
        Applies 2d elastic transformations according to parameters in self.alpha_range and self.sigma_range.

        :param image: torch.Tensor of shape [N, C, H, W], [N, H, W] or [H, W]
        :returns: torch.Tensor of shape [N, C, H, W], [N, H, W] or [H, W]
        """
        return elastic_transform_2d(image=image,
                                    alphas=self.alpha_range.sample(),
                                    sigmas=self.sigma_range.sample(),
                                    interpolation=self.interpolation)


if __name__ == "__main__":
    import matplotlib.pyplot as plt
    from skimage.data import camera
    image = torch.from_numpy(camera()).to(torch.float32)
    elastic_aug = ElasticTransform2d(alpha_range=[2, 6], sigma_range=[15, 25])

    for _ in range(10):
        augmented_im = elastic_aug(image)
        fig, (ax1, ax2) = plt.subplots(ncols=2, figsize=(12, 6))
        ax1.imshow(image.numpy(), cmap="gray")
        ax2.imshow(augmented_im.numpy(), cmap="gray")
        fig.tight_layout()
        plt.show()
        plt.close(fig)
