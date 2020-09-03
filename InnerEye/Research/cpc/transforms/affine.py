import torch
import numpy as np
from InnerEye.Research.cpc.transforms.utils import get_as_2d_batch
from torchvision.transforms import RandomAffine


class RandomAffine2d(RandomAffine):
    """
    Implementation over torchvision.transforms.RandomAffine which accepts
    tensors instead of PIL Images and can be run on GPU.
    """
    def __init__(self, degrees, translate=None, scale=None, shear=None, interpolation="bilinear"):
        super(RandomAffine2d, self).__init__(degrees, translate, scale, shear)
        assert interpolation in ("bilinear", "nearest")
        self.interpolation = interpolation

    def __call__(self, image):
        """
        image (torch.Tensor): Image to be transformed. Must be shape [N, C, H, W], [N, H, W] or [H, W]

        Returns:
            Torch.Tensor: Transformed image, same shape as input
        """
        image, in_shape = get_as_2d_batch(image)
        image_size = image.shape[2:]
        angle, translations, scale, shear = self.get_params(self.degrees, self.translate, self.scale, self.shear, image_size)
        shear = shear if isinstance(shear, list) else [shear, shear]
        angle = np.deg2rad(angle)
        cos_angle, sin_angle = np.cos(angle), np.sin(angle)
        theta = torch.as_tensor(
            [[scale*cos_angle - shear[0]*sin_angle, shear[0]*cos_angle + scale*sin_angle, 0],
             [shear[1]*cos_angle - scale*sin_angle, scale*cos_angle + shear[1]*sin_angle, 0],
             [translations[0], translations[1], 1]],
            device=image.device, dtype=torch.float32
        )
        r1, r2 = map(lambda s: torch.arange(s, device=image.device, dtype=torch.float32), image_size)
        grid = torch.stack(torch.meshgrid([r1, r2]))
        coord_grid = grid.reshape(2, -1)
        center = torch.mean(coord_grid, dim=1).view((2, 1))
        coord_grid_normed = coord_grid - center
        coord_grid_normed = torch.cat((coord_grid_normed, torch.ones_like(coord_grid_normed[0]).unsqueeze(0)), dim=0)

        # Center grid
        coord_grid = (torch.matmul(theta.T, coord_grid_normed)[:-1] + center)
        coord_grid = coord_grid.reshape(2, image_size[0], image_size[1])

        # Grid must be normalized by image size to [-1, 1] values
        size_normalizer = torch.as_tensor(image_size).view(-1, 1, 1)
        coord_grid = (coord_grid / size_normalizer.to(coord_grid.device)) * 2 - 1

        # Permute grid and image
        image = image.permute(0, 1, 3, 2)
        coord_grid = torch.stack(len(image) * [coord_grid.permute(1, 2, 0)], dim=0)

        # Sample the image on the displaced grid
        sampled = torch.nn.functional.grid_sample(
            image,
            coord_grid,
            mode=self.interpolation,
            padding_mode="zeros",
            align_corners=False
        )
        return sampled.view(in_shape)
