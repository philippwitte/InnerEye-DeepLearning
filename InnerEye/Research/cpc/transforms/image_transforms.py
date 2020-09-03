import torch
import numpy as np
from InnerEye.Research.cpc.transforms.utils import broadcast_to_n_dims
from abc import abstractmethod


class _Crop3d:
    """
    Implements cropping of 3D image tensor/arrays of ndim = 4
    Input images must be tensor-like of shape [C, d1, d2, d3]
    Does not support padding. Input is expected to be at least as large as the
    requested crop in each dimension.
    """

    def __init__(self, crop_size):
        """
        TODO
        """
        self.crop_size = broadcast_to_n_dims(crop_size, 3)

    @staticmethod
    @abstractmethod
    def get_params(img, output_size):
        raise NotImplemented

    def __call__(self, image):
        if len(image.shape) not in (4, 5):
            raise ValueError("Only valid for 3D images and sequences of such "
                             "(ndim 4 or 5), got shape {}".format(image.shape))
        p_h, p_w, p_d = self.get_params(image, self.crop_size)
        cropped_image = image[..., p_h[0]:p_h[1], p_w[0]:p_w[1], p_d[0]:p_d[1]]
        return cropped_image


class RandomCrop3d(_Crop3d):
    @staticmethod
    def get_params(img, output_size):
        """
        Adapted for 3D inputs from torchvision.transforms.RandomCrop
        Get parameters for ``crop`` for a random crop.

        Args:
            img (Tensor-like): Image to be cropped.
            output_size (tuple): Expected output size of the crop.

        Returns:
            TODO
        """
        h, w, d = img.shape[-3:]
        th, tw, td = output_size
        if w == tw and h == th and d == td:
            return (0, h), (0, w), (0, d)
        i = np.random.randint(0, h - th + 1)
        j = np.random.randint(0, w - tw + 1)
        k = np.random.randint(0, d - td + 1)
        return (i, i+th), (j, j+tw), (k, k+td)


class CenterCrop3d(_Crop3d):
    """
    Implements center cropping of 3D image tensor/arrays of ndim = 4
    Input images must be tensor-like of shape [C, d1, d2, d3]
    Input is expected to be at least as large as the requested crop in each dimension.
    """
    @staticmethod
    def get_params(img, output_size):
        """
        Adapted for 3D inputs from torchvision.transforms.CenterCrop
        Get parameters for ``crop`` for a random crop.

        Args:
            img (Tensor-like): Image to be cropped.
            output_size (tuple): Expected output size of the crop.

        Returns:
            TODO
        """
        h, w, d = img.shape[-3:]
        th, tw, td = output_size
        if w == tw and h == th and d == td:
            return (0, h), (0, w), (0, d)
        i = round((h - th) // 2)
        j = round((w - tw) // 2)
        k = round((d - td) // 2)
        return (i, i+th), (j, j+tw), (k, k+td)


class OneHotEncode:
    """
    Applies a one-hot encoding transform to an input tensor along a specified dim
    :param dim:
    """

    def __init__(self, num_classes, dim=1):
        self.num_classes = int(num_classes)
        self.dim = int(dim)

    def __call__(self, img):
        """
        :param img: torch.Tensor, of arbitrary shape

        :return:
        """
        if img.ndim > self.dim and img.size(self.dim) == 1:
            img = img.squeeze(self.dim)
        dtype = img.dtype
        img = torch.nn.functional.one_hot(img.to(torch.long), self.num_classes)
        # Re-order dims to [N, K, ...]
        dim_order = list(range(img.ndim-1))
        dim_order.insert(self.dim, img.ndim-1)
        return img.permute(*dim_order).to(dtype)


class SliceCropper:
    """
    Crops a MED down in size by discarding Slices outside the specified range.
    """

    def __init__(self, range_start, range_end):
        self.start = int(range_start)
        self.end = int(range_end)
        assert self.start < self.end, "Start of range must be smaller than end"

    def __call__(self, img):
        """
        Crops input Tensor of shape [N, C, D1, D2, D3] along D1

        :param img: torch.Tensor
        :return: cropped torch.Tensor
        """
        if not img.ndim == 5:
            raise ValueError("Input image must be of ndim == 5, shape [N, C, D1, D2, D3], "
                             "but got ndim {} with shape {}".format(img.ndim, img.shape))
        return img[:, :, self.start:self.end]


class SliceDistanceTransform:
    """
    Computes the L2 distance to the center of a 2D image for each pixel in a 2D image
    and concatenates this distance map along the channel dimension to all input 2D images
    of a 3D volume.
    """

    def __init__(self, slice_shape):
        """

        :param slice_shape:
        """
        assert len(slice_shape) == 2, "Should specify only the 2 spatial dimensions of the Slice"
        from scipy.ndimage import distance_transform_edt
        map_ = np.ones(shape=slice_shape)
        map_[:, map_.shape[1]//2] = 0
        shape = [1, 1, 1, map_.shape[0], map_.shape[1]]
        distance_map = torch.from_numpy(distance_transform_edt(map_).reshape(shape)).to(torch.float32)
        self.distance_map = distance_map / distance_map.max()

    def __call__(self, img):
        """
        Concatenates a distance map to input Tensor of shape [N, C, D1, D2, D3] on dim C

        :param img: torch.Tensor
        :return: image with distance map concatenated torch.Tensor
        """
        if not img.ndim == 5:
            raise ValueError("Input image must be of ndim == 5, shape [N, C, D1, D2, D3], "
                             "but got ndim {} with shape {}".format(img.ndim, img.shape))
        repeated_distance_map = self.distance_map.repeat(img.shape[0], 1, img.shape[2], 1, 1)
        repeated_distance_map = repeated_distance_map.to(img.dtype).to(img.device)
        img_with_distance_map = torch.cat((img, repeated_distance_map), dim=1)
        return img_with_distance_map
