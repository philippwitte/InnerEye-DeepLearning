import numpy as np


class TensorToTorchIOFormat:
    def __init__(self, permute=None):
        """
        TODO
        """
        self.permute = permute

    def __call__(self, img_tensor):
        if self.permute:
            img_tensor = img_tensor.permute(self.permute)
        return {"image": {
            "data": img_tensor,
            "affine": np.eye(4),
            "type": "intensity"
        }}


class TorchIOFormatToTensor:
    def __init__(self, permute=None):
        """
        TODO
        """
        self.permute = permute

    def __call__(self, img_dict):
        image = img_dict["image"]["data"]
        if self.permute:
            image = image.permute(self.permute)
        return image


class ToCuda:
    def __init__(self, device=None):
        self.device = device

    def __call__(self, img_tensor):
        return img_tensor.cuda(self.device)


class DictBatchWrapper:
    def __init__(self, transform, key="images"):
        self.transform = transform
        self.key = key

    def __call__(self, dict_batch):
        dict_batch[self.key] = self.transform(dict_batch[self.key])
        return dict_batch


class ApplyToAllSlices:
    """
    Re-orders a 3d image Tensor of shape [C, d1, d2, d3] into shape [d1, C, 1, d2, d3],
    making the image appear as a batch of 2d images with a dummy 3rd dimension.

    Can be used to apply e.g. an Affine2d transform along all d1 images of shape [d2, d2] along dimension 1 of the input.

    This could be used to apply the same transformation to all Slice images in a 3D MED image
    (same 2d transform to all Slices in stack of Slices).
    """

    def __init__(self, transform):
        self.transform = transform

    def __str__(self):
        return "ApplyToAllSlices(\n{}\n)".format(self.transform)

    def __call__(self, image):
        """
        Takes a torch.Tensor [C, d1, d2, d3] and applies self.transform
        to reordered batch of shape [d1, C, 1, d2, d3]. Returns

        :param image: tensor of shape [C, d1, d2, d3]
        :return: tensor of shape [C, d1, d2, d3]
        """
        image = image.permute(1, 0, 2, 3).unsqueeze(2)
        image = self.transform(image).squeeze(2)
        return image.permute(1, 0, 2, 3)
