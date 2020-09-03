import torch.nn as nn


class DualViewEncoder(nn.Module):
    """
    Defines a module of multiple encoders, e.g. 2d or 3d Resnets
    that is applied to different items in a dictionary input batch.
    """
    def __init__(self, image_encoder, segmentation_encoder):
        """
        :param image_encoder: torch.nn.Module
             Encoder module to apply to images
        :param segmentation_encoder: torch.nn.Module
            Encoder module to apply to segmentations
        """
        super(DualViewEncoder, self).__init__()
        self.image_encoder = image_encoder
        self.segmentation_encoder = segmentation_encoder

    def forward(self, images, segmentations, transform_key=None):
        """
        TODO
        :param images: torch.Tensor
        :param segmentations: torch.Tensor
        :param transform_key: string
            A string identifier for the transform to apply in the given encoder
        :return: A list of outputs of len(self.encoders)
        """
        return self.image_encoder(images, transform_key=transform_key), \
               self.segmentation_encoder(segmentations, transform_key=transform_key)
