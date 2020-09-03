import logging
import torch
import torch.nn as nn
import torch.nn.functional as F

from InnerEye.Research.cpc.models.resnet_encoder import standardize_patch_size


def _size_16_encoder(num_channels, init_dim, res_block_depth,
                     encoding_dim, use_norm):
    """
    Defines a Resnet encoder for encoding patches of size 16x16x16 into vector
    """
    kwargs = {'depth': res_block_depth, "use_norm": use_norm}
    return nn.ModuleList([
        ConvNxN(num_channels, init_dim, 3, 1, 0, False),
        ConvResBlock(init_dim * 1, init_dim * 2, 3, 1, 0, **kwargs),
        ConvResBlock(init_dim * 2, init_dim * 2, 4, 2, 0, **kwargs),
        SwitchableBatchNorm3d(init_dim * 2, False, use_norm),
        ConvResBlock(init_dim * 2, init_dim * 4, 3, 1, 0, **kwargs),
        ConvNxN(init_dim * 4, encoding_dim, 3, 1, 0, False, False),
    ])


def _size_32_encoder(num_channels, init_dim, res_block_depth,
                     encoding_dim, use_norm):
    """
    Defines a Resnet encoder for encoding patches of size 32x32x32 into rep
    """
    kwargs = {'depth': res_block_depth, "use_norm": use_norm}
    return nn.ModuleList([
        ConvNxN(num_channels, init_dim, 3, 1, 0, False),
        ConvResNxN(init_dim, init_dim, 1, 1, 0, use_norm),
        ConvResBlock(init_dim * 1, init_dim * 2, 4, 2, 0, **kwargs),
        ConvResBlock(init_dim * 2, init_dim * 4, 2, 2, 0, **kwargs),
        SwitchableBatchNorm3d(init_dim * 4, False, use_norm),
        ConvResBlock(init_dim * 4, init_dim * 4, 3, 1, 0, **kwargs),
        ConvResBlock(init_dim * 4, init_dim * 4, 3, 1, 0, **kwargs),
        ConvNxN(init_dim * 4, encoding_dim, 3, 1, 0, False, False),
    ])


def _size_64_encoder(num_channels, init_dim, res_block_depth,
                     encoding_dim, use_norm):
    """
    Defines a Resnet encoder for encoding patches of size 64x64x64 into rep
    """
    kwargs = {'depth': res_block_depth, "use_norm": use_norm}
    return nn.ModuleList([
        ConvNxN(num_channels, init_dim, 3, 1, 0, False),
        ConvResNxN(init_dim, init_dim, 1, 1, 0, use_norm),
        ConvResBlock(init_dim * 1, init_dim * 2, 4, 2, 0, **kwargs),
        ConvResBlock(init_dim * 2, init_dim * 4, 4, 2, 0, **kwargs),
        ConvResBlock(init_dim * 4, init_dim * 8, 2, 2, 0, **kwargs),
        SwitchableBatchNorm3d(init_dim * 8, False, use_norm),
        ConvResBlock(init_dim * 8, init_dim * 8, 3, 1, 0, **kwargs),
        ConvResBlock(init_dim * 8, init_dim * 8, 3, 1, 0, **kwargs),
        ConvNxN(init_dim * 8, encoding_dim, 3, 1, 0, False, False),
    ])


def _size_128_encoder(num_channels, init_dim, res_block_depth,
                      encoding_dim, use_norm):
    """
    Defines a Resnet encoder for encoding patches of size 128x128x128 to vector
    """
    kwargs = {'depth': res_block_depth, "use_norm": use_norm}
    return nn.ModuleList([
        ConvNxN(num_channels, init_dim, 5, 2, 2, False, pad_mode='constant'),
        ConvNxN(init_dim, init_dim, 3, 1, 0, False),
        ConvResBlock(init_dim * 1, init_dim * 2, 4, 2, 0, **kwargs),
        ConvResBlock(init_dim * 2, init_dim * 4, 4, 2, 0, **kwargs),
        ConvResBlock(init_dim * 4, init_dim * 8, 2, 2, 0, **kwargs),
        SwitchableBatchNorm3d(init_dim * 8, False, use_norm),
        ConvResBlock(init_dim * 8, init_dim * 8, 3, 1, 0, **kwargs),
        ConvResBlock(init_dim * 8, init_dim * 8, 3, 1, 0, **kwargs),
        ConvNxN(init_dim * 8, encoding_dim, 3, 1, 0, False, False)
    ])


def _size_128_7_128_encoder(num_channels, init_dim, res_block_depth,
                            encoding_dim, use_norm):
    """
    Defines a Resnet encoder for encoding patches of size 128x7x128 to vector
    """
    kwargs = {'depth': res_block_depth, "use_norm": use_norm}
    return nn.ModuleList([
        ConvNxN(num_channels, init_dim, [5, 1, 5], [2, 1, 2], [2, 0, 2], False, pad_mode='constant'),
        ConvNxN(init_dim, init_dim, [3, 1, 3], 1, 0, False),
        ConvResBlock(init_dim * 1, init_dim * 2, [4, 1, 4], [2, 1, 2], 0, **kwargs),
        ConvResBlock(init_dim * 2, init_dim * 4, [4, 1, 4], [2, 1, 2], 0, **kwargs),
        ConvResBlock(init_dim * 4, init_dim * 8, [2, 1, 2], [2, 1, 2], 0, **kwargs),
        SwitchableBatchNorm3d(init_dim * 8, False, use_norm),
        ConvResBlock(init_dim * 8, init_dim * 8, [3, 3, 3], 1, 0, **kwargs),
        ConvResBlock(init_dim * 8, init_dim * 8, [3, 3, 3], 1, 0, **kwargs),
        ConvNxN(init_dim * 8, encoding_dim, [3, 3, 3], 1, 0, False, False),
    ])


def _size_16_98_170_encoder(num_channels, init_dim, res_block_depth,
                            encoding_dim, use_norm):
    """
    Defines a Resnet encoder for encoding patches of size 128x7x128 to vector
    """
    kwargs = {'depth': res_block_depth, "use_norm": use_norm}
    return nn.ModuleList([
        ConvNxN(num_channels, init_dim, [3, 5, 5], [2, 3, 3], 0, False),
        ConvResBlock(init_dim, init_dim, [3, 3, 5], [2, 2, 3], 0, **kwargs),
        ConvResBlock(init_dim, init_dim * 2, [1, 3, 6], [1, 2, 2], 0, **kwargs),
        SwitchableBatchNorm3d(init_dim * 2, False, use_norm),
        ConvResBlock(init_dim * 2, init_dim * 4, [1, 3, 3], 1, 0, **kwargs),
        ConvResBlock(init_dim * 4, init_dim * 8, [1, 3, 3], 1, 0, **kwargs),
        ConvNxN(init_dim * 8, encoding_dim, [3, 3, 3], 1, 0, False, False),
    ])


def _size_1_260_484_encoder(num_channels, init_dim, res_block_depth,
                            encoding_dim, use_norm):
    """
    Defines a Resnet encoder for encoding patches of size 1x256x484 to vector
    """
    kwargs = {'depth': res_block_depth, "use_norm": use_norm}
    return nn.ModuleList([
        ConvNxN(num_channels, init_dim, [1, 5, 6], [1, 1, 2], 0, False),
        ConvResBlock(init_dim, init_dim*2, [1, 4, 6], [1, 2, 2], 0, **kwargs),
        ConvResBlock(init_dim*2, init_dim*4, [1, 5, 4], [1, 2, 2], 0, **kwargs),
        ConvResBlock(init_dim*4, init_dim*8, [1, 4, 4], [1, 2, 2], 0, **kwargs),
        ConvResBlock(init_dim*8, init_dim*8, [1, 3, 1], [1, 1, 1], 0, **kwargs),
        ConvResBlock(init_dim*8, init_dim*8, [1, 4, 4], [1, 2, 2], 0, **kwargs),
        ConvResBlock(init_dim*8, init_dim*16, [1, 3, 3], [1, 2, 2], 0, **kwargs),
        ConvResBlock(init_dim*16, init_dim*32, [1, 3, 3], 1, 0, **kwargs),
        ConvResBlock(init_dim*32, init_dim*64, [1, 3, 3], 1, 0, **kwargs),
        ConvNxN(init_dim*64, encoding_dim, [1, 2, 2], 1, 0, False, False),
    ])


def _size_7_260_484_encoder(num_channels, init_dim, res_block_depth,
                            encoding_dim, use_norm):
    """
    Defines a Resnet encoder for encoding patches of size 1x256x484 to vector
    """
    kwargs = {'depth': res_block_depth, "use_norm": use_norm}
    return nn.ModuleList([
        ConvNxN(num_channels, init_dim, [3, 5, 7], [1, 3, 3], 0, False),
        ConvResBlock(init_dim, init_dim, [3, 5, 7], [1, 3, 3], 0, **kwargs),
        ConvResBlock(init_dim, init_dim * 2, [1, 4, 4], [1, 2, 2], 0, **kwargs),
        ConvResBlock(init_dim * 2, init_dim * 2, [3, 5, 5], [1, 1, 2], 0, **kwargs),
        ConvResBlock(init_dim * 2, init_dim * 4, [1, 5, 5], 1, 0, **kwargs),
        ConvResBlock(init_dim * 4, init_dim * 8, [1, 3, 5], 1, 0, **kwargs),
        ConvNxN(init_dim * 8, encoding_dim, [1, 3, 3], 1, 0, False, False),
    ])


class ConvNxN(nn.Module):
    """
    Implements a 3d conv. block with optional layer norm and relu activation.
    TODO
    """
    def __init__(self, n_in, n_out, width, n_stride, n_pad,
                 use_norm=False, use_activation=True, pad_mode='constant'):
        """
        TODO
        """
        super(ConvNxN, self).__init__()
        assert(pad_mode in ['constant', 'reflect'])
        self.conv = nn.Conv3d(n_in, n_out, width, n_stride, n_pad,
                              bias=(not use_norm),
                              padding_mode=pad_mode)
        self.activation = nn.ReLU(inplace=True) if use_activation else None
        self.norm = SwitchableBatchNorm3d(n_out, False, use_norm)

    def forward(self, x):
        x = self.conv(x)
        x = self.norm(x)
        if self.activation:
            return self.activation(x)
        else:
            return x


class ConvResNxN(nn.Module):
    """
    A 3d resnet module with avg. pooled residual connection and relu activation
    TODO
    """
    def __init__(self, n_in, n_out, width, stride, pad, use_norm=False):
        """
        TODO
        """
        super(ConvResNxN, self).__init__()
        assert (n_out >= n_in)
        self.n_in = n_in
        self.n_out = n_out
        self.width = width
        self.stride = stride
        self.pad = pad
        self.activation = nn.ReLU(inplace=True)
        self.conv1 = nn.Conv3d(n_in, n_out, width, stride, pad, bias=False)
        self.conv2 = nn.Conv3d(n_out, n_out, 1, 1, 0, bias=False)
        self.norm = SwitchableBatchNorm3d(n_out, False, use_norm)

    def forward(self, x):
        x_res = x
        x = self.norm(self.conv1(x))
        x = self.conv2(self.activation(x))
        x_res = F.avg_pool3d(x_res, self.width, self.stride, self.pad)
        if self.n_out > self.n_in:
            x_res = F.pad(x_res, (0, 0, 0, 0, 0, 0, 0, self.n_out - self.n_in))
        return x + x_res


class ConvResBlock(nn.Module):
    """
    Applies 'depth' convolutional resnet modules to an input.
    TODO
    """
    def __init__(self, n_in, n_out, width, stride, pad, depth, use_norm):
        """
        TODO
        """
        super(ConvResBlock, self).__init__()
        layer_list = [ConvResNxN(n_in, n_out, width, stride, pad, use_norm)]
        for i in range(depth - 1):
            layer_list.append(ConvResNxN(n_out, n_out, 1, 1, 0, use_norm))
        self.layer_list = nn.Sequential(*layer_list)
        return

    def forward(self, x):
        x_out = self.layer_list(x)
        return x_out


class SwitchableBatchNorm3d(nn.Module):
    """
    Implements a BatchNorm layer that may be toggled on/off by setting the
    use_norm attribute. With use_norm == False the layer is not applied in
    the forward call.
    """
    def __init__(self, n_ftr, affine=True, use_norm=True, running=True):
        super(SwitchableBatchNorm3d, self).__init__()
        if use_norm:
            self.norm = nn.BatchNorm3d(n_ftr,
                                       affine=affine,
                                       track_running_stats=running)
        self.use_norm = use_norm

    def forward(self, x):
        if self.use_norm:
            x = self.norm(x)
        return x


PATCH_SIZE_STR_TO_MODULE_FUNC = {
    "16x16x16": _size_16_encoder,
    "32x32x32": _size_32_encoder,
    "64x64x64": _size_64_encoder,
    "128x128x128": _size_128_encoder,
    "128x7x128": _size_128_7_128_encoder,
    "16x98x170": _size_16_98_170_encoder,
    "1x260x484": _size_1_260_484_encoder,
    "7x260x484": _size_7_260_484_encoder
}


class ResnetEncoder3d(nn.Module):
    """
    Defines a set of Resnet-like 3D encoder modules for use within the InnerEye.Research.cpc project.

    This class primarily serves as an interface for initializing encoders that accept differently sized patched inputs.
    The input to self.forward must either be a patched input, or a transform_key must be passed and a corresponding
    transform (callable) set under the self.transform dict. See self.forward for details.

    Encoder architechtures are loosely inspired by work in GreedyInfoMax:
    https://github.com/loeweX/Greedy_InfoMax
    https://arxiv.org/abs/1905.11786v2

    Supports the following input image sizes:
        16x16x16
        32x32x32
        64x64x64
        128x128x128
        128x7x128
        16x98x170
        1x260x484
        7x260x484

    Each model is defined in its own function named _size_{size}_encoder, please refer to PATCH_SIZE_STR_TO_MODULE_FUNC
    Each model differs in layer topology and number of parameters for a given set of inputs to ResnetEncoder3d.
    """
    def __init__(self,
                 input_patch_size=32,
                 num_channels=1,
                 init_dim=64,
                 encoding_dim=512,
                 res_block_depth=3,
                 use_norm=False,
                 return_n_last_layers=0,
                 transforms=None):
        """
        TODO
        """
        super(ResnetEncoder3d, self).__init__()
        self.init_dim = init_dim
        self.encoding_dim = encoding_dim
        self.use_norm = use_norm
        self.return_n_last_layers = max(1, int(return_n_last_layers) + 1)

        # encoding block for local features
        kwargs = {'num_channels': num_channels, 
                  "init_dim": init_dim,
                  "res_block_depth": res_block_depth,
                  "encoding_dim": encoding_dim,
                  "use_norm": use_norm}
        input_patch_size = standardize_patch_size(input_patch_size, dim=3)
        logging.info('Using a {} encoder'.format(input_patch_size))
        if input_patch_size not in PATCH_SIZE_STR_TO_MODULE_FUNC:
            raise RuntimeError("Could not build encoder. "
                               "ResnetEncoder size {} is not "
                               "supported".format(input_patch_size))
        self.layer_list = PATCH_SIZE_STR_TO_MODULE_FUNC[input_patch_size](**kwargs)

        for m in self.modules():
            if isinstance(m, nn.Conv3d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, (nn.BatchNorm3d, nn.GroupNorm)):
                if m.weight is not None:
                    nn.init.constant_(m.weight, 1)
                    nn.init.constant_(m.bias, 0)
        logging.info('Encoder parameters: {:.4f} million'.format(
            sum(s.numel() for s in self.parameters()) / 10 ** 6)
        )
        self.transforms = None
        self.set_transforms(transforms)

    def set_transforms(self, transforms):
        """
        Register a dictionary of transforms to apply in if specified in the forward pass
        :param transforms:
        """
        if transforms is not None and not isinstance(transforms, dict):
            raise ValueError("Must specify a dictionary of {key: transform} mapping.")
        self.transforms = transforms

    def forward(self, x_patches, transform_key=None):
        """
        Note: Operates on pre-patched inputs or inputs which are to be processed by a callable
        self.transforms[transform_key] before encoding.

        A patched input is considered a tensor of shape [N, g1, g2, g3, C, d1, d2, d3] where g* are grid dimensions
        and d* are spatial dimensions of the 3D data. C is the number of channels and N the number of samples.

        :param x_patches: torch.Tensor
            Input patches of shape [N, g1, g2, g3, C, d1, d2, d3] if transform_key=None or some tensor
            (typically [N, C, D1, D2, D3]) to which self.transforms[transform_key] can be applied to get a torch.Tensor
            of that shape.
        :param transform_key: None or str
            An optional key to a callable in self.transforms (see self.set_transforms) to apply to x_patches to
            pre-process, patch, augment etc. before encoding.
        :returns: torch.Tensor
            The inputs encoded, shape [N, C_enc, g1, g2, g3] where g1, g2 and g3 are grid dimensions.
        """
        if transform_key:
            with torch.no_grad():
                x_patches = self.transforms[transform_key](x_patches)
        assert x_patches.ndim == 8, "Input after transforming must be of nim 8, got {} {}".format(x_patches.ndim,
                                                                                                  x_patches.shape)
        # Stack patches from [N, g1, g2, g3, C, d1, d2, d3] --> [-1, C, d1, d2, d3]
        n_patches_x, n_patches_y, n_patches_z = x_patches.shape[1:4]
        patches_in_batch = len(x_patches)*n_patches_x*n_patches_y*n_patches_z
        x = x_patches.view([patches_in_batch] + list(x_patches.shape[4:]))

        # Encode all the patches
        to_return = []
        for i, layer in enumerate(self.layer_list):
            x = layer(x)
            if i >= len(self.layer_list)-self.return_n_last_layers:
                to_return.append(x)

        for i, x in enumerate(to_return):
            # Ensure vector encodings by adaptive avg. pooling
            x = nn.functional.adaptive_avg_pool3d(x, [1, 1, 1]).squeeze()  # shape [-1, C]

            # Go back to patch-view [-1, C] --> [N, g1, g2, g3, C]
            x = x.view(-1, n_patches_x, n_patches_y, n_patches_z, x.shape[1])
            x = x.permute(0, 4, 1, 2, 3).contiguous()  # --> [N, C, g1, g2, g3]
            to_return[i] = x
        to_return = to_return[::-1]
        if self.return_n_last_layers == 1:
            return to_return[0]
        else:
            return to_return[0], to_return[1:]
