import logging
import torch.nn as nn
import torch.nn.functional as F


def standardize_patch_size(patch_size, dim=2):
    """
    """
    to_str = lambda patch_size: "x".join(map(str, patch_size))
    if isinstance(patch_size, int):
        return to_str([patch_size] * dim)
    elif isinstance(patch_size, str):
        return to_str(patch_size.lower().strip().split("x"))
    elif isinstance(patch_size, (list, tuple)):
        if len(patch_size) != dim:
            raise ValueError("Expected to create an encoder for inputs of "
                             "dimensionality {}, but got patch_size "
                             "{}".format(dim, patch_size))
        return to_str(patch_size)
    else:
        raise ValueError("Received unexpected patch_size argument {} of"
                         " type {}".format(patch_size, type(patch_size)))


def _size_16_encoder(num_channels, init_dim, res_block_depth,
                     encoding_dim, use_norm):
    """
    Defines a Resnet encoder for encoding patches of size 16x16 into vector rep
    """
    kwargs = {'depth': res_block_depth, "use_norm": use_norm}
    return nn.ModuleList([
        ConvNxN(num_channels, init_dim, 3, 1, 0, False),
        ConvResBlock(init_dim * 1, init_dim * 2, 3, 1, 0, **kwargs),
        ConvResBlock(init_dim * 2, init_dim * 2, 4, 2, 0, **kwargs),
        SwitchableInstanceNorm2d(init_dim * 2, False, use_norm),
        ConvResBlock(init_dim * 2, init_dim * 4, 3, 1, 0, **kwargs),
        ConvResNxN(init_dim * 4, encoding_dim, 3, 1, 0, False),
    ])


def _size_32_encoder(num_channels, init_dim, res_block_depth,
                     encoding_dim, use_norm):
    """
    Defines a Resnet encoder for encoding patches of size 32x32 into vector rep
    """
    kwargs = {'depth': res_block_depth, "use_norm": use_norm}
    return nn.ModuleList([
        ConvNxN(num_channels, init_dim, 3, 1, 0, False),
        ConvResNxN(init_dim, init_dim, 1, 1, 0, use_norm),
        ConvResBlock(init_dim * 1, init_dim * 2, 4, 2, 0, **kwargs),
        ConvResBlock(init_dim * 2, init_dim * 4, 2, 2, 0, **kwargs),
        SwitchableInstanceNorm2d(init_dim * 4, False, use_norm),
        ConvResBlock(init_dim * 4, init_dim * 4, 3, 1, 0, **kwargs),
        ConvResBlock(init_dim * 4, init_dim * 4, 3, 1, 0, **kwargs),
        ConvResNxN(init_dim * 4, encoding_dim, 3, 1, 0, False),
    ])


def _size_64_encoder(num_channels, init_dim, res_block_depth,
                     encoding_dim, use_norm):
    """
    Defines a Resnet encoder for encoding patches of size 64x64 into vector rep
    """
    kwargs = {'depth': res_block_depth, "use_norm": use_norm}
    return nn.ModuleList([
        ConvNxN(num_channels, init_dim, 3, 1, 0, False),
        ConvResNxN(init_dim, init_dim, 1, 1, 0, use_norm),
        ConvResBlock(init_dim * 1, init_dim * 2, 4, 2, 0, **kwargs),
        ConvResBlock(init_dim * 2, init_dim * 4, 4, 2, 0, **kwargs),
        ConvResBlock(init_dim * 4, init_dim * 8, 2, 2, 0, **kwargs),
        SwitchableInstanceNorm2d(init_dim * 8, False, use_norm),
        ConvResBlock(init_dim * 8, init_dim * 8, 3, 1, 0, **kwargs),
        ConvResBlock(init_dim * 8, init_dim * 8, 3, 1, 0, **kwargs),
        ConvResNxN(init_dim * 8, encoding_dim, 3, 1, 0, False),
    ])


def _size_128_encoder(num_channels, init_dim, res_block_depth,
                      encoding_dim, use_norm):
    """
    Defines a Resnet encoder for encoding patches of size 128x128 to vector rep
    """
    kwargs = {'depth': res_block_depth, "use_norm": use_norm}
    return nn.ModuleList([
        ConvNxN(num_channels, init_dim, 5, 2, 2, False, pad_mode='reflect'),
        ConvNxN(init_dim, init_dim, 3, 1, 0, False),
        ConvResBlock(init_dim * 1, init_dim * 2, 4, 2, 0, **kwargs),
        ConvResBlock(init_dim * 2, init_dim * 4, 4, 2, 0, **kwargs),
        ConvResBlock(init_dim * 4, init_dim * 8, 2, 2, 0, **kwargs),
        SwitchableInstanceNorm2d(init_dim * 8, False, use_norm),
        ConvResBlock(init_dim * 8, init_dim * 8, 3, 1, 0, **kwargs),
        ConvResBlock(init_dim * 8, init_dim * 8, 3, 1, 0, **kwargs),
        ConvResNxN(init_dim * 8, encoding_dim, 3, 1, 0, False),
    ])


class ConvNxN(nn.Module):
    """
    Implements a 2D conv. block with optional layer norm and elu activaiton.
    TODO
    """

    def __init__(self, n_in, n_out, width, n_stride, n_pad,
                 use_norm=True, pad_mode='constant'):
        """
        TODO
        """
        super(ConvNxN, self).__init__()
        assert(pad_mode in ['constant', 'reflect'])
        self.n_pad = (n_pad,)*4
        self.pad_mode = pad_mode
        self.conv = nn.Conv2d(n_in, n_out, width, n_stride, 0, bias=(not use_norm))
        self.elu = nn.ELU(inplace=True)
        self.norm = SwitchableInstanceNorm2d(n_out, False, use_norm)

    def forward(self, x):
        if self.n_pad[0] > 0:
            x = F.pad(x, self.n_pad, mode=self.pad_mode)
        x = self.conv(x)
        x = self.norm(x)
        out = self.elu(x)
        return out


class ConvResNxN(nn.Module):
    """
    A 2D resnet module with avg. pooled residual connection and elu activation
    TODO
    """

    def __init__(self, n_in, n_out, width, stride, pad, use_norm=True):
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
        self.elu = nn.ELU(inplace=True)
        self.conv1 = nn.Conv2d(n_in, n_out, width, stride, pad, bias=False)
        self.conv2 = nn.Conv2d(n_out, n_out, 1, 1, 0, bias=False)
        self.norm = SwitchableInstanceNorm2d(n_out, False, use_norm)

    def forward(self, x):
        x_res = x
        x = self.norm(self.conv1(x))
        x = self.conv2(self.elu(x))
        if self.n_in == self.n_out:
            x_res = F.avg_pool2d(x_res, self.width, self.stride, self.pad)
        else:
            x_res = F.avg_pool2d(x_res, self.width, self.stride, self.pad)
            x_res = F.pad(x_res, (0, 0, 0, 0, 0, self.n_out - self.n_in))
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


class SwitchableInstanceNorm2d(nn.Module):
    """
    Implements a InstanceNorm layer that may be toggled on/off by setting the
    use_norm attribute. With use_norm the layer is not applied in
    the forward call.
    """

    def __init__(self, n_ftr, affine=False, use_norm=True, running=False):
        super(SwitchableInstanceNorm2d, self).__init__()
        self.norm = nn.InstanceNorm2d(n_ftr,
                                      affine=affine,
                                      track_running_stats=running)
        self.use_norm = use_norm

    def forward(self, x):
        if self.use_norm:
            x = self.norm(x)
        return x


PATCH_SIZE_STR_TO_MODULE_FUNC = {
    "16x16": _size_16_encoder,
    "32x32": _size_32_encoder,
    "64x64": _size_64_encoder,
    "128x128": _size_128_encoder
}


class ResnetEncoder2d(nn.Module):
    def __init__(self,
                 input_patch_size=32,
                 num_channels=1,
                 init_dim=64,
                 encoding_dim=512,
                 res_block_depth=3,
                 use_norm=True):
        """
        TODO
        """
        super(ResnetEncoder2d, self).__init__()
        self.init_dim = init_dim
        self.encoding_dim = encoding_dim
        self.use_norm = use_norm

        # encoding block for local features
        kwargs = {'num_channels': num_channels,
                  "init_dim": init_dim,
                  "res_block_depth": res_block_depth,
                  "encoding_dim": encoding_dim,
                  "use_norm": use_norm}
        input_patch_size = standardize_patch_size(input_patch_size, dim=2)
        logging.info('Using a {} encoder'.format(input_patch_size))
        if input_patch_size not in PATCH_SIZE_STR_TO_MODULE_FUNC:
            raise RuntimeError("Could not build encoder. "
                               "ResnetEncoder size {} is not "
                               "supported".format(input_patch_size))
        self.layer_list = PATCH_SIZE_STR_TO_MODULE_FUNC[input_patch_size](**kwargs)
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, (nn.InstanceNorm2d, nn.GroupNorm)):
                if m.weight is not None:
                    nn.init.constant_(m.weight, 1)
                    nn.init.constant_(m.bias, 0)
        logging.info('Encoder parameters: {:.4f} million'.format(
            sum(s.numel() for s in self.parameters()) / 10 ** 6)
        )

    def forward(self, x_patches):
        # Stack patches
        n_patches_x, n_patches_y = x_patches.shape[1:3]
        patches_in_batch = len(x_patches)*n_patches_x*n_patches_y
        x = x_patches.view([patches_in_batch] + list(x_patches.shape[3:]))

        # Encode all the patches
        for layer in self.layer_list:
            x = layer(x)

        # Ensure vector encodings by adaptive avg. pooling
        x = nn.functional.adaptive_avg_pool2d(x, 1)

        # Go back to patch-view
        x = x.view(-1, n_patches_x, n_patches_y, x.shape[1])
        return x.permute(0, 3, 1, 2).contiguous()
