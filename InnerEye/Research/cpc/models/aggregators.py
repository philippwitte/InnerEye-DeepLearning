from torch import nn
from InnerEye.Research.cpc.transforms.utils import broadcast_to_n_dims


class VerticallyMaskedConv2d(nn.Conv2d):
    """
    TODO
    """
    def __init__(self,  in_channels, out_channels, kernel_size, bias=True):
        """
        TODO
        """
        kernel_size = [int(k) for k in broadcast_to_n_dims(kernel_size, 2)]
        padding = [int(k//2) for k in kernel_size]
        super(VerticallyMaskedConv2d, self).__init__(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            bias=bias,
            padding=padding,
            padding_mode='zeros'
        )
        self.register_buffer('mask', self.weight.clone())
        kernel_height = self.weight.shape[3]
        self.mask.fill_(1)
        self.mask[:, :, kernel_height // 2 + 1:, :] = 0

    def forward(self, x):
        self.weight.data *= self.mask
        return super(VerticallyMaskedConv2d, self).forward(x)


class VerticallyMaskedConv3d(nn.Conv3d):
    """
    TODO
    """
    def __init__(self,  in_channels, out_channels, kernel_size, bias=True):
        """
        TODO
        :param in_channels:
        :param out_channels:
        :param kernel_size:
        :param bias:
        """
        kernel_size = [int(k) for k in broadcast_to_n_dims(kernel_size, 3)]
        padding = [int(k//2) for k in kernel_size]
        super(VerticallyMaskedConv3d, self).__init__(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            bias=bias,
            padding=padding,
            padding_mode='zeros'
        )
        self.register_buffer('mask', self.weight.clone())
        kernel_height = self.weight.shape[3]
        self.mask.fill_(1)
        self.mask[:, :, kernel_height // 2 + 1:, :, :] = 0

    def forward(self, x):
        self.weight.data *= self.mask
        return super(VerticallyMaskedConv3d, self).forward(x)


class ConvAggregator(nn.Module):
    """
    Masked CNN base class, normally not initialized directly.
    See ConvAggregator2d, ConvAggregator3d
    """
    def __init__(self,
                 in_filters,
                 hidden_filters,
                 out_filters,
                 depth,
                 kernel_size,
                 activation,
                 use_norm,
                 masked_conv_layer,
                 instance_norm_layer,
                 conv_layer):
        super(ConvAggregator, self).__init__()
        if activation == 'elu':
            activation = nn.ELU
        elif activation == 'relu':
            activation = nn.ReLU
        else:
            raise ValueError("Invalid activation {}. "
                             "Must be elu/relu.".format(activation))
        self.layers = nn.ModuleList([])
        for i in range(depth):
            self.layers.append(masked_conv_layer(in_filters if not i else hidden_filters,
                                                 hidden_filters,
                                                 kernel_size))
            self.layers.append(activation())
            if use_norm:
                self.layers.append(instance_norm_layer(hidden_filters))
        self.layers.append(conv_layer(in_channels=hidden_filters,
                                      out_channels=out_filters,
                                      kernel_size=1))

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x


class ConvAggregator2d(ConvAggregator):
    def __init__(self,
                 in_filters,
                 hidden_filters,
                 out_filters,
                 depth,
                 kernel_size=3,
                 activation="elu",
                 use_norm=False):
        super(ConvAggregator2d, self).__init__(
            in_filters=in_filters,
            hidden_filters=hidden_filters,
            out_filters=out_filters,
            depth=depth,
            kernel_size=kernel_size,
            activation=activation,
            use_norm=use_norm,
            masked_conv_layer=VerticallyMaskedConv2d,
            instance_norm_layer=nn.InstanceNorm2d,
            conv_layer=nn.Conv2d,
        )


class ConvAggregator3d(ConvAggregator):
    def __init__(self,
                 in_filters,
                 hidden_filters,
                 out_filters,
                 depth,
                 kernel_size=3,
                 activation="elu",
                 use_norm=False):
        super(ConvAggregator3d, self).__init__(
            in_filters=in_filters,
            hidden_filters=hidden_filters,
            out_filters=out_filters,
            depth=depth,
            kernel_size=kernel_size,
            activation=activation,
            use_norm=use_norm,
            masked_conv_layer=VerticallyMaskedConv3d,
            instance_norm_layer=nn.InstanceNorm3d,
            conv_layer=nn.Conv3d,
        )
