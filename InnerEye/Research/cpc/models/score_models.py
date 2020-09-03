from torch import nn


class ScoreModel(nn.Module):
    """
    A simple MLP projecting encodings from D to D dimensional space linearly
    or non-linearly (with n_layers > 1). Dimensionality is fixed throughout the
    layers.

    Note: Implemented as point 1D convolutions. This allows the score model
    to be applied across an input of encoded patches etc.
    """
    def __init__(self, in_channels, n_layers=1, hidden_activation="elu"):
        super().__init__()
        if hidden_activation == 'elu':
            hidden_activation = nn.ELU
        elif hidden_activation == 'relu':
            hidden_activation = nn.ReLU
        else:
            raise ValueError("Invalid hidden_activation {}. "
                             "Must be elu/relu.".format(hidden_activation))
        modules = []
        for i, layer in enumerate(range(n_layers)):
            modules.append(nn.Conv1d(in_channels, in_channels, 1, bias=False))
            if i < (n_layers-1):
                # Not last layer, add hidden activation
                modules.append(hidden_activation())
        self.model = nn.Sequential(*modules)

    def forward(self, x_enc):
        return self.model(x_enc)
