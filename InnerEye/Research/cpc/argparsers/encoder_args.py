

def add_encoder_args(parser):
    group = parser.add_argument_group("ResnetEncoder model hyperparameters")
    group.add_argument('--encoder_init_dim', type=int, default=7,
                       help="Number of filters in the initial encoder "
                            "ResNet block.")
    group.add_argument('--encoder_out_dim', type=int, default=48,
                       help="Number of filters in the final encoder "
                            "layer (output encoding dim).")
    group.add_argument('--encoder_res_block_depth', type=int, default=1,
                       help="Depth of each residual block.")
    group.add_argument('--encoder_use_norm', action='store_true',
                       help="Use normalization layers in conv blocks")


def add_segmentation_encoder_args(parser):
    group = parser.add_argument_group("Segmentation ResnetEncoder model hyperparameters")
    group.add_argument('--segmentation_encoder_input_channels', type=int, default=10,
                       help="Number of segmentation maps that the encoder should accept.")
    group.add_argument('--segmentation_encoder_init_dim', type=int, default=7,
                       help="Number of filters in the initial encoder "
                            "ResNet block.")
    group.add_argument('--segmentation_encoder_out_dim', type=int, default=48,
                       help="Number of filters in the final encoder "
                            "layer (output encoding dim).")
    group.add_argument('--segmentation_encoder_res_block_depth', type=int, default=1,
                       help="Depth of each residual block.")
    group.add_argument('--segmentation_encoder_use_norm', action='store_true',
                       help="Use normalization layers in conv blocks")
