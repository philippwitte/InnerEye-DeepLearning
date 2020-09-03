

def add_aggregator_args(parser):
    group = parser.add_argument_group("Aggregator model hyperparameters")
    group.add_argument("--use_aggregator", action="store_true",
                       help="Use an aggregator model to expand the "
                            "encoding context to include previous encodings.")
    group.add_argument("--aggregator_hidden_dim", type=int, default=96,
                       help="Number of filters in each of the hidden layers.")
    group.add_argument("--aggregator_depth", type=int, default=2,
                       help="Number of conv. layers.")
    group.add_argument("--aggregator_kernel_size", type=int, nargs="+",
                       default=[3, 1, 1],
                       help="The size of the kernels in each conv. layer")
    group.add_argument("--aggregator_use_norm", action="store_true",
                       help="Use normalization layers in conv blocks.")
