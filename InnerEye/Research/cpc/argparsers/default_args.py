

def add_default_args(parser):
    group = parser.add_argument_group("Default arguments needed for all PL systems")
    group.add_argument("--run_name", type=str, default="my_run",
                       help="An optional run name.")
    group.add_argument("--random_seed", type=int, default=1,
                       help="Set a random seed for python, numpy, torch.random, "
                            "and torch.cuda on all GPUs. Default=1.")
