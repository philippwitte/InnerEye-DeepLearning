

def add_default_training_args(parser):
    group = parser.add_argument_group("Default optimization hyperparameters")
    group.add_argument('--learning_rate', type=float,
                       default=1e-4, help='learning rate')
    group.add_argument('--batch_size', type=int,
                       default=8, help='batch size')
    group.add_argument('--num_epochs', type=int, default=400,
                       help='Number of epochs to train for.')
    group.add_argument('--resume_from', type=str, default=False,
                       help='Continue the training session from a specified'
                            ' checkpoint file.')
    group.add_argument("--amp_training", action="store_true",
                       help="Run mixed FP16 training instead of (default) FP32 training.")


def add_cpc_training_args(parser):
    group = parser.add_argument_group("CPC optimization hyperparameters")
    group.add_argument('--negative_samples', type=int,
                       default=128, help='Number of negative samples pr. '
                                         'positive sample.')
    group.add_argument("--no_patch_augmentation", action="store_true",
                       help="Do not apply patch-wise augmentation.")
    group.add_argument("--use_memory_bank", action="store_true",
                       help="Sample negatives from a memory bank of encodings cached "
                            "through training. This allows for larger-than-batch diversity "
                            "in the sampled negatives, at the cost of stale features. Gradients are not "
                            "computed for negatives sampled from the memory bank.")


def add_multi_view_cpc_training_args(parser):
    group = parser.add_argument_group("CPC hyperparameters for multi-view CPC models")
    group.add_argument("--negative_sampling_mode", type=str, choices=["random", "nearest"], default="random",
                       help="Negative sampling mode. Must be one of {'random', 'nearest'} (default='random')")
    group.add_argument("--negatives_exclude_weeks_within_dist", type=int, default=1,
                       help="When sampling negatives, exclude patches that are <= negatives_exclude_weeks_within_dist "
                            "weeks away from the reference. E.g. with 0 exclude same week, with 1 exclude same, "
                            "previous and next week (if available). Set a negative value to exclude nothing.")
    group.add_argument("--negatives_exclude_spatial_within_dist", type=int, default=3,
                       help="When sampling negatives, exclude patches that are <= negatives_exclude_spatial_within_dist"
                            " spatial steps away from the reference. E.g. with 0 exclude same week, with 1 exclude the "
                            "reference patch itself, with 1 exclude the reference, previous and next patches "
                            "(if available) in all spatial directions. Set a negative value to exclude nothing.")
    group.add_argument("--tau", type=float, default=1.0,
                       help="Tau temperature scaling parameter for typical negative samples.")
    group.add_argument("--tau_for_excluded_negatives", type=float, default=False,
                       help="If set and --negatives_exclude_weeks_within_dist and "
                            "--negatives_exclude_spatial_within_dist define a set of negatives that would normally "
                            "be excluded entirely, included also those samples but with a different "
                            "temperature/margin parameter --tau_for_excluded_negatives. E.g. one may include those "
                            "samples but with a less strict margin requirement (larger tau value).")


def add_directional_cpc_training_args(parser):
    group = parser.add_argument_group("Directional CPC optimization hyperparameters")
    group.add_argument('--k_prediction_steps', type=int, nargs="+",
                       default=[5],
                       help="Number of time steps in the future "
                            "(rows below for images) to predict. "
                            "Must be a single integer, or 1 integer for each "
                            "direction (see --directions).")
    group.add_argument('--num_skip_steps', type=int, nargs="+",
                       default=[0],
                       help="Number of time steps in the future to skip. "
                            "This is usually 1 when patches are overlapping. "
                            "Must be a single integer, or 1 integer for each "
                            "direction (see --directions).")
    group.add_argument("--directions", type=str, nargs="+",
                       default=["down", "up"],
                       choices=["down", "up", "left", "right",
                                "ahead", "backward"],
                       help="Specify one or more spatial directions in the"
                            " images along which to perform CPC "
                            "(defaults to up/down along first spatial dimension)")
    group.add_argument("--score_model_depth", type=int, default=2,
                       help="Number of layers in the CPC score/projection "
                            "models. Default is 1 (a linear projection "
                            "model).")
    group.add_argument("--no_unit_sphere_norm", action="store_true",
                       help="Do not apply unit sphere normalization before InfoNCELoss "
                            "computation.")


def add_concurrent_downstream_clf_args(parser):
    group = parser.add_argument_group("Arguments for training a linear classifier "
                                      "concurrently to CPC training for performance "
                                      "evaluation.")
    group.add_argument("--classifier_train_every", type=int, default=1,
                       help="Train a downstream linear classifier every N epochs "
                            "while training the CPC encoder to evaluate its "
                            "performance. Defaults to 0, i.e. not used.")
    group.add_argument("--classifier_train_on_additional_features", type=str, nargs="+",
                       default=["Parameter_1"],
                       help="Space separated list of column names describing additional features "
                            "to cache through training and use for training the downstream classifier. ")


def add_downstream_clf_training_args(parser):
    group = parser.add_argument_group("Down-stream classifier hyperparameters")
    group.add_argument("--cpc_model_ckpt", type=str, default="cpc.ckpt",
                       help="Path to a .ckpt file storing parameters and "
                            "hyperparameters of the CPC trained encoder.")
    group.add_argument("--random_init", action="store_true",
                       help="Use an encoder with ranomly initialized weights "
                       "(do not use the CPC learned weights)")
    group.add_argument("--n_classes", type=int, default=10,
                       help="Number of classes to predict.")
    group.add_argument("--finetune_encoder", action="store_true",
                       help="If set, update the parameters of the encoder.")
