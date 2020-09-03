import argparse
from .default_args import add_default_args
from .data_args import (add_default_data_args, add_cpc_data_args,
                        add_med_default_data_args, add_med_data_selection_args,
                        add_dual_view_data_args)
from .encoder_args import add_encoder_args, add_segmentation_encoder_args
from .training_args import (add_default_training_args, add_cpc_training_args,
                            add_directional_cpc_training_args,
                            add_concurrent_downstream_clf_args,
                            add_multi_view_cpc_training_args,
                            add_downstream_clf_training_args)
from .aggregator_args import add_aggregator_args
from .logger_args import add_logger_args


def cpc_stl10_argparser():
    """
    Defines an argparser for STL10 CPC (patch-based) experiments
    """
    parser = argparse.ArgumentParser()
    add_default_args(parser)
    add_logger_args(parser)
    add_default_data_args(parser)
    add_cpc_data_args(parser)
    add_encoder_args(parser)
    add_default_training_args(parser)
    add_cpc_training_args(parser)
    add_directional_cpc_training_args(parser)
    add_aggregator_args(parser)
    return parser


def cpc_med_argparser():
    """
    Defines an argparser for MED CPC (patch-based) experiments
    """
    parser = argparse.ArgumentParser()
    add_default_args(parser)
    add_logger_args(parser)
    add_med_default_data_args(parser)
    add_med_data_selection_args(parser)
    add_cpc_data_args(parser)
    add_encoder_args(parser)
    add_default_training_args(parser)
    add_concurrent_downstream_clf_args(parser)
    add_cpc_training_args(parser)
    add_directional_cpc_training_args(parser)
    add_aggregator_args(parser)
    return parser


def cpc_dual_view_med_argparser():
    """
    Defines an argparser for MED CPC (dual-view) experiments
    """
    parser = argparse.ArgumentParser()
    add_default_args(parser)
    add_logger_args(parser)
    add_med_default_data_args(parser)
    add_med_data_selection_args(parser)
    add_cpc_data_args(parser)
    add_encoder_args(parser)
    add_segmentation_encoder_args(parser)
    add_dual_view_data_args(parser)
    add_default_training_args(parser)
    add_concurrent_downstream_clf_args(parser)
    add_cpc_training_args(parser)
    add_multi_view_cpc_training_args(parser)
    return parser


def cpc_tri_view_med_argparser():
    """
    Defines an argparser for MED CPC (triple-view) experiments
    """
    # Same parameters for now
    return cpc_dual_view_med_argparser()


def cpc_downstream_clf_argparser():
    """
    Defines an argparser for general downstream classification tasks
    """
    parser = argparse.ArgumentParser()
    add_default_args(parser)
    add_logger_args(parser)
    add_downstream_clf_training_args(parser)
    add_default_data_args(parser)
    add_default_training_args(parser)
    return parser


def med_dual_view_supervised():
    """
    Defines an argparser for MED dual-view supervised experiments
    """
    parser = argparse.ArgumentParser()
    add_default_args(parser)
    add_logger_args(parser)
    add_med_default_data_args(parser)
    add_med_data_selection_args(parser)
    add_cpc_data_args(parser)
    add_encoder_args(parser)
    add_segmentation_encoder_args(parser)
    add_dual_view_data_args(parser)
    add_default_training_args(parser)
    return parser


def cpc_med_encode_argparser():
    """
    Defines an argparser for MED CPC encoding (applying trained MED CPC modules to data)
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("--encodings_out_path", type=str, required=True,
                        help="String path to where encodings will be stored.")
    parser.add_argument("--cpc_model_ckpt", type=str, required=True,
                        help="Path to a .ckpt file storing parameters and "
                             "hyperparameters of the CPC trained encoder.")
    parser.add_argument("--apply_mean_pooling", action="store_true",
                        help="Apply mean pooling over the spatial grid dims before storing the encodings.")

    # Define hparams @ ckpt file overwrites group
    overwrites = parser.add_argument_group("Optional arguments to specify to overwrite those stored in the hparams "
                                           "object of the loaded model ckpt file (--cpc_model_ckpt)")
    overwrites.add_argument("--csv_file_name", type=str, required=False,
                            help="Name of dataset database file. Must be stored at --hdf5_files_path.")
    overwrites.add_argument("--hdf5_files_path", type=str, required=False,
                            help="Path to folder storing  HDF5 image files")
    overwrites.add_argument("--input_weeks", type=int, nargs="+", required=False,
                            help="Integer valued weeks to load values for")

    # Default run args
    add_default_args(parser)
    add_logger_args(parser)
    return parser
