

def add_default_data_args(parser):
    group = parser.add_argument_group("Data input/output arguments")
    group.add_argument('--data_folder', type=str, default='./data',
                       help='Folder to store/read data to/from.')
    group.add_argument('--download_data', action='store_true',
                       help='Download the data.')
    group.add_argument('--input_channels', type=int, default=1,
                       help='Number of channels in the input image.')
    group.add_argument('--num_workers', type=int, default=6,
                       help='Number of worker processes to use for loading '
                            '(each of) training and validation data batches.')
    group.add_argument("--train_data_fraction", type=float, default=1.0,
                       help="Fraction of the training data to use for "
                            "training the classifier (random samples)")


def add_med_default_data_args(parser):
    group = parser.add_argument_group("Data input/output arguments")
    group.add_argument('--input_channels', type=int, default=1,
                       help='Number of channels in the input image.')
    group.add_argument('--num_workers', type=int, default=8,
                       help='Number of worker processes to use for loading '
                            '(each of) training and validation data batches.')
    group.add_argument("--csv_file_name", type=str,
                       default="mt76562w_21APR20_comma.csv",
                       help="Name of dataset database file. Must be stored at --hdf5_files_path.")
    group.add_argument("--hdf5_files_path", type=str,
                       default="F:\\data\\20200303-130219_processed",
                       help="Path to folder storing  HDF5 image files")
    group.add_argument("--azure_dataset_id", type=str, default="20200303-130219_processed",
                       help="The string identifier of the dataset to mount iff running on"
                            "azure (see cpc.py --aml flag).")
    group.add_argument("--azure_conda_env_path", type=str, default="environment.yaml",
                       help="Path to a conda environment file to use for iff training on azure"
                            " (see cpc.py --aml flag).")
    group.add_argument("--normalization_level", type=str, choices=["Slice", "Volume"], default="Slice",
                       help="Normalize MED images on a (channel-wise) Slice or Volume level.")
    group.add_argument("--train_split_fraction", type=float, default=1.0,
                       help="Fraction of the dataset to use for training, the rest will be used for validation. "
                            "Set this to 1.0 if no validation is to be performed.")


def add_med_data_selection_args(parser):
    group = parser.add_argument_group("Control the data loaded from MED datasets")
    group.add_argument("--input_weeks", type=int, nargs="+", default=[0, 1],
                       help="Integer valued weeks to load values for")
    group.add_argument("--target_weeks", type=int, nargs="+", default=[1],
                       help="Integer valued week(s) to load target values for.")
    group.add_argument("--non_sequential_inputs", action="store_true",
                       help="If set, do not sample all images from each subject as 1 sample (sequantial inputs), "
                            "but rather consider all images individual samples. OBS: Currently, only images and "
                            "segmentations will be available in each batch when --non_sequential_inputs is set.")


def add_cpc_data_args(parser):
    group = parser.add_argument_group("CPC input (patching) arguments")
    group.add_argument('--input_image_size', type=int, nargs="+",
                       default=[35, 260, 484],
                       help="Size of patches to extract from the full images."
                            " Padding and/or cropping may be applied "
                            "(determined by the augmentation "
                            "configuration for the specific module).")
    group.add_argument("--patch_sizes", type=int, nargs="+",
                       default=[1, 260, 484],
                       help="Size of patches input to the encoder model.")
    group.add_argument("--patch_strides", type=int, nargs="+",
                       default=[1, 1, 1],
                       help="The striding of the sliding patch window. "
                            "--patch_sizes and --patch_strides in combination"
                            " determines the total number of patches to "
                            "extract along each dim of an image of size "
                            "--input_image_size.")


def add_dual_view_data_args(parser):
    group = parser.add_argument_group("Dual view (image + segmentation) specific data args.")
    group.add_argument("--input_slice_slice_range", type=int, nargs=2, default=[7, 32],
                       help="Two integer numbers specifying the range of Slices to use from each "
                            "MED image. Range is 0-indexed and open-ended (a, b gives slices of "
                            "inds [a, ..., b-1]. NOTE: Slice cropping is applied BEFORE crops "
                            "depending on the --input_image_size flag.")
    group.add_argument("--add_slice_distance_transform", action="store_true",
                       help="Concatenate a channel to each Slice storing an EDF "
                            "transform the center slice. OBS: This increases the input channel "
                            "depth by 1 (remember to update --input_channels flag).")
