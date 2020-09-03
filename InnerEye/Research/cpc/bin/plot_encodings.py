import torch
import numpy as np
import argparse
from InnerEye.Research.cpc.pl_systems.cpc_med_encode import MEDEncoder


def get_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--pickled_encodings_path", type=str, required=True,
                        help="Path to a pickled set of encodings as output by PL system cpc_med_encode.")
    parser.add_argument("--slice_range", type=int, nargs=2, default=None,
                        help="Optional slice range of encoded patches to select for encoding. Selection is taken along "
                             "the last dimension of the encoded patch grid after a squeeze operation. E.g. if the "
                             "stored encodings are of shape [N, T, C, 10, 1, 1] with slice range 0 5 the encodings "
                             "considered will be of shape [N, T, C, 5]. If not specified, considers all patches.")
    parser.add_argument("--keep_weeks", type=int, nargs="+", default=0,
                        help="Space separated list of week integers to consider.")
    parser.add_argument("--interactive", action="store_true",
                        help="Open an interactive plot instead of saving the plot to disk.")
    parser.add_argument("--out_path", type=str, default=None,
                        help="Optional path to store the plot at on disk. Only effective when --interactive is not set.")
    return parser


def _load_slices(h5_path, slice_range=None):
    """
    Load images and segmentation Slices in a slice range along the Z-dimension from a hdf5 at 'h5_path'.
    :param h5_path: Path, path to a MED archive storing images and segmentaitons
    :param slice_range: None, or list of two intergers giving the range of Slices to return
    :return: np.ndarray of images shape [n, H, W], np.ndarray of segmentations [n, H, W] where n is the number of slices
    """
    import h5py
    with h5py.File(h5_path, "r") as in_f:
        volume = in_f["volume"]
        segmentation = in_f["segmentation"]
        if slice_range is None:
            slice_range = [int(volume.shape[0]//2), int(volume.shape[0]//2)+1]
        return volume[slice_range[0]:slice_range[1]].squeeze(), \
               segmentation[slice_range[0]:slice_range[1]].squeeze()


def load_image_slices(subjects, input_weeks, keep_week_inds=None, slice_range=None):
    print("Loading images for {} subjects...".format(len(subjects)))
    dataset = None

    images_to_return, segmentations_to_return = [], []
    for i, subject in enumerate(subjects):
        print(f"{i+1}/{len(subjects)} Loading subject: {subject}")
        h5_paths = dataset.get_h5_paths(subject=subject)
        if keep_week_inds:
            h5_paths = h5_paths[keep_week_inds]
        images, segmentations = [], []
        for path in h5_paths:
            im, seg = _load_slices(path, slice_range)
            images.append(im), segmentations.append(seg)
        images_to_return.append(np.stack(images, axis=0))
        segmentations_to_return.append(np.stack(segmentations, axis=0))
    return np.stack(images_to_return, axis=0).astype(np.float32), \
           np.stack(segmentations_to_return, axis=0).astype(np.float32),


def plot_encodings_script_func(pickled_encodings_path, slice_range=None,
                               keep_weeks=None, non_img_feature_mapping=None, interactive=False, out_path=None):
    """
    TODO

    :param pickled_encodings_path:
    :param slice_range:
    :param keep_weeks:
    :param non_img_feature_mapping:
    :param interactive:
    :return:
    """
    # Load subject: encodings dict
    encodings = MEDEncoder.load_pickled_encodings(pickled_encodings_path)
    encodings, input_weeks, target_weeks = encodings["encodings"], \
                                           encodings["input_weeks"], \
                                           encodings["target_weeks"]
    print("Loaded N subject encodings:", len(encodings.keys()))

    # Stack across subjects
    stacked_encodings = MEDEncoder.stack_encodings(encodings)
    image_features = stacked_encodings["image_features"].squeeze()
    non_img_features = stacked_encodings["non_image_features"]
    print("Stacked image features:", image_features.shape)

    if slice_range:
        # Choose slices
        image_features = image_features[..., slice_range[0]:slice_range[1]]
        print("Image features after slice selection:", image_features.shape)
    keep_week_inds = list(range(len(input_weeks)))
    if keep_weeks:
        # Choose weeks to keep
        keep_week_inds = [input_weeks.index(i) for i in keep_weeks]
        input_weeks = keep_weeks
        image_features = image_features[:, keep_week_inds, ...]
        non_img_features = non_img_features[:, keep_week_inds, ...]
        print("Image features after week selection:", image_features.shape)
    if image_features.ndim > 3:
        # Compute mean over slices
        mean_dims = range(3, image_features.ndim)
        image_features = np.mean(image_features, axis=tuple(mean_dims))
        print("Image features after slice mean:", image_features.shape)

    # Normalize features
    from InnerEye.Research.cpc.utils.training import normalize_features
    print("Normalizing features on dim 0...")
    image_features = normalize_features(torch.from_numpy(image_features), dim=0)[0].numpy()

    # Stack non-imaging features if specified
    if non_img_feature_mapping:
        print("Keeping {} non-imaging features pr. time-step".format(len(non_img_feature_mapping)))
        non_im_names, non_im_inds = zip(*list(non_img_feature_mapping.items()))
        selected_non_im = []
        for k in range(non_img_features.shape[1]):
            selected_non_im.append(non_img_features[:, k, non_im_inds])
        selected_non_im = np.stack(selected_non_im, axis=1)
        image_features = np.concatenate((image_features, selected_non_im), axis=-1)
        print("Image features with non-imagee concatenated:", image_features.shape)
    else:
        print("Using no non-imaging features")
        non_im_names = []

    # Reshape [N, S, C] array --> [N, -1]
    n_subjects, n_weeks = image_features.shape[:2]
    image_features = image_features.reshape(n_subjects, -1)
    print("Num subjects: {} (num weeks: {}))".format(n_subjects, n_weeks))
    print("Image features after subject-wise flattening:", image_features.shape)

    # Init color and marker arrays
    color = np.asarray(["Switch" if i == 1 else "No switch" for i in stacked_encodings["label"]])
    marker = np.zeros(len(color))
    title = "Weeks: {} - Targets: {}".format(input_weeks, target_weeks)

    if interactive:
        from InnerEye.Research.cpc.utils.plotting import interactive_plot_encoding_space
        # Extract images corresponding to each encoding
        images, segmentations = load_image_slices(
            subjects=stacked_encodings["subjects"],
            keep_week_inds=keep_week_inds,
            input_weeks=input_weeks,
            slice_range=None
        )
        print("Loaded images array of shape: {}".format(images.shape))
        print("Loaded segmentations array of shape: {}".format(images.shape))
        interactive_plot_encoding_space(
            encodings=image_features,
            images=images,
            segmentations=segmentations,
            subjects=stacked_encodings["subjects"],
            color_by=color,
            marker_by=marker,
            perplexity=50
        )
    else:
        # Create plot
        out_path = out_path or "weeks_{}_target_{}_slice_range_{}_non_image_{}.png".format(
            input_weeks, target_weeks, slice_range or "all", len(non_im_names)
        )
        print("Saving plot to {}".format(out_path))
        fig, ax = MEDEncoder.plot(
            encodings=image_features,
            color_by=color,
            marker_by=marker,
            title=title,
            out_path=None,
            perplexity=50
        )
        fig.suptitle("Non-image features: {}".format(non_im_names))
        fig.savefig(out_path)


def entry_func(args=None):
    args = get_parser().parse_args(args)
    plot_encodings_script_func(**vars(args))


if __name__ == "__main__":
    entry_func()
