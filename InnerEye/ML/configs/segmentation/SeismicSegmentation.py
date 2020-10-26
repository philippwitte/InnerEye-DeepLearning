#  ------------------------------------------------------------------------------------------
#  Copyright (c) Microsoft Corporation. All rights reserved.
#  Licensed under the MIT License (MIT). See LICENSE in the repo root for license information.
#  ------------------------------------------------------------------------------------------
from random import Random
from typing import Any

from azureml.train.estimator import Estimator
from azureml.train.hyperdrive import BanditPolicy, HyperDriveConfig, PrimaryMetricGoal, RandomParameterSampling, uniform
from networkx.tests.test_convert_pandas import pd

from InnerEye.ML.common import TrackedMetrics
from InnerEye.ML.config import PhotometricNormalizationMethod, SegmentationModelBase, SegmentationLoss, equally_weighted_classes
from InnerEye.ML.utils.model_metadata_util import generate_random_colours_list
from InnerEye.ML.utils.split_dataset import DatasetSplits
from Tests.fixed_paths_for_tests import full_ml_test_data_path
from pathlib import Path
from InnerEye.ML.deep_learning_config import OptimizerType

class SeismicSegmentation(SegmentationModelBase):
    """
    Seismic image segmentation example.
    """

    def __init__(self, **kwargs: Any) -> None:
        fg_classes = ["basement", "slope_mudstone_a", "mass_transport_deposit", "slope_mudstone_b", "slope_valley", "submarine_canyon"]
        super().__init__(
            # Data definition - in this section we define where to load the dataset from
            #local_dataset=full_ml_test_data_path(),
            local_dataset=Path('/home/pwitte/SeismicSegmentationChallenge'),
            #azure_dataset_id='parihaka_patch_size_128',
            
            # Model definition - in this section we define what model to use and some related configurations
            architecture="UNet3D",
            feature_channels=[2],
            crop_size=(64, 64, 64),
            image_channels=["image"],
            ground_truth_ids=fg_classes,
            class_weights=equally_weighted_classes(fg_classes, background_weight=0.02),
            #mask_id="mask",

            # Model training and testing - in this section we define configurations pertaining to the model
            # training loop (ie: batch size, how many epochs to train, number of epochs to save)
            # and testing (ie: how many epochs to test)
            use_gpu=False,
            num_dataload_workers=0,
            train_batch_size=4,
            inference_batch_size=1,
            start_epoch=0,
            num_epochs=100,
            save_start_epoch=20,
            save_step_epochs=20,
            test_start_epoch=50,
            # test_diff_epochs=1,
            test_step_epochs=50,
            use_mixed_precision=True,

            # Optimizer
            loss_type=SegmentationLoss.SoftDice,
            l_rate=1e-3,
            min_l_rate=1e-5,
            l_rate_polynomial_gamma=0.9,
            optimizer_type=OptimizerType.Adam,
            opt_eps=1e-4,
            adam_betas=(0.9, 0.999),
            momentum=0.9,

            # Pre-processing - in this section we define how to normalize our inputs, in this case we are doing
            # CT Level and Window based normalization.
            norm_method=PhotometricNormalizationMethod.SimpleNorm,
            level=50,
            window=200,

            # Post-processing - in this section we define our post processing configurations, in this case
            # we are filling holes in the generated segmentation masks for all of the foreground classes.
            fill_holes=[True] * len(fg_classes),

            # Output - in this section we define settings that determine how our output looks like in this case
            # we define the structure names and colours to use.
            ground_truth_ids_display_names=fg_classes,
            colours=generate_random_colours_list(Random(5), len(fg_classes)),
        )
        self.add_and_validate(kwargs)


    def get_model_train_test_dataset_splits(self, dataset_df: pd.DataFrame) -> DatasetSplits:
        return DatasetSplits.from_proportions(dataset_df, proportion_train=0.8, proportion_val=0.1,
                                              proportion_test=0.1,
                                              random_seed=0)
