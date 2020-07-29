#  ------------------------------------------------------------------------------------------
#  Copyright (c) Microsoft Corporation. All rights reserved.
#  Licensed under the MIT License (MIT). See LICENSE in the repo root for license information.
#  ------------------------------------------------------------------------------------------

import logging
import shutil
import tempfile
from pathlib import Path

import param
from typing import Optional

from azureml.core import Experiment, Model, Run
from azureml.core.workspace import WORKSPACE_DEFAULT_BLOB_STORE_NAME
from azureml.train.estimator import Estimator

from InnerEye.Azure.azure_config import AzureConfig
from InnerEye.Common.generic_parsing import GenericConfig
from score import DEFAULT_DATA_FOLDER, DEFAULT_TEST_IMAGE_NAME

RUN_MODEL = "run_model.py"


class SubmitForInferenceConfig(GenericConfig):
    """
    Command line parameter class.
    """
    experiment_name: str = param.String(default=f"model_inference",
                                        doc="Name of experiment the run should belong to")
    model_name: Optional[str] = param.String(default=None, doc="Name of model, e.g. Prostate")
    model_version: Optional[int] = param.Number(default=None, doc="Version of model, e.g. 123")
    model_id: Optional[str] = param.String(default=None, doc="Id of model, e.g. Prostate:123")
    image_file: Path = param.ClassSelector(class_=Path, doc="Image file to segment, ending in .nii.gz")
    yaml_file_path: Path = param.ClassSelector(
        class_=Path, doc="File containing subscription details, typically your train_variables.yml")

    def validate(self) -> None:
        assert self.yaml_file_path is not None
        if self.model_id is None:
            # We need at least a model name to identify a model
            assert self.model_name is not None
        elif self.model_name is not None and self.model_version is not None:
            # If all three parameters are set, they must be consistent
            assert self.model_id == f"{self.model_name}:{self.model_version}"
        # The image file must be specified, must exist, and must end in .nii.gz, i.e. be
        # a compressed Nifti file.
        assert self.image_file is not None
        if not self.image_file.exists():
            raise FileNotFoundError(self.image_file)
        basename = str(self.image_file.name)
        if not basename.endswith(".nii.gz"):
            raise ValueError(f"Bad image file name, does not end with .nii.gz: {self.image_file.name}")


def copy_image_file(image_path: Path, image_directory: Path) -> None:
    image_directory.mkdir(parents=True, exist_ok=True)
    dst_path = image_directory / DEFAULT_TEST_IMAGE_NAME
    logging.info(f"Copying {image_path} to {dst_path}")
    shutil.copyfile(str(image_path), str(dst_path))


def submit_for_inference(args: SubmitForInferenceConfig) -> Run:
    azure_config = AzureConfig.from_yaml(args.yaml_file_path)
    workspace = azure_config.get_workspace()
    model = Model(workspace=workspace, name=args.model_name, version=args.model_version, id=args.model_id)
    model_id = model.id
    source_directory = Path(tempfile.TemporaryDirectory().name)
    copy_image_file(args.image_file, source_directory / DEFAULT_DATA_FOLDER)
    for base in [RUN_MODEL, "run_scoring.py"]:
        shutil.copyfile(base, str(source_directory / base))
    environment_variables = {
        "AZUREML_OUTPUT_UPLOAD_TIMEOUT_SEC": "36000"
    }
    estimator = Estimator(
        source_directory=str(source_directory),
        entry_script=RUN_MODEL,
        script_params={"--data-folder": DEFAULT_DATA_FOLDER, "--spawnprocess": "python",
                       "--model-id": model_id, "score.py": ""},
        compute_target=azure_config.gpu_cluster_name,
        # Use blob storage for storing the source, rather than the FileShares section of the storage account.
        source_directory_data_store=workspace.datastores.get(WORKSPACE_DEFAULT_BLOB_STORE_NAME),
        inputs=[],
        environment_variables=environment_variables,
        shm_size=azure_config.docker_shm_size,
        use_docker=True,
        use_gpu=True,
    )
    exp = Experiment(workspace=workspace, name="dacart_inference")
    run = exp.submit(estimator)
    return run


def main() -> None:
    run = submit_for_inference(SubmitForInferenceConfig.parse_args())
    print(f"Submitted run {run.id} in experiment {run.experiment.name}")


if __name__ == '__main__':
    main()
