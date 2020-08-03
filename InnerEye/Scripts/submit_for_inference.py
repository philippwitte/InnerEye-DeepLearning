#  ------------------------------------------------------------------------------------------
#  Copyright (c) Microsoft Corporation. All rights reserved.
#  Licensed under the MIT License (MIT). See LICENSE in the repo root for license information.
#  ------------------------------------------------------------------------------------------

import logging
import shutil
import tempfile
import requests
from pathlib import Path

import param
from typing import List, Optional

from azureml.core import Experiment, Model, Run

from InnerEye.Azure.azure_config import AzureConfig, SourceConfig
from InnerEye.Azure.azure_runner import create_estimator_from_configs
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
    yaml_file: Path = param.ClassSelector(
        class_=Path, doc="File containing subscription details, typically your train_variables.yml")

    def validate(self) -> None:
        assert self.yaml_file is not None
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


def copy_image_file(image: Path, image_directory: Path) -> None:
    image_directory.mkdir(parents=True, exist_ok=True)
    dst = image_directory / DEFAULT_TEST_IMAGE_NAME
    logging.info(f"Copying {image} to {dst}")
    shutil.copyfile(str(image), str(dst))


def download_conda_dependency_files(model: Model, dir_path: Path) -> List[Path]:
    url_dict = model.get_sas_urls()
    downloaded: List[Path] = []
    for path, url in url_dict.items():
        if Path(path).name == "environment.yml":
            tgt_path = dir_path / f"tmp_environment_{len(downloaded) + 1:03d}.yml"
            with tgt_path.open('wb') as out:
                out.write(requests.get(url, allow_redirects=True).content)
            print(f"Downloaded {tgt_path} from {url}")
            downloaded.append(tgt_path)
    return downloaded


def submit_for_inference(args: SubmitForInferenceConfig) -> Run:
    azure_config = AzureConfig.from_yaml(args.yaml_file)
    workspace = azure_config.get_workspace()
    model = Model(workspace=workspace, name=args.model_name, version=args.model_version, id=args.model_id)
    model_id = model.id
    source_directory_name = tempfile.TemporaryDirectory().name
    source_directory_path = Path(source_directory_name)
    copy_image_file(args.image_file, source_directory_path / DEFAULT_DATA_FOLDER)
    for base in [RUN_MODEL, "run_scoring.py", "score.py"]:
        shutil.copyfile(base, str(source_directory_path / base))

    source_config = SourceConfig(
        root_folder=source_directory_name,
        entry_script=str(source_directory_path / RUN_MODEL),
        script_params={"--data-folder": DEFAULT_DATA_FOLDER, "--spawnprocess": "python",
                       "--model-id": model_id, "score.py": ""},
        conda_dependencies_files=download_conda_dependency_files(model, source_directory_path)
    )
    estimator = create_estimator_from_configs(workspace, azure_config, source_config, [])
    exp = Experiment(workspace=workspace, name=args.experiment_name)
    run = exp.submit(estimator)
    return run


def main() -> None:
    logging.getLogger().setLevel(logging.INFO)
    run = submit_for_inference(SubmitForInferenceConfig.parse_args())
    logging.info(f"Submitted run {run.id} in experiment {run.experiment.name}")
    logging.info(f"Run URL: {run.get_portal_url()}")


if __name__ == '__main__':
    main()
