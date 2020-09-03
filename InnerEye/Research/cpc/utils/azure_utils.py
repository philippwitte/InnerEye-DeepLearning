import logging
import os
import shutil
from pathlib import Path
from argparse import Namespace

class TRAIN_VARIABLES_YAML:
    var1 = None

def get_azure_config():
    from InnerEye.Azure.azure_config import AzureConfig
    from InnerEye.Research.cpc.bin.cpc import read_yaml_config
    azure_config = AzureConfig(**vars(read_yaml_config(TRAIN_VARIABLES_YAML))["variables"])
    return azure_config


def get_mounted(run_context):
    if hasattr(run_context, "input_datasets"):
        from InnerEye.Azure.azure_runner import INPUT_DATA_KEY
        return Path(run_context.input_datasets[INPUT_DATA_KEY])
    else:
        return None


def get_cached_or_download(run_context, hdf5_files_path, azure_dataset_id):
    if azure_dataset_id and (hdf5_files_path / azure_dataset_id).is_dir():
        hdf5_files_path = hdf5_files_path / azure_dataset_id
    logging.info("Looking for dataset at path: {}".format(hdf5_files_path))
    from InnerEye.ML.run_ml import download_dataset
    config = Namespace(azure_dataset_id=azure_dataset_id,
                       local_dataset=hdf5_files_path)
    try:
        hdf5_files_path = download_dataset(
            run_context,
            azure_config=get_azure_config(),
            config=config,
            dataset_path=hdf5_files_path
        )
    except Exception as e:
        logging.error("Could not mount or download dataset due to error: {}".format(str(e)))
    return hdf5_files_path


def mount_or_download_dataset(hdf5_files_path=None, azure_dataset_id=None):
    hdf5_files_path = Path(hdf5_files_path)
    # Get mounted or local dataset path
    from azureml.core import Run
    run_context = Run.get_context()
    dataset_path = get_mounted(run_context)
    if not dataset_path:
        if hdf5_files_path:
            dataset_path = hdf5_files_path
            if azure_dataset_id and (dataset_path / azure_dataset_id).exists():
                dataset_path = dataset_path / azure_dataset_id
        else:
            dataset_path = get_cached_or_download(run_context, hdf5_files_path, azure_dataset_id)
    logging.info("Continuing with dataset path: {}".format(dataset_path))
    logging.info("Entries at path: {}".format(len(os.listdir(dataset_path))))
    return dataset_path


def download_run(run_recovery_id, output_path, download=("logs", "outputs", "config.yaml")):
    """
    Download files and/or folders from an AML Run instance
    :param run_recovery_id:
    :param output_path:
    :param download:
    :return:
    """
    from InnerEye.Azure.azure_util import fetch_run
    from azureml.exceptions import UserErrorException
    output_path = Path(output_path)
    workspace = get_azure_config().get_workspace()
    run = fetch_run(workspace=workspace,
                    run_recovery_id=run_recovery_id)
    logging.info(f"Fetched workspace: {workspace}")
    logging.info(f"Fetched run: {run}")

    for file_or_folder in download:
        logging.info(f"Downloading file or folder '{file_or_folder}'")
        try:
            if "." in file_or_folder:
                # Assume single file
                run.download_file(file_or_folder,
                                  output_file_path=output_path/file_or_folder)
            else:
                run.download_files(prefix=file_or_folder,
                                   output_directory=output_path,
                                   append_prefix=True)
        except UserErrorException as e:
            logging.error(f"Could not download file/folder {file_or_folder}, error: {e.message}")


def add_file_or_folder_to_aml_root_folder(input_path, root_folder, overwrite=True):
    """
    Copies a file or folder at path input_path to root_folder.
    The file or folder will be stored at root_folder/[input_file_or_folder_name]
    If the target path exists it will be overwritten if overwrite=True or an OSError will be raised.

    :return: None
    """
    input_path = Path(input_path).absolute()
    root_folder = Path(root_folder).absolute()
    root_folder.mkdir(parents=True, exist_ok=True)
    out_path = root_folder / input_path.name
    if input_path.is_file():
        if out_path.exists() and not overwrite:
            raise OSError(f"A file of name {input_path.name} already exists at root_folder {root_folder}")
        else:
            shutil.copy(input_path, out_path)
    else:
        if out_path.exists():
            if not overwrite:
                raise OSError(f"A folder of name {input_path.name} already exists "
                              f"at root_folder {root_folder}")
            else:
                shutil.rmtree(out_path)
        shutil.copytree(input_path, out_path)


def prepare_resume_from_aml(local_ckpt_path, root_folder, temp_dir_name="tmp_aml_inputs"):
    """
    Copies a local checkpoint file into the root folder and returns a relative path to the
    checkpoint inside the root folder relative to the root folder. If the root folder
    is uploaded to AML, the relative path can then be used to access the checkpoint in AML.

    :return: Relative path to check-point file within the root folder
    """
    local_ckpt_path = Path(local_ckpt_path)
    root_folder = Path(root_folder)
    target_folder = root_folder / temp_dir_name
    if target_folder.exists():
        # Remove potential files inside the folder
        for fname in os.listdir(target_folder):
            os.remove(target_folder/fname)
    add_file_or_folder_to_aml_root_folder(local_ckpt_path, target_folder)
    return Path(temp_dir_name) / local_ckpt_path.name
