"""
Entry script redirecting all command line arguments to a specified PL system
script from the cpc.pl_systems folder.

Usage:
cpc [--init_yaml_config] [yaml_path] [--from_yaml_config] [yaml_path]
    [pl system] [system script args...]
"""

import os
import argparse
import shutil
import logging
from InnerEye.Research.cpc.pl_systems import PL_MODULES


# Get list of names of available lightning modules
AVAILABLE_MODULES = list(PL_MODULES.keys())


def get_parser():
    ids = "InnerEye Research CPC entry script"
    sep = "-" * len(ids)
    usage = ("cpc [--init_yaml_config] [yaml_path] [--from_yaml_config] [yaml_path] [--aml] "
             "[--pl_system] [pl system] "
             "[pl system script args...]\n\n"
             "%s\n%s\n"
             "Available PyTorch lightning systems:\n- %s") % \
            (ids, sep, "\n- ".join(AVAILABLE_MODULES))

    # Top level parser
    parser = argparse.ArgumentParser(usage=usage)
    parser.add_argument("--init_yaml_config", type=str, required=False,
                        help="Init a default configuration file at the path,"
                             " then exists.")
    parser.add_argument("--from_yaml_config", type=str, required=False,
                        help="A path to a yaml configuration file. If "
                             "specified, all command line arguments are read "
                             "from this file.")
    parser.add_argument("--aml", action="store_true",
                        help="Submit this job to AML instead of running locally.")
    parser.add_argument("--pl_system", type=str,
                        help="Name of the lightning system to run.",
                        choices=AVAILABLE_MODULES,
                        required=True)
    return parser


def namespace_to_yaml_file(namespace, yaml_file_path):
    """
    Takes a Namespace object of parameters and saves the key:value pairs to
    a yaml formatted file at yaml_file_path.
    :param namespace:
    :param yaml_file_path:
    :return:
    """
    args = vars(namespace)
    yamls = [f"{arg}: {default if default is not None else 'Null'}"
             for arg, default in args.items()]
    with open(yaml_file_path, "w") as out_file:
        out_file.write("\n".join(yamls))


def init_yaml_config(yaml_path, argparser):
    """
    Save a YAML formatted file to path 'yaml_path' with each entry being the
    argument name : default value of an input argparser 'argparser'.

    Parameters
    ----------
    yaml_path : str
        Path to the output yaml file
    argparser : Argparser
        Argument parser from which configuration file is created

    Returns
    -------
    None
    """
    args = argparser.parse_args([])
    namespace_to_yaml_file(namespace=args,
                           yaml_file_path=yaml_path)


def read_yaml_config(yaml_path):
    """
    Reads parameters from a YAML formatted configuration file and returns a
    Namespace object for parameter lookup.

    Parameters
    ----------
    yaml_path : str
        Path to the yaml formatted file to load parameters from

    Returns
    -------
    A Namespace object
    """
    import yaml
    from argparse import Namespace
    with open(yaml_path, "r") as in_file:
        args = Namespace(**yaml.safe_load(in_file))
    return args


def extract_args_for_aml(args):
    """
    Takes a Namespace of arguments as parsed by a argparser.Argparser instance, returns a dictionary of arguments
    --{arg_key}:{arg_value} with True arg_value replaced by None values and arg_key with False arg_value ignored.
    :param args:
    :return: dict of arguments
    """
    extracted = {}
    for key, value in vars(args).items():
        if value is True:
            value = None
        if value is False:
            continue
        extracted[f"--{key}"] = value
    return extracted


def run_in_aml(pl_system, conda_dependencies_path, script_args, azure_dataset_id):
    """
    Submit this run to AML. Copies the cpc.py entry script into the main repository root context and submits
    a job calling this same script with identical arguments (with some overwrite exceptions) in AML.

    :param pl_system: str, the name of the PL-system to run.
    :param conda_dependencies_path: Path, A path to a environment.yaml file from which the AML environment is setup
    :param script_args: Namespace, arguments to pass to the PL-system script
    :param azure_dataset_id: str, the dataset ID in AML to launch against
    :return: Azure Run instance and run status
    """
    from InnerEye.ML.runner import submit_to_azureml
    from InnerEye.Azure.azure_config import SourceConfig
    from InnerEye.Research.cpc.utils.azure_utils import get_azure_config, prepare_resume_from_aml
    from pathlib import Path

    # Get entry script path (this file) and copy it to the root folder
    root_folder = Path('.')
    entry_script_path = Path(__file__).absolute()
    entry_script_target_path = root_folder / entry_script_path.name
    shutil.copy(entry_script_path, entry_script_target_path)

    # Add / overwrite entry parameters to script_params
    script_args.pl_system = pl_system
    if hasattr(script_args, "hdf5_files_path"):
        # Overwrite potentially local path.
        script_args.hdf5_files_path = "datasets"

    if script_args.resume_from:
        # Copy the parameters to a temporary directory within the root folder.
        # Set relative path to weights from root folder
        script_args.resume_from = str(prepare_resume_from_aml(script_args.resume_from, root_folder))

    # Create config at root folder for reference in Run context
    namespace_to_yaml_file(script_args, root_folder / "config.yaml")

    # Setup SourceConfig and submit to AML
    azure_config = get_azure_config()
    azure_config.user_friendly_name = script_args.run_name
    source_config = SourceConfig(
        root_folder=root_folder,
        entry_script=entry_script_target_path,
        conda_dependencies_file=os.path.abspath(conda_dependencies_path),
        hyperdrive_config_func=None,
        upload_timeout_seconds=86400
    )
    source_config.script_params = extract_args_for_aml(script_args)
    azure_run = submit_to_azureml(
        azure_config=azure_config,
        source_config=source_config,
        model_config_overrides="",
        azure_dataset_id=azure_dataset_id
    )
    return azure_run, azure_run.get_status()


def run_locally(modname, script_args, logger):
    """
    Imports a PL-system and invokes its entry_script function passing script arguments and a logger instance.

    :param modname: string, a module to import (e. g. "InnerEye.Research.cpc.pl_systems.cpc_dual_view_med_train")
    :param script_args: Namespace, arguments to be passed to the PL-system entry_script
    :param logger: AMLTensorBoardLogger logger instance.
    """
    # Import the script, call entry_func with selected arguments
    import importlib
    mod = importlib.import_module(modname)
    mod.entry_func(args=script_args, logger=logger)


def prepare_run(script_args, pl_system):
    """
    Helper called immediately prior to invoking the PL-system's entry_script function.
    Sets up logging and creates folders in the local run-context. Sets a random-seed. May overwrite or modify
    parameters in script_args if the run is in AML or non.

    :param script_args: Namespace of arguments to be passed to the PL-system script.
    :param pl_system: string, the PL system being executed (e. g. "cpc_dual_view_med_train")
    :return: A AMLTensorBoardLogger logger instance.
    """
    # Setup logging and output dirs
    from InnerEye.Research.cpc.utils.system import make_dirs
    from InnerEye.ML.utils.ml_util import set_random_seed
    from InnerEye.Research.cpc.utils.logging_utils import configure_logging, get_default_logger
    from InnerEye.Research.cpc.utils.callbacks import CheckpointConfig
    script_args.pl_system = pl_system  # Add reference to which system was executed
    make_dirs(["outputs", "logs"])
    log_level = getattr(script_args, "log_level", "INFO")
    configure_logging(log_path=os.path.join("logs", "log.txt"),
                      log_level=log_level)
    logging.getLogger().setLevel(log_level)
    logger = get_default_logger(script_args)
    if logger.aml_run:
        # On AML, if any checkpoint is already stored, this means that the container was pre empted.
        # We take the latest version here and continue from that no matter the script_args.resume_from flag.
        script_args.resume_from = CheckpointConfig.get_latest_ckpt_path() or script_args.resume_from
    logging.info(f"Args dump: {script_args}")
    set_random_seed(script_args.random_seed)
    return logger


def parse_args_helper():
    """
    Parses arguments in sys.argv[1:] by known cpc.py entry-script args and leaves other arguments aside for
    separate PL-system-specific parsing.

    -h or --help arguments are handled specifically. If a help argument is passed without a --pl_system argument
    specified, the cpc.py entry script's help page is shown. Otherwise, if a PL-system is also specified, the help
    flag is passed to the script-specific parser for showing the PL-system help page.
    """
    import sys
    args = list(sys.argv[1:])
    help_ = None
    if "-h" in args:
        help_ = args.pop(args.index("-h"))
    elif "--help" in args:
        help_ = args.pop(args.index("--help"))
    if "--pl_system" not in args:
        args += [help_]
        help_ = None
    entry_args, non_parsed_script_args = get_parser().parse_known_args(args)
    if help_:
        non_parsed_script_args += [help_]
    return entry_args, non_parsed_script_args


def entry_func():
    """
    Entry function for all CPC PL-system modules. Call this function with command-line arguments as per the
    cpc.py entry script argparser (see cpc.py --help) and PL-system argparser selected with argument --pl_system.
    """
    # Parse the cpc.py entry script args, leave all script-specific args aside for now
    entry_args, non_parsed_script_args = parse_args_helper()

    # Fetch module name and its associated argparser function name
    modname, argparser_func_name = PL_MODULES[entry_args.pl_system]

    # Import and init the argparser
    from InnerEye.Research.cpc import argparsers
    script_arg_parser = getattr(argparsers, argparser_func_name)()
    if entry_args.init_yaml_config:
        init_yaml_config(entry_args.init_yaml_config, script_arg_parser)
    else:
        if entry_args.from_yaml_config:
            if non_parsed_script_args:
                raise RuntimeError("Should not pass command line arguments to "
                                   "the script when loading parameters from "
                                   "conf file (--from_yaml_config set)")
            # Overwrite command line arguments with conf file arguments
            script_args = read_yaml_config(entry_args.from_yaml_config)
        else:
            script_args = script_arg_parser.parse_args(non_parsed_script_args)

        if entry_args.aml:
            run_in_aml(pl_system=entry_args.pl_system,
                       conda_dependencies_path=script_args.azure_conda_env_path,
                       script_args=script_args,
                       azure_dataset_id=script_args.azure_dataset_id)
        else:
            logger = prepare_run(script_args=script_args, pl_system=entry_args.pl_system)
            run_locally(modname=modname, script_args=script_args, logger=logger)


if __name__ == "__main__":
    entry_func()
