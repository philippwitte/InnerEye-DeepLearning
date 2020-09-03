import logging
from sys import stdout
from pytorch_lightning.loggers import TensorBoardLogger
from numpy import ndarray
from torch import Tensor, as_tensor


class AMLTensorBoardLogger(TensorBoardLogger):
    def __init__(self, *args, **kwargs):
        from azureml.core import Run
        from azureml.exceptions import RunEnvironmentException
        try:
            self.aml_run = Run.get_context(allow_offline=False)
        except RunEnvironmentException:
            logging.warning("Running offline, disabling AML Run logging.")
            self.aml_run = None
        super(AMLTensorBoardLogger, self).__init__(*args, **kwargs)

    def set_tags(self, tags):
        if self.aml_run:
            self.aml_run.set_tags(tags)
        self.tags.update(tags)

    @staticmethod
    def log_hyperparameters_to_aml(args, aml_run):
        for key, value in args.items():
            if isinstance(value, (list, tuple, ndarray, Tensor)):
                aml_run.log_list(key, value)
            else:
                aml_run.log(key, value)

    @staticmethod
    def log_metrics_to_aml(metrics, aml_run):
        for k, v in metrics.items():
            if isinstance(v, Tensor):
                v = v.item()
            aml_run.log(k, v)

    def log_hyperparams(self, params):
        from argparse import Namespace
        if isinstance(params, Namespace):
            params = vars(params)
        if self.aml_run:
            self.log_hyperparameters_to_aml(params, self.aml_run)
        # Check all TensorBoard safe types
        safe_params = {}
        for key, value in params.items():
            if not isinstance(value, (int, float, str, bool, Tensor)):
                if isinstance(value, (list, tuple)) and isinstance(value[0], (int, float)):
                    safe_params[key] = as_tensor(value)
                else:
                    safe_params[key] = f'"{str(value)}"'
        super(AMLTensorBoardLogger, self).log_hyperparams(safe_params)

    def log_metrics(self, metrics, step):
        if self.aml_run:
            self.log_metrics_to_aml(metrics, self.aml_run)
        super(AMLTensorBoardLogger, self).log_metrics(metrics, step)


def get_innereye_build(dist_name="InnerEye"):
    import pkg_resources
    return pkg_resources.get_distribution(dist_name).version


def get_default_logger(args):
    tags = {
        "run_name": getattr(args, "run_name"),
        "source_name": "my_project",
        "innereye_build": get_innereye_build()
    }
    args.innereye_build = tags["innereye_build"]
    from InnerEye.Research.cpc.utils.version_controller import VersionController
    from git import InvalidGitRepositoryError
    try:
        vc = VersionController()
        tags["user"] = vc.user_name,
        tags["remote_url"] = vc.remote_url
        tags["commit"] = vc.current_commit,
        tags["branch"] = vc.branch,
    except InvalidGitRepositoryError:
        logging.warning("Cannot set Git related tags, specified "
                        "path was not a valid git repo.")
    log_name = getattr(args, "run_name", "run")
    logger = AMLTensorBoardLogger("logs/tensorboard", name=log_name)
    try:
        logger.log_hyperparams(args)
    except Exception as e:
        logging.error("Could not log hyperparameters due to error: {}".format(str(e)))
    try:
        logger.set_tags(tags)
    except Exception as e:
        logging.error("Could not set tags due to error: {}".format(str(e)))
    return logger


def configure_logging(log_path, log_level=logging.INFO):
    if isinstance(log_level, str):
        log_level = getattr(logging, log_level.upper())
    logging.basicConfig(level=log_level,
                        format='%(levelname)s | %(asctime)s | %(message)s',
                        datefmt='%m/%d/%Y %H:%M:%S',
                        handlers=[
                            logging.FileHandler(log_path, mode="w"),
                            logging.StreamHandler(stdout)
                        ])
