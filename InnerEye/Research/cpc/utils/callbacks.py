import logging
from os import path, listdir
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping
from dataclasses import dataclass
from re import compile


@dataclass(frozen=True)
class CheckpointConfig:
    out_dir = "outputs"
    regex = compile(r"epoch=(\d+).*?=(\d+.\d+).ckpt")
    name_format = "{epoch:02d}-{%s:.4f}"

    @staticmethod
    def get_ckpt_paths_in_dir(dir_path=None):
        """
        Returns a list of tuple (ckpt_name, (epoch, score)) for all valid ckpt files in a dir.
        :param dir_path:
        :return:
        """
        models = listdir(dir_path or CheckpointConfig.out_dir)
        hits = [(m, tuple(map(float, CheckpointConfig.regex.findall(m)[0])))
                if CheckpointConfig.regex.match(m) else None for m in models]
        return list(filter(None, hits))

    @staticmethod
    def get_best_ckpt_path(dir_path=None, mode="min"):
        assert mode in ["min", "max"]
        op = min if mode == "min" else max
        models = CheckpointConfig.get_ckpt_paths_in_dir(dir_path)
        if not models:
            return None
        else:
            return op(models, key=lambda p: p[1][1])

    @staticmethod
    def get_latest_ckpt_path(dir_path=None):
        models = CheckpointConfig.get_ckpt_paths_in_dir(dir_path)
        if not models:
            return None
        else:
            return max(models, key=lambda p: p[1][0])


def get_best_model_from_checkpoint_cb(checkpoint_cb):
    op = max if checkpoint_cb.mode == 'max' else min
    return op(checkpoint_cb.best_k_models, 
              key=checkpoint_cb.best_k_models.get)


def get_model_checkpoint_cb(file_path=None,
                            monitor="avg_val_loss",
                            verbose=True,
                            save_top_k=1,
                            save_weights_only=False,
                            mode="min"):
    if file_path is None:
        file_path = path.join(CheckpointConfig.out_dir, CheckpointConfig.name_format % monitor)
    logging.info("ModelCheckpoint monitoring '{}', mode '{}'".format(
        monitor, mode
    ))
    return ModelCheckpoint(filepath=file_path,
                           monitor=monitor,
                           verbose=verbose,
                           save_top_k=save_top_k,
                           save_weights_only=save_weights_only,
                           mode=mode)


def get_early_stopping_cb(monitor="avg_val_loss", patience=20):
    logging.info("EarlyStopping monitoring '{}', patience {}".format(
        monitor, patience
    ))
    return EarlyStopping(monitor=monitor, patience=patience)
