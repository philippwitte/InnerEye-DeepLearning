import torch
import pytorch_lightning as pl

from abc import abstractmethod
from InnerEye.Research.cpc.models import LinearClassifier
from sklearn.metrics import roc_auc_score, f1_score
from InnerEye.Research.cpc.utils.training import (compute_average_metrics,
                                                  concatenate_outputs,
                                                  compute_metrics,
                                                  gather_metrics,
                                                  compute_sklearn_metrics)


class DownstreamLinearClassifier(pl.LightningModule):
    """
    This class defines an interface for downstream classifier PL-systems.

    Must be sub-classed implementing at least the prepare_data method which should set the train_loader, val_loader and
    optionally test_loader attributes.
    """
    def __init__(self, encoder, in_channels, n_classes, hparams, metrics=None):
        """
        :param encoder: torch.nn.Module, an encoder model that maps batch as output by e.g. self.train_loader to vectors
        :param in_channels: int, number of channels in the input (output from encoder module)
        :param n_classes: int, number of channels in the output (number of classes to predict among)
        :param hparams: Namespace, hyperparameters as determined by the concrete class.
        """
        super(DownstreamLinearClassifier, self).__init__()

        # Set encoder, init the linear classifier
        self.encoder = encoder
        self.linear_clf = LinearClassifier(in_channels=in_channels, num_classes=n_classes)

        # Train loaders for the STL10 dataset will be set in
        # self.prepare_data()
        self.train_loader, self.val_loader, self.test_loader = None, None, None

        # Init loss and metrics
        self.loss = torch.nn.CrossEntropyLoss()
        metrics = metrics or {
            "accuracy": (("train", "val", "test"), "torch", lambda y_hat, y: (y_hat.argmax(1) == y).sum().float()/len(y)),
            "roc_auc": (("test",), "sklearn", lambda y_hat, y: roc_auc_score(y, y_hat, average="macro")),   # binary only
            "f1_score": (("test",), "sklearn", lambda y_hat, y: f1_score(y, y_hat >= 0.5, average="macro")) # binary only
        }
        self.metrics_by_split = {
            s: gather_metrics(metrics, s) for s in ("train", "val", "test")
        }

        # Set hyperparameters
        self.hparams = hparams

    def forward(self, batch, transform_key=None):
        """ Encode a batch of images, linearly classify each encoding """
        # First returned value is the raw encoding, second is None
        x, y = batch
        if getattr(self.hparams, "finetune_encoder", False):
            x_enc = self.encoder(x)
        else:
            with torch.no_grad():
                x_enc = self.encoder(x)
        pred = self.linear_clf(x_enc)
        return pred

    def training_step(self, batch, batch_idx):
        y_hat = self.forward(batch, transform_key="train")
        loss = self.loss(y_hat, batch["labels"].squeeze())
        torch_metrics, sklearn_metrics = self.metrics_by_split["train"]

        # Add loss and metrics to logs
        log = {"train_loss": loss.detach()}
        log.update(compute_metrics(torch_metrics, y_hat, batch["labels"]))
        log.update(compute_sklearn_metrics(sklearn_metrics, y_hat, batch["labels"]))
        return {"loss": loss, "log": log}

    def validation_step(self, batch, batch_idx):
        """ Computes validation loss and metrics for a single batch """
        y_hat = self.forward(batch, transform_key="val")
        val_loss = self.loss(y_hat, batch["labels"])
        torch_metrics, sklearn_metrics = self.metrics_by_split["val"]

        # Add loss and metrics to output dict
        # Note: we add nothing to logs here, only interested in mean val scores
        output = {"val_loss": val_loss.detach()}
        output.update(compute_metrics(torch_metrics, y_hat, batch["labels"], log_prefix="val"))
        output.update(compute_sklearn_metrics(sklearn_metrics, y_hat, batch["labels"], log_prefix="val"))
        return output

    def validation_epoch_end(self, outputs):
        """ Returns the average loss and metrics over all validation steps """
        if not outputs:
            return {}
        avg_metrics = compute_average_metrics(outputs)
        return {"val_loss": avg_metrics["avg_val_loss"], "log": avg_metrics}

    def test_step(self, batch, batch_idx):
        y_hat = self.forward(batch)
        return {"predictions": y_hat, "labels": batch["labels"].squeeze()}

    def test_epoch_end(self, outputs):
        outputs = concatenate_outputs(outputs)
        y_hat, y = outputs["predictions"], outputs["labels"]
        torch_metrics, sklearn_metrics = self.metrics_by_split["test"]

        log = {"test_loss": self.loss(y_hat, y).detach(),
               "test_support": len(y_hat)}
        log.update(compute_metrics(torch_metrics, y_hat, y, log_prefix="test"))
        log.update(compute_sklearn_metrics(sklearn_metrics, y_hat, y, log_prefix="test"))
        return {"log": log}

    def configure_optimizers(self):
        params = list(self.linear_clf.parameters())
        if getattr(self.hparams, "finetune_encoder", False):
            params += list(self.encoder.parameters())
        return torch.optim.Adam(
            params=params,
            lr=self.hparams.learning_rate
        )

    @abstractmethod
    def prepare_data(self):
        raise NotImplemented

    def train_dataloader(self):
        return self.train_loader

    def val_dataloader(self):
        return self.val_loader

    def test_dataloader(self):
        return self.test_loader
