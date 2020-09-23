#  ------------------------------------------------------------------------------------------
#  Copyright (c) Microsoft Corporation. All rights reserved.
#  Licensed under the MIT License (MIT). See LICENSE in the repo root for license information.
#  ------------------------------------------------------------------------------------------
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from pandas import DataFrame
from sklearn import metrics
from sklearn.metrics import roc_auc_score

from InnerEye.ML.utils.metrics_constants import LoggingColumns


def plot_roc(labels: np.ndarray, posteriors: np.ndarray) -> None:
    num_elements = labels.size
    fpr, tpr, threshold = metrics.roc_curve(labels, posteriors, pos_label=1)
    roc_auc = roc_auc_score(labels, posteriors)
    plt.figure()
    plt.plot(fpr, tpr, 'b', label='AUC = %0.3f' % roc_auc)
    plt.ylabel('True positive rate')
    plt.xlabel('False positive rate')
    plt.title(f'ROC - N={num_elements}')
    plt.plot([0, 1], [0, 1], 'r--')
    plt.legend(loc='lower right')
    plt.grid()
    plt.show()


def plot_roc_on_axis(labels: np.ndarray, posteriors: np.ndarray, ax: Any, title_prefix: str = '') -> None:
    num_elements = labels.size
    fpr, tpr, threshold = metrics.roc_curve(labels, posteriors, pos_label=1)
    roc_auc = roc_auc_score(labels, posteriors)
    ax.plot(fpr, tpr, 'b', label='AUC = %0.3f' % roc_auc)
    ax.set_ylabel('True positive rate')
    ax.set_xlabel('False positive rate')
    ax.set_title(f'{title_prefix} ROC - N={num_elements}')
    ax.plot([0, 1], [0, 1], 'r--')
    ax.legend(loc='lower right')
    ax.grid()


def plot_aggregates_per_epoch(df: DataFrame) -> None:
    """
    Consumes a dataframe that holds a cross validation metrics_aggregates file, with metrics per epoch,
    and plots a per-epoch breakdown of the important metrics.
    """
    fig, axs = plt.subplots(nrows=2, ncols=3, figsize=(15, 10))
    axs = axs.transpose().flatten()
    metrics = [LoggingColumns.AreaUnderRocCurve,
               LoggingColumns.AreaUnderPRCurve,
               LoggingColumns.FalsePositiveRateAtOptimalThreshold,
               LoggingColumns.FalseNegativeRateAtOptimalThreshold,
               LoggingColumns.CrossEntropy]
    for i, metric in enumerate(metrics):
        ax = axs[i]
        if metric.value in df:
            plt.sca(ax)
            sns.lineplot(data=df, x="epoch", y=metric.value, hue="prediction_target")
            plt.title(metric.value)
            ax.get_xaxis().get_label().set_visible(False)
            ax.get_yaxis().get_label().set_visible(False)
            ax.grid()
            if i < 4:
                ax.set_ybound(0, 1)
    axs[-1].set_visible(False)
