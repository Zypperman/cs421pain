#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: Guansong Pang
The algorithm was implemented using Python 3.6.6, Keras 2.2.2 and TensorFlow 1.10.1.
More details can be found in our KDD19 paper.
Guansong Pang, Chunhua Shen, and Anton van den Hengel. 2019.
Deep Anomaly Detection with Deviation Networks.
In The 25th ACM SIGKDD Conference on Knowledge Discovery and Data Mining (KDD '19),
August 4-8, 2019, Anchorage, AK, USA. ACM, New York, NY, USA, 10 pages.
https://doi.org/10.1145/3292500.3330871
"""

import pandas as pd
import numpy as np
from sklearn.metrics import (
    roc_auc_score, average_precision_score,
    precision_recall_curve,
)
from sklearn import preprocessing
import matplotlib.pyplot as plt

# ---------------------------------------------------------------------------
# Optional: svm data loading (kept for backwards compatibility)
# sklearn >= 0.23 removed externals.joblib; provide a no-cache fallback
# ---------------------------------------------------------------------------
try:
    from sklearn.externals.joblib import Memory
    from sklearn.datasets import load_svmlight_file
    mem = Memory("./dataset/svm_data")

    @mem.cache
    def get_data_from_svmlight_file(path):
        data = load_svmlight_file(path)
        return data[0], data[1]

except ImportError:
    from joblib import Memory
    from sklearn.datasets import load_svmlight_file
    mem = Memory("./dataset/svm_data")

    @mem.cache
    def get_data_from_svmlight_file(path):
        data = load_svmlight_file(path)
        return data[0], data[1]


def dataLoading(path):
    """Load a CSV with a 'class' column (0 = normal, 1 = anomaly)."""
    df = pd.read_csv(path)
    labels = df['class']
    x_df = df.drop(['class'], axis=1)
    x = x_df.values
    print("Data shape: (%d, %d)" % x.shape)
    return x, labels


def aucPerformance(mse, labels):
    """Compute and print AUC-ROC and AUC-PR."""
    roc_auc = roc_auc_score(labels, mse)
    ap = average_precision_score(labels, mse)
    print("AUC-ROC: %.4f, AUC-PR: %.4f" % (roc_auc, ap))
    return roc_auc, ap


def f1Performance(scores, labels):
    """
    Find the threshold that maximises F1 by sweeping the precision-recall
    curve, then report precision, recall and F1 at that threshold.

    This mirrors exactly how the logistic regression notebooks derive the
    best operating threshold from cross-validated scores.

    Parameters
    ----------
    scores : array-like of float   -- raw anomaly scores (higher = more anomalous)
    labels : array-like of int     -- ground-truth labels (0 = normal, 1 = anomaly)

    Returns
    -------
    best_f1   : float
    precision : float  at best-F1 threshold
    recall    : float  at best-F1 threshold
    threshold : float  decision boundary that maximises F1
    """
    precision_vals, recall_vals, thresholds = precision_recall_curve(labels, scores)

    # precision_recall_curve appends a sentinel final point that has no
    # corresponding threshold entry, so we work on [:-1] slices.
    denom = precision_vals[:-1] + recall_vals[:-1]
    f1_vals = np.where(
        denom > 0,
        2 * precision_vals[:-1] * recall_vals[:-1] / denom,
        0.0,
    )

    best_idx  = int(np.argmax(f1_vals))
    best_f1   = float(f1_vals[best_idx])
    threshold = float(thresholds[best_idx])
    prec      = float(precision_vals[best_idx])
    rec       = float(recall_vals[best_idx])

    print(
        "Best F1: %.4f  (Precision: %.4f, Recall: %.4f, Threshold: %.4f)"
        % (best_f1, prec, rec, threshold)
    )
    return best_f1, prec, rec, threshold


def writeResults(
    name, n_samples, dim, n_samples_trn, n_outliers_trn, n_outliers,
    depth, rauc, ap, std_auc, std_ap,
    mean_f1, std_f1, mean_prec, mean_rec,
    train_time, test_time,
    path="./results/auc_performance_cl0.5.csv",
):
    """
    Append one result row to the CSV results file.

    Columns added vs the original signature:
        mean_f1, std_f1, mean_prec, mean_rec
    """
    csv_file = open(path, 'a')
    row = (
        name + ","
        + str(n_samples) + ","
        + str(dim) + ","
        + str(n_samples_trn) + ","
        + str(n_outliers_trn) + ","
        + str(n_outliers) + ","
        + str(depth) + ","
        + str(rauc) + ","
        + str(std_auc) + ","
        + str(ap) + ","
        + str(std_ap) + ","
        + str(mean_f1) + ","
        + str(std_f1) + ","
        + str(mean_prec) + ","
        + str(mean_rec) + ","
        + str(train_time) + ","
        + str(test_time) + "\n"
    )
    csv_file.write(row)
    csv_file.close()