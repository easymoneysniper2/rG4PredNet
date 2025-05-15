import os, sys
import numpy as np
from six.moves import cPickle
from sklearn.metrics import roc_curve, auc, precision_recall_curve, accuracy_score, roc_auc_score, confusion_matrix
from scipy import stats

__all__ = [
    "pearsonr",
    "rsquare",
    "accuracy",
    "roc",
    "pr",
    "calculate_metrics"
]

class MLMetrics(object):
    def __init__(self, objective='binary'):
        self.objective = objective
        self.metrics = []

    def update(self, label, pred, other_lst):
        met, _ = calculate_metrics(label, pred, self.objective)
        if len(other_lst) > 0:
            met.extend(other_lst)
        self.metrics.append(met)
        self.compute_avg()

    def compute_avg(self):
        if len(self.metrics) > 1:
            self.avg = np.array(self.metrics).mean(axis=0)
            self.sum = np.array(self.metrics).sum(axis=0)
        else:
            self.avg = self.metrics[0]
            self.sum = self.metrics[0]
        self.acc = self.avg[0]
        self.auc = self.avg[1]
        self.prc = self.avg[2]
        self.tp = int(self.sum[3])
        self.tn = int(self.sum[4])
        self.fp = int(self.sum[5])
        self.fn = int(self.sum[6])
        if len(self.avg) > 7:
            self.other = self.avg[7:]

def pearsonr(label, prediction):
    ndim = np.ndim(label)
    if ndim == 1:
        return [stats.pearsonr(label, prediction)]
    else:
        return [stats.pearsonr(label[:, i], prediction[:, i])[0] for i in range(label.shape[1])]

def rsquare(label, prediction):
    ndim = np.ndim(label)
    if ndim == 1:
        y, X = label, prediction
        m = np.dot(X, y) / np.dot(X, X)
        resid = y - m * X
        ym = y - np.mean(y)
        rsqr2 = 1 - np.dot(resid.T, resid) / np.dot(ym.T, ym)
        return [rsqr2], [m]
    else:
        metrics, slopes = [], []
        for i in range(label.shape[1]):
            y, X = label[:, i], prediction[:, i]
            m = np.dot(X, y) / np.dot(X, X)
            resid = y - m * X
            ym = y - np.mean(y)
            rsqr2 = 1 - np.dot(resid.T, resid) / np.dot(ym.T, ym)
            metrics.append(rsqr2)
            slopes.append(m)
        return metrics, slopes

def accuracy(label, prediction, threshold=0.5):
    def apply_threshold(pred, threshold):
        return (pred >= threshold).astype(int)
    
    ndim = np.ndim(label)
    if ndim == 1:
        return np.array(accuracy_score(label, apply_threshold(prediction, threshold)))
    else:
        return np.array([accuracy_score(label[:, i], apply_threshold(prediction[:, i], threshold)) for i in range(label.shape[1])])

def roc(label, prediction):
    ndim = np.ndim(label)
    if ndim == 1:
        fpr, tpr, _ = roc_curve(label, prediction)
        return np.array(auc(fpr, tpr)), [(fpr, tpr)]
    else:
        metrics, curves = [], []
        for i in range(label.shape[1]):
            fpr, tpr, _ = roc_curve(label[:, i], prediction[:, i])
            metrics.append(auc(fpr, tpr))
            curves.append((fpr, tpr))
        return np.array(metrics), curves

def pr(label, prediction):
    ndim = np.ndim(label)
    if ndim == 1:
        precision, recall, _ = precision_recall_curve(label, prediction)
        return np.array(auc(recall, precision)), [(precision, recall)]
    else:
        metrics, curves = [], []
        for i in range(label.shape[1]):
            precision, recall, _ = precision_recall_curve(label[:, i], prediction[:, i])
            metrics.append(auc(recall, precision))
            curves.append((precision, recall))
        return np.array(metrics), curves

def tfnp(label, prediction, threshold=0.5):
    binary_prediction = (prediction >= threshold).astype(int)
    try:
        tn, fp, fn, tp = confusion_matrix(label, binary_prediction).ravel()
    except ValueError:
        tp, tn, fp, fn = 0, 0, 0, 0
    return int(tp), int(tn), int(fp), int(fn)

def calculate_metrics(label, prediction, objective):
    if objective in ["binary", "hinge"]:
        ndim = np.ndim(label)
        correct = accuracy(label, prediction)
        auc_roc, _ = roc(label, prediction)
        auc_pr, _ = pr(label, prediction)
        if ndim == 2:
            prediction, label = prediction[:, 0], label[:, 0]
        tp, tn, fp, fn = tfnp(label, prediction > 0.5)
        mean = [np.nanmean(correct), np.nanmean(auc_roc), np.nanmean(auc_pr), tp, tn, fp, fn]
        std = [np.nanstd(correct), np.nanstd(auc_roc), np.nanstd(auc_pr)]

    elif objective == "categorical":
        correct = np.mean(np.equal(np.argmax(label, axis=1), np.argmax(prediction, axis=1)))
        auc_roc, _ = roc(label, prediction)
        auc_pr, _ = pr(label, prediction)
        mean = [np.nanmean(correct), np.nanmean(auc_roc), np.nanmean(auc_pr)]
        std = [np.nanstd(correct), np.nanstd(auc_roc), np.nanstd(auc_pr)]
        for i in range(label.shape[1]):
            auc_roc, _ = roc(label[:, i], prediction[:, i])
            mean.append(np.nanmean(auc_roc))
            std.append(np.nanstd(auc_roc))

    elif objective in ['squared_error', 'kl_divergence', 'cdf']:
        ndim = np.ndim(label)
        label = (label >= 0.5).astype(int)
        correct = accuracy(label, prediction)
        auc_roc, _ = roc(label, prediction)
        auc_pr, _ = pr(label, prediction)
        if ndim == 2:
            prediction, label = prediction[:, 0], label[:, 0]
        tp, tn, fp, fn = tfnp(label, prediction > 0.5)
        corr = pearsonr(label, prediction)
        rsqr, slope = rsquare(label, prediction)
        mean = [np.nanmean(correct), np.nanmean(auc_roc), np.nanmean(auc_pr), tp, tn, fp, fn, np.nanmean(corr), np.nanmean(rsqr), np.nanmean(slope)]
        std = [np.nanstd(correct), np.nanstd(auc_roc), np.nanstd(auc_pr), np.nanstd(corr), np.nanstd(rsqr), np.nanstd(slope)]

    else:
        mean, std = 0, 0

    return [mean, std]