# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
import os
import numpy as np
import pandas as pd
import typing as T
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, average_precision_score, f1_score, precision_score, recall_score, precision_recall_curve

def create_metric_df(
    labels: np.array,
    predictions: np.array,
    idx2name: T.Dict[int, str],
    thresh: float = 0.5,
    savecsv: T.Optional[str] = None,
) -> pd.DataFrame:
    
    class_metrics = {}
    
    for i, c in idx2name.items():
        class_aupr = average_precision_score(labels[:,i], predictions[:,i])
        class_f1 = f1_score(labels[:,i], predictions[:,i] > thresh, zero_division = 0.)
        class_precision = precision_score(labels[:,i], predictions[:,i] > thresh, zero_division = 0.)
        class_recall = recall_score(labels[:,i], predictions[:,i] > thresh, zero_division = 0.)
        class_metrics[c] = [class_aupr, class_f1, class_precision, class_recall]
        
    class_metrics = pd.DataFrame(class_metrics).T
    class_metrics.columns = ["AUPR", "F1", "Precision", "Recall"]
    
    if savecsv is not None:
        class_metrics.to_csv(savecsv, sep = "\t", float_format = "%.3f")
    return class_metrics

def plot_metric_df(
    class_metrics: pd.DataFrame,
    figsize: T.Tuple[int, int] = (30, 15),
    savefig: T.Optional[str] = None,
    hue_col: T.Optional[str] = None,
):
    sns.set(style="whitegrid", font_scale = 1)
    _, ax = plt.subplots(2, 2, figsize = figsize)
    
    
    metric_columns = [x for x in class_metrics.columns if x != hue_col]
    
    for i, c in enumerate(metric_columns):
        
        if hue_col is not None:
            df = class_metrics[[c, hue_col]].reset_index()
            sns.barplot(data = df, x = "index", y = c, hue = hue_col, ax = ax[i // 2][i % 2])
        else:
            df = class_metrics[c].reset_index()
            sns.barplot(data = df, x = "index", y = c, color = "grey", ax = ax[i // 2][i % 2])
        ax[i // 2][i % 2].set_title(c)
        ax[i // 2][i % 2].set_ylim(0, 1)
        ax[i // 2][i % 2].set_yticks(np.arange(0, 1.1, 0.1))
        ax[i // 2][i % 2].set_xlabel("Class")
        ax[i // 2][i % 2].set_xticklabels(ax[i // 2][i % 2].get_xticklabels(), rotation = 45)
        ax[i // 2][i % 2].set_ylabel("Value")
    sns.despine()
    plt.legend()
    
    if savefig is not None:
        plt.savefig(savefig, dpi = 300, bbox_inches = "tight")
    plt.show()

def plot_bar(
    labels: np.array,
    predictions: np.array,
    idx2name: T.Dict[int, str],
    figsize: T.Tuple[int, int] = (30, 15),
    savefig: T.Optional[str] = None,
):
    class_metrics = create_metric_df(labels, predictions, idx2name, savecsv = None)
    plot_metric_df(class_metrics, figsize = figsize, savefig = savefig)
    

def plot_confusion_matrix(
    labels: np.array,
    predictions: np.array,
    idx2name: T.Dict[int, str],
    normalize: T.Optional[bool] = True,
    savefig: T.Optional[str] = None,
):
    
    classes = idx2name.values()
    labels = np.argmax(labels, axis = 1)
    
    normalize = "true" if normalize else "none"
    cm_norm = pd.DataFrame(confusion_matrix(labels, predictions.argmax(axis=1), normalize = normalize))
    cm_norm.index = classes
    cm_norm.columns = classes
    sns.heatmap(cm_norm, cmap = "viridis", vmin = 0, vmax = 1)
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.title("Confusion Matrix")
    
    if savefig is not None:
        plt.savefig(savefig, dpi = 300, bbox_inches = "tight")
    plt.show()
    
def plot_aupr_curves(
    labels: np.array,
    predictions: np.array,
    idx2name: T.Dict[int, str],
    figsize: T.Tuple[int, int] = (20, 20),
    savefig: T.Optional[str] = None,
):
    sns.set(style="whitegrid")
    ncol = 4
    _, ax = plt.subplots((len(idx2name.values()) // ncol)+1, ncol, figsize = figsize)

    for i, c in idx2name.items():
        pr, rc, _ = precision_recall_curve(labels[:,i], predictions[:,i])
        ax[i // ncol][i % ncol].plot(rc, pr)
        ax[i // ncol][i % ncol].set_title(f"Class: {c}")
        ax[i // ncol][i % ncol].set_xlim(0, 1)
    
    if savefig is not None:
        plt.savefig(savefig, dpi = 300, bbox_inches = "tight")
    plt.show()

def plot_distributions(
    labels: np.array,
    predictions: np.array,
    idx2name: T.Dict[int, str],
    figsize: T.Tuple[int, int] = (20, 20),
    savefig: T.Optional[str] = None,
):
    sns.set(style="whitegrid")
    ncol = 4
    _, ax = plt.subplots((len(idx2name.values()) // ncol)+1, ncol, figsize = figsize)

    for i, c in idx2name.items():
        df = pd.DataFrame({"Label": labels[:,i], "Score": predictions[:,i]})
        sns.histplot(data = df, x = "Score", hue = "Label", ax = ax[i // ncol][i % ncol], bins = 20, stat = "probability", common_norm = False, element = "step")
        ax[i // ncol][i % ncol].set_title(f"Class: {c}")
        ax[i // ncol][i % ncol].set_xlim(0, 1)
        ax[i // ncol][i % ncol].set_xlabel("")
        ax[i // ncol][i % ncol].set_ylabel("")
    
    if savefig is not None:
        plt.savefig(savefig, dpi = 300, bbox_inches = "tight")
    plt.show()
    
def evaluation_package(
    labels: np.array,
    predictions: np.array,
    idx2name: T.Dict[int, str],
    basename: str = "evaluation",
) -> pd.DataFrame:
    os.makedirs(basename, exist_ok = True)
    
    class_metrics = create_metric_df(labels, predictions, idx2name, savecsv = f"{basename}/metrics.tsv")
    
    plot_bar(labels, predictions, idx2name, savefig = f"{basename}/{basename}_metrics_bar.png")
    plot_confusion_matrix(labels, predictions, idx2name, savefig = f"{basename}/{basename}_confusion_matrix.png")
    plot_aupr_curves(labels, predictions, idx2name, savefig = f"{basename}/{basename}_aupr_curves.png")
    plot_distributions(labels, predictions, idx2name, savefig = f"{basename}/{basename}_distributions.png")
    
    return class_metrics