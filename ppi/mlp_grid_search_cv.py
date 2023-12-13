# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
import h5py
import sys
import pickle as pk
import numpy as np
import pandas as pd
from tqdm.auto import tqdm

from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GridSearchCV, PredefinedSplit
from sklearn.neural_network import MLPClassifier

from sklearn.metrics import average_precision_score, precision_recall_curve, accuracy_score, f1_score, precision_score, recall_score, matthews_corrcoef, confusion_matrix
from sklearn.calibration import CalibrationDisplay
import matplotlib.pyplot as plt

SHUFFLE = True
PCT_DATA_TO_USE = 1
PPI_DATA_TABLE = # Path to `gold_standard_data_table.tsv`, see README in `data`
EMBEDDING_FILE = # Path to .h5 file containing language model embeddings for sequences in data table

mode = sys.argv[1]
key = sys.argv[2]
assert mode in ["train", "eval"], mode
do_train = mode == "train"

print("Training..." if mode == "train" else "Evaluating...")

data_frame = pd.read_csv(PPI_DATA_TABLE,sep="\t")

train_data = data_frame[data_frame["split"] == "train"]
valid_data = data_frame[data_frame["split"] == "valid"]
test_data = data_frame[data_frame["split"] == "test"]

X_train = []
X_valid = []
X_test = []

def combine_feats(f1, f2):
    return (f1 + f2) / 2

with h5py.File(EMBEDDING_FILE,"r") as h5fi:
    for _, r in tqdm(train_data.iterrows(), total = len(train_data)):
        X_train.append(combine_feats(h5fi[r.p0][:], h5fi[r.p1][:]))
    for _, r in tqdm(valid_data.iterrows(), total = len(valid_data)):
        X_valid.append(combine_feats(h5fi[r.p0][:], h5fi[r.p1][:]))
    for _, r in tqdm(test_data.iterrows(), total = len(test_data)):
        X_test.append(combine_feats(h5fi[r.p0][:], h5fi[r.p1][:]))

X_train = np.stack(X_train, axis = 0)
y_train = train_data["label"].values

X_valid = np.stack(X_valid, axis = 0)
y_valid = valid_data["label"].values

X_test = np.stack(X_test, axis = 0)
y_test = test_data["label"].values

if SHUFFLE:
    import torch
    indices = np.arange(len(X_train))
    total_size = int(X_train.shape[1] * PCT_DATA_TO_USE)
    g = torch.Generator()
    g.manual_seed(0)
    indices = indices[torch.randperm(len(indices), generator=g)]

    # per each gpu
    indices = indices[:total_size]
    
    X_train = X_train[indices]
    y_train = y_train[indices]

print(X_train.shape, y_train.shape, X_valid.shape, y_valid.shape, X_test.shape, y_test.shape)

def normalize_train_test(X_train, X_valid, X_test):
    scaler = StandardScaler()
    scaler.fit(X_train)
    X_train = scaler.transform(X_train)
    X_valid = scaler.transform(X_valid)
    X_test = scaler.transform(X_test)
    return X_train, X_valid, X_test

X_train_norm, X_valid_norm, X_test_norm = normalize_train_test(X_train, X_valid, X_test)
X_trainval_norm = np.concatenate([X_train_norm, X_valid_norm], axis = 0)
y_trainval = np.concatenate([y_train, y_valid], axis = 0)

if do_train:
    param_grid = {
        "activation": ["logistic", "relu", "identity"],
        "alpha": [0.0001, 0.001, 0.01],
        "learning_rate": ["adaptive"],
        "solver": ["adam"],
        "learning_rate_init": [0.001, 0.01],
        "max_iter": [1000, 2000],
        "hidden_layer_sizes": [
            (64,), (128,), (512,),
            (64, 64), (128, 128),
            (64, 64, 64),
            ],
        "early_stopping": [True],
        "validation_fraction": [0.1],
        "tol": [1e-4, 1e-5],
    }


    valid_index = np.concatenate([-1 * np.ones(X_train_norm.shape[0]), np.zeros(X_valid_norm.shape[0])], axis = 0)

    scoring = ["accuracy", "average_precision", "f1", "recall", "precision", "roc_auc"]
    refit = "average_precision"
    verbose = 10
    n_jobs = -1
    cv = PredefinedSplit(valid_index)

    mlp = MLPClassifier()
    clsf = GridSearchCV(mlp, param_grid, cv=cv, scoring=scoring, verbose=verbose, n_jobs=n_jobs, refit=refit)

    clsf.fit(X_trainval_norm, y_trainval)

    with open(f"./results/{key}_grid_search_cv.pk", "wb+") as fi:
        pk.dump(clsf, fi)
        
    df = pd.DataFrame(clsf.cv_results_)
    df.to_csv(f"./results/{key}_grid_search_cv_results.csv")
    
    best_estimator = clsf.best_estimator_
    with open(f"./results/{key}_best_estimator.pk", "wb+") as fi:
        pk.dump(best_estimator, fi)
        
    with open(f"./results/{key}_best_estimator_validation_metrics.txt", "w+") as fi:
        for score in scoring:
            print(f'{score}: {clsf.cv_results_[f"mean_test_{score}"][clsf.best_index_]}', file = fi, end = "\n")

with open(f"./results/{key}_grid_search_cv.pk", "rb") as fi:
    clsf = pk.load(fi)

with open(f"./results/{key}_best_estimator.pk", "rb") as fi:
    best_estimator = pk.load(fi)


y_test_pred = best_estimator.predict_proba(X_test_norm)[:, 1]

def precision_at_k(pr: np.ndarray, rec: np.ndarray, thr: np.ndarray, k: float, return_thresh = False):
    prc_df = pd.DataFrame([pr, rec, thr]).T
    prc_df.columns = ["precision", "recall", "threshold"]
    pr_at_k = prc_df[prc_df["recall"] >= k]["precision"].iloc[-1]
    if return_thresh:
        thresh = prc_df[prc_df["recall"] >= k]["threshold"].iloc[-1]
        return pr_at_k, thresh
    else:
        return pr_at_k

thresh = 0.5
with open(f"./results/{key}_best_estimator_metrics.txt", "w+") as fi:
    print("Logistic Regression Test", file = fi, end = "\n")
    pr, rec, thr = precision_recall_curve(y_test, y_test_pred)
    print(f"Accuracy: {accuracy_score(y_test, y_test_pred > thresh)}", file = fi, end = "\n")
    print(f"F1: {f1_score(y_test, y_test_pred > thresh)}", file = fi, end = "\n")
    print(f"MCC: {matthews_corrcoef(y_test, y_test_pred > thresh)}", file = fi, end = "\n")
    print(f"AUPR: {average_precision_score(y_test, y_test_pred)}", file = fi, end = "\n")
    print(f"Precision: {precision_score(y_test, y_test_pred > thresh, zero_division=0)}", file = fi, end = "\n")
    print(f"Recall: {recall_score(y_test, y_test_pred > thresh, zero_division=0)}", file = fi, end = "\n")
    tn, fp, fn, tp = confusion_matrix(y_test, y_test_pred > thresh).ravel()
    print(f"Specificity: {tn / (tn + fp)}", file = fi, end = "\n")

    prc_df = pd.DataFrame([pr, rec, thr]).T
    prc_df.columns = ["precision", "recall", "threshold"]
    
    print(f"Precision at 0.01: {precision_at_k(pr, rec, thr, 0.01)}", file = fi, end = "\n")
    print(f"Precision at 0.05: {precision_at_k(pr, rec, thr, 0.05)}", file = fi, end = "\n")
    print(f"Precision at 0.1: {precision_at_k(pr, rec, thr, 0.1)}", file = fi, end = "\n")
    print(f"Precision at 0.5: {precision_at_k(pr, rec, thr, 0.5)}", file = fi, end = "\n")
    print(f"Precision at 0.77: {precision_at_k(pr, rec, thr, 0.77)}", file = fi, end = "\n")

    plt.plot(rec, pr)
    plt.axhline(y=0.5, color='grey', linestyle='--')
    plt.ylim([0.0, 1.05])
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.savefig(f"./results/{key}_best_estimator_pr_curve.png")
    
    plt.clf()
    CalibrationDisplay.from_estimator(best_estimator, X_test_norm, y_test, n_bins = 20).plot()
    plt.savefig(f"./results/{key}_best_estimator_calibration_curve.png")
    
    