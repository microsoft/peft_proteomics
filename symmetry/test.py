# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
import torch
import os
import numpy as np
import pickle as pk
from pathlib import Path
from tqdm.auto import tqdm
from torch.utils.data import DataLoader
from torch.nn.utils.rnn import pad_sequence
import torchmetrics as tm
from omegaconf import OmegaConf

from logutil import BaseLogger
from modeling import get_model
from data import init_data_dictionary, get_token_datasets, ESMTokenizerDataset
from plotting import (
    plot_confusion_matrix,
    plot_bar,
    plot_metric_df,
    plot_aupr_curves,
    plot_distributions,
)

import argparse
from sklearn.metrics import average_precision_score, precision_recall_curve, accuracy_score, f1_score, precision_score, recall_score, matthews_corrcoef, confusion_matrix
import matplotlib.pyplot as plt

parser = argparse.ArgumentParser()

parser.add_argument("--run_name", type=str)
parser.add_argument("--config", type=str)
parser.add_argument("--checkpoint", type=str)
parser.add_argument("--device", type=int, default=0)
parser.add_argument("--valid_also", action="store_true", help="Compute validation metrics also")
parser.add_argument("--valid_only", action="store_true", help="Compute validation metrics only")
args = parser.parse_args()

run_name = args.run_name
config = args.config

logg = BaseLogger(run_name)

DEVICE_ID = args.device
os.environ["CUDA_VISIBLE_DEVICES"] = str(DEVICE_ID)
device = torch.device(f"cuda:0" if torch.cuda.is_available() else "cpu")

CONFIG_PATH = args.config
params = OmegaConf.load(CONFIG_PATH)
params.rank = 0

train_df, valid_df, test_df = init_data_dictionary(params.META_DATA_FILE)
valid_dataset = ESMTokenizerDataset(valid_df, params.max_crop, params.esm_pretrained, params.n_classes)
test_dataset = ESMTokenizerDataset(test_df, params.max_crop, params.esm_pretrained, params.n_classes)

def get_metric_collection(params, prefix: str = "valid/"):
    collection = tm.MetricCollection({
        "accuracy": tm.Accuracy(task = "multiclass", num_classes = params.n_classes, average = "macro"),
        "aupr": tm.AveragePrecision(task = "multiclass", num_classes = params.n_classes, average = "macro"),
        "precision": tm.Precision(task = "multiclass", num_classes = params.n_classes, average = "macro"),
        "recall": tm.Recall(task = "multiclass", num_classes = params.n_classes, average = "macro"),
        "f1": tm.F1Score(task = "multiclass", num_classes = params.n_classes, average = "macro"),
        "mcc": tm.MatthewsCorrCoef(task = "multiclass", num_classes = params.n_classes, average = "macro"),
        "specificity": tm.Specificity(task = "multiclass", num_classes = params.n_classes, average = "macro"),
        "perclass_aupr": tm.AveragePrecision(task = "multiclass", num_classes = params.n_classes, average = None),
        "perclass_accuracy": tm.Accuracy(task = "multiclass", num_classes = params.n_classes, average = None),
        "perclass_f1": tm.F1Score(task = "multiclass", num_classes = params.n_classes, average = None),
    }, prefix = prefix)
    return collection

def loader_collate_fn(batch_list):
        feats = []
        labels = []

        for (f, l) in batch_list:
            feats.append(f)
            labels.append(l)

        PAD_VALUE = 0.
        feats = pad_sequence(feats, batch_first = True, padding_value = PAD_VALUE)
        feats = feats[:, :params.max_crop]
        atten_map = (feats != PAD_VALUE).int()
        
        labels = torch.stack(labels)


        return (feats, labels, atten_map) 

valid_loader = DataLoader(
        valid_dataset,
        batch_size=1,
        shuffle=False,
        num_workers=params.num_workers,
        pin_memory=params.pin_memory,
        collate_fn = loader_collate_fn,
    )
test_loader = DataLoader(
        test_dataset,
        batch_size=1,
        shuffle=False,
        num_workers=params.num_workers,
        pin_memory=params.pin_memory,
        collate_fn = loader_collate_fn,
    )

model = get_model(params, logg, inference_mode = True)

MODEL_SAVE_PATH = args.checkpoint
logg.info(f"Loading checkpoint from {MODEL_SAVE_PATH}")
checkpoint = torch.load(MODEL_SAVE_PATH)
model.load_state_dict(checkpoint["model_state_dict"])

if not args.valid_only:
    logg.info(f"Computing test predictions")
    if os.path.exists(f"./symmetry/results/{run_name}.test_preds.pk"):
        logg.info(f"Found ./symmetry/results/{run_name}.test_preds.pk, loading...")
        with open(f"./symmetry/results/{run_name}.test_preds.pk", "rb") as f:
            test_labels, test_yhat = pk.load(f)
    else:
        test_labels = []
        test_yhat = []

        
        with torch.inference_mode():
            for b in tqdm(test_loader, total = len(test_loader)):
                feats, labels, attn = b
                feats = feats[:, :params.max_crop].to(params.rank)
                attn = attn[:, :params.max_crop].to(params.rank)

                logits = model(feats, attention_mask=attn)
                test_labels.append(labels.squeeze().cpu().detach())
                test_yhat.append(logits[0].squeeze().cpu().detach())
                
        test_labels = torch.stack(test_labels)
        test_yhat = torch.stack(test_yhat)

        with open(f"./symmetry/results/{run_name}.test_preds.pk", "wb") as f:
            pk.dump((test_labels, test_yhat), f)

    if isinstance(test_yhat, np.ndarray):
        test_yhat = torch.from_numpy(test_yhat)
    if isinstance(test_labels, np.ndarray):
        test_labels = torch.from_numpy(test_labels)
    test_yhat = torch.sigmoid(test_yhat)
    test_collection = get_metric_collection(params, prefix = "test/")
    test_collection.reset()
    test_collection(test_yhat.float().cpu().detach(), torch.argmax(test_labels.cpu(), dim = -1))
    test_metrics = test_collection.compute()
    for k,v in test_metrics.items():
        logg.info(f"{k}: {v}")

if args.valid_also or args.valid_only:
    logg.info(f"Computing validation predictions")

    if os.path.exists(f"./symmetry/results/{run_name}.valid_preds.pk"):
        logg.info(f"Found ./symmetry/results/{run_name}.valid_preds.pk, loading...")
        with open(f"./symmetry/results/{run_name}.valid_preds.pk", "rb") as f:
            valid_labels, valid_yhat = pk.load(f)
    else:
        
        valid_labels = []
        valid_yhat = []
        
        with torch.inference_mode():
            for b in tqdm(valid_loader, total = len(valid_loader)):
                feats, labels, attn = b
                feats = feats[:, :params.max_crop].to(params.rank)
                attn = attn[:, :params.max_crop].to(params.rank)

                logits = model(feats, attention_mask=attn)
                valid_labels.append(labels.squeeze().cpu().detach())
                valid_yhat.append(logits[0].squeeze().cpu().detach())
                
        valid_labels = np.stack(valid_labels)
        valid_yhat = np.stack(valid_yhat)

        with open(f"./symmetry/results/{run_name}.valid_preds.pk", "wb") as f:
            pk.dump((valid_labels, valid_yhat), f)

    valid_yhat = torch.sigmoid(valid_yhat)
    valid_collection = get_metric_collection(params, prefix = "valid/")
    valid_collection.reset()
    valid_collection(valid_yhat.float().cpu().detach(), torch.argmax(valid_labels.cpu(), dim = -1))
    valid_metrics = valid_collection.compute()
    for k,v in valid_metrics.items():
        logg.info(f"{k}: {v}")