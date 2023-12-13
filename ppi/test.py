# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
import torch
import os
import numpy as np
import pickle as pk
from tqdm.auto import tqdm
from torch.utils.data import DataLoader
from torch.nn.utils.rnn import pad_sequence
from omegaconf import OmegaConf

from logutil import BaseLogger
from modeling import get_model
from data import init_data_dictionary, ESMTokenizerDataset


import argparse
from sklearn.metrics import average_precision_score, precision_recall_curve, accuracy_score, f1_score, precision_score, recall_score, matthews_corrcoef, confusion_matrix

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

train_dict, valid_dict, test_dict = init_data_dictionary(params.META_DATA_FILE, params.SPLITS_FILE)
valid_dataset = ESMTokenizerDataset(valid_dict, params.TRAIN_SEQUENCES_FILE, params.max_crop, params.esm_pretrained)
test_dataset = ESMTokenizerDataset(test_dict, params.TRAIN_SEQUENCES_FILE, params.max_crop, params.esm_pretrained)

def loader_collate_fn(batch_list):
    feats0 = []
    feats1 = []
    labels = []

    for (f, l) in batch_list:
        feats0.append(f[0])
        feats1.append(f[1])
        labels.append(l)

    PAD_VALUE = 0.
    feats0 = pad_sequence(feats0, batch_first = True, padding_value = PAD_VALUE)
    feats0 = feats0[:, :params.max_crop]
    attn_map0 = (feats0 != PAD_VALUE).int()
    feats1 = pad_sequence(feats1, batch_first = True, padding_value = PAD_VALUE)
    feats1 = feats1[:, :params.max_crop]
    attn_map1 = (feats1 != PAD_VALUE).int()
    
    labels = torch.stack(labels)


    return ((feats0, feats1), labels, (attn_map0, attn_map1)) 

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
    if os.path.exists(f"./ppi/results/{run_name}.test_preds.pk"):
        logg.info(f"Found ./ppi/results/{run_name}.test_preds.pk, loading...")
        with open(f"./ppi/results/{run_name}.test_preds.pk", "rb") as f:
            test_labels, test_yhat = pk.load(f)
    else:
        test_labels = []
        test_yhat = []

        
        with torch.inference_mode():
            for b in tqdm(test_loader, total = len(test_loader)):
                tokens, labels, mask = b
                tokens = [i.to(device) for i in tokens]
                mask = [i.to(device) for i in mask]
                predictions = model(tokens, attention_mask=mask)
                test_labels.append(labels.squeeze().cpu().detach().numpy())
                test_yhat.append(predictions[0].squeeze().cpu().detach().numpy())
                
        test_labels = np.stack(test_labels)
        test_yhat = np.stack(test_yhat)

        with open(f"./ppi/results/{run_name}.test_preds.pk", "wb") as f:
            pk.dump((test_labels, test_yhat), f)

    test_yhat = torch.sigmoid(torch.from_numpy(test_yhat)).numpy()
    logg.info("Test")
    thresh = 0.5
    pr, rec, thr = precision_recall_curve(test_labels, test_yhat)
    logg.info(f"Accuracy: {accuracy_score(test_labels, test_yhat > thresh)}")
    logg.info(f"F1: {f1_score(test_labels, test_yhat > thresh)}")
    logg.info(f"MCC: {matthews_corrcoef(test_labels, test_yhat > thresh)}")
    logg.info(f"AUPR: {average_precision_score(test_labels, test_yhat)}")
    logg.info(f"Precision: {precision_score(test_labels, test_yhat > thresh, zero_division=0)}")
    logg.info(f"Recall: {recall_score(test_labels, test_yhat > thresh, zero_division=0)}")
    tn, fp, fn, tp = confusion_matrix(test_labels, test_yhat > thresh).ravel()
    logg.info(f"Specificity: {tn / (tn + fp)}")

if args.valid_also or args.valid_only:
    logg.info(f"Computing validation predictions")

    if os.path.exists(f"./ppi/results/{run_name}.valid_preds.pk"):
        logg.info(f"Found ./ppi/results/{run_name}.valid_preds.pk, loading...")
        with open(f"./ppi/results/{run_name}.valid_preds.pk", "rb") as f:
            valid_labels, valid_yhat = pk.load(f)
    else:
        
        valid_labels = []
        valid_yhat = []
        
        with torch.inference_mode():
            for b in tqdm(valid_loader, total = len(valid_loader)):
                tokens, labels, mask = b
                tokens = [i.to(device) for i in tokens]
                mask = [i.to(device) for i in mask]
                predictions = model(tokens, attention_mask=mask)
                valid_labels.append(labels.squeeze().cpu().detach().numpy())
                valid_yhat.append(predictions[0].squeeze().cpu().detach().numpy())
                
        valid_labels = np.stack(valid_labels)
        valid_yhat = np.stack(valid_yhat)

        with open(f"ppi/results/{run_name}.valid_preds.pk", "wb") as f:
            pk.dump((valid_labels, valid_yhat), f)

    valid_yhat = torch.sigmoid(torch.from_numpy(valid_yhat)).numpy()
    logg.info("Validation:")
    thresh = 0.5
    pr, rec, thr = precision_recall_curve(valid_labels, valid_yhat)
    logg.info(f"Accuracy: {accuracy_score(valid_labels, valid_yhat > thresh)}")
    logg.info(f"F1: {f1_score(valid_labels, valid_yhat > thresh)}")
    logg.info(f"MCC: {matthews_corrcoef(valid_labels, valid_yhat > thresh)}")
    logg.info(f"AUPR: {average_precision_score(valid_labels, valid_yhat)}")
    logg.info(f"Precision: {precision_score(valid_labels, valid_yhat > thresh, zero_division=0)}")
    logg.info(f"Recall: {recall_score(valid_labels, valid_yhat > thresh, zero_division=0)}")
    tn, fp, fn, tp = confusion_matrix(valid_labels, valid_yhat > thresh).ravel()
    logg.info(f"Specificity: {tn / (tn + fp)}")