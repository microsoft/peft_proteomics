# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
import os
import logging
import typing as T
import torch
import torch.nn as nn
import torchmetrics as tm
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.nn.parallel import DistributedDataParallel as DDP
from transformers import EsmModel
from peft import LoraConfig, TaskType, get_peft_model

torch.backends.cudnn.benchmark = False
torch.backends.cudnn.deterministic = True

from omegaconf import OmegaConf
from logutil import timer, BaseLogger
from multi import setup, cleanup
from data import get_datasets, get_dataloaders
from scheduler import get_cosine_decay_schedule

class PPIModel(nn.Module):
    def __init__(self, language_model, classifier):
        super().__init__()
        self.language_model = language_model
        self.classifier = classifier
        
    def forward(self, input_ids, attention_mask):
        p1_tokens = input_ids[0]
        p2_tokens = input_ids[1]
        
        p1_attention = attention_mask[0]
        p2_attention = attention_mask[1]
        
        p1_embedding = self.language_model(p1_tokens, p1_attention)[1]
        p2_embedding = self.language_model(p2_tokens, p2_attention)[1]
        mean_embedding = torch.div(torch.add(p1_embedding, p2_embedding), 2)
        
        clsf = self.classifier(mean_embedding)
        
        return clsf
        

class PPIClassificationHead(nn.Module):
    """Head for sentence-level classification tasks."""

    def __init__(self, input_size, hidden_size, hidden_dropout_prob):
        super().__init__()
        self.dense = nn.Linear(input_size, hidden_size)
        self.dropout = nn.Dropout(hidden_dropout_prob)
        self.out_proj = nn.Linear(hidden_size, 1)

    def forward(self, features, **kwargs):
        x = self.dropout(features)
        x = self.dense(x)
        x = torch.tanh(x)
        x = self.dropout(x)
        x = self.out_proj(x)
        return x

def count_grad_parameters(model):
        return sum(p.numel() for p in model.parameters() if p.requires_grad)

def count_all_parameters(model):
    return sum(p.numel() for p in model.parameters())

def set_param_grad(
    model: nn.Module,
    key_requires_grad: T.List[str] = ["lora"]
) -> nn.Module:

    for (name, param) in model.named_parameters():
        param.requires_grad = False
        for k in key_requires_grad:
            if k in name:
                param.requires_grad = True
                break

    return model

def get_model(
    params: OmegaConf, 
    logg: BaseLogger,
    inference_mode: bool = False,
    ) -> nn.Module:
    
    rank = params.rank
    
    # Main ESM Pre-trained model
    language_model = EsmModel.from_pretrained(
        f"facebook/{params.esm_pretrained}", 
        return_dict = False,
    )
    
    # Add LoRA parameters
    if params.do_peft:
        esm_target_modules = []
        if params.do_query: esm_target_modules.append("query")
        if params.do_key: esm_target_modules.append("key")
        if params.do_value: esm_target_modules.append("value")
        
        modules_to_save = []
        layers_to_transform = getattr(params, "layers_to_transform", None)
        
        if inference_mode: params.inference_mode = True

        peft_config = LoraConfig(
            task_type=TaskType.FEATURE_EXTRACTION,
            inference_mode=params.inference_mode,
            r=params.lora_r,
            lora_alpha=params.lora_alpha,
            lora_dropout=params.lora_dropout,
            bias=params.lora_bias,
            target_modules=esm_target_modules,
            layers_to_transform=layers_to_transform,
            modules_to_save=modules_to_save,
        )
        language_model = get_peft_model(language_model, peft_config)

    input_size = language_model.pooler.dense.out_features
    cls_head = PPIClassificationHead(
        input_size = input_size,
        hidden_size = params.esm_clsf_hidden_size,
        hidden_dropout_prob = params.esm_clsf_dropout_prob,
        )
    
    logg.info("Using protein averaging model")
    model = PPIModel(
        language_model = language_model,
        classifier = cls_head,
    )
        
    # Move to GPU
    model = model.to(rank, non_blocking = True)
    
    
    if not inference_mode:
        # Add DDP wrapper
        model = DDP(
            model,
            device_ids=[rank],
            static_graph=True,
            find_unused_parameters=True
        )
    
        # Set requires_grad
        key_requires_grad = ["classifier"]
        key_requires_grad.extend(params.key_requires_grad)
        if params.do_peft: key_requires_grad.extend(["lora"])
    else:
        key_requires_grad = []
    
    model = set_param_grad(model, key_requires_grad = key_requires_grad)
    
    logg.info(model)
    if (rank == 0):
        for (n,p) in model.named_parameters():
            if p.requires_grad: logg.debug(f"Param requires grad: {n}")
    n_trainable_params = count_grad_parameters(model)
    n_total_params = count_all_parameters(model)
    logg.info(f"Total number of parameters: {n_total_params}")
    logg.info(f"Number of trainable parameters: {n_trainable_params} ({100 * n_trainable_params / n_total_params:.2f}%)")
    
    if inference_mode:
        model = model.eval()
    
    return model

def add_weight_decay(
    model: nn.Module,
    l2_coeff: float,
    lora_l2_coeff: T.Optional[float] = None
    ) -> T.List[T.Dict[str, T.Union[nn.Parameter, float]]]:

    lora_l2_coeff = lora_l2_coeff or l2_coeff

    lora_decay, decay, no_decay = [], [], []
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        elif "lora" in name:
            lora_decay.append(param)
        elif "norm" in name or name.endswith(".bias"):
            no_decay.append(param)
        else:
            decay.append(param)
    return [{'params': lora_decay, 'weight_decay': lora_l2_coeff}, {'params': no_decay, 'weight_decay': 0.0}, {'params': decay, 'weight_decay': l2_coeff}]

def train_epoch(
    epoch: int,
    model: nn.Module,
    optimizer: optim.Optimizer,
    scheduler: optim.lr_scheduler._LRScheduler,
    train_loader: DataLoader,
    params: OmegaConf, 
    logg: BaseLogger
    ):
    
    rank = params.rank
    loss_fn = nn.BCEWithLogitsLoss()
    active_fn = nn.Sigmoid()
    model.train()
    model.to(rank)
    
    running_loss = 0
    tm_metrics = get_metric_collection(params, prefix = "train/").to(rank)
    tm_metrics.reset()
    
    for i, batch in enumerate(train_loader):
        with torch.cuda.amp.autocast(enabled = params.USE_AMP), model.no_sync():
            feats, labels, attn = batch
            feats0 = feats[0][:, :params.max_crop].to(rank)
            feats1 = feats[1][:, :params.max_crop].to(rank)
            feats = (feats0, feats1)
            attn0 = attn[0][:, :params.max_crop].to(rank)
            attn1 = attn[1][:, :params.max_crop].to(rank)
            attn = (attn0, attn1)
            labels = labels.to(rank)

            logits = model(feats, attention_mask=attn)
            loss = loss_fn(logits, labels.float()) / params.accum_step
            running_loss += loss.detach()
            loss.backward()
            
            tm_metrics(active_fn(logits).float().detach(), labels.int())
            
        if ((i + 1) % params.accum_step == 0) or (i + 1 == len(train_loader)):
            torch.nn.utils.clip_grad_norm_(model.parameters(), 0.2)
            optimizer.step()
            optimizer.zero_grad()
            scheduler.step()
            
        if ((i + 1) % params.report_step == 0):
            report_loss = running_loss.item() * params.accum_step / (i+1)
            logg.metric("train/loss", report_loss, step = (epoch * len(train_loader)) + i, log_also = False)
            logg.info(f"[Train] [Epoch {epoch}] Step {i+1}/{len(train_loader)} | Loss: {report_loss:.6f}")
        
    
    logg.info("[Train] [Epoch {epoch}] Computing metrics...")
    metrics = compute_metrics(tm_metrics)
    metrics["train/loss"] = running_loss * params.accum_step / len(train_loader)
    
    for k, v in metrics.items():
        logg.metric(k, v.item(), step = epoch)
  
    logg.info(f"[Train] [Epoch {epoch}] Max memory allocated: {torch.cuda.max_memory_allocated(rank) / 1024**3:.2f} GB")
  
    return model, optimizer, scheduler, metrics

def valid_epoch(
    epoch: int,
    model: nn.Module,
    valid_loader: DataLoader,
    params: OmegaConf, 
    logg: BaseLogger
    ):
    
    rank = params.rank
    loss_fn = nn.BCEWithLogitsLoss()
    active_fn = nn.Sigmoid()
    model.eval()
    model.to(rank)
    
    running_loss = 0
    tm_metrics = get_metric_collection(params, prefix = "valid/").to(rank)
    tm_metrics.reset()
    
    for i, batch in enumerate(valid_loader):
        with torch.cuda.amp.autocast(enabled = params.USE_AMP), model.no_sync():
            feats, labels, attn = batch
            feats0 = feats[0][:, :params.max_crop].to(rank)
            feats1 = feats[1][:, :params.max_crop].to(rank)
            feats = (feats0, feats1)
            attn0 = attn[0][:, :params.max_crop].to(rank)
            attn1 = attn[1][:, :params.max_crop].to(rank)
            attn = (attn0, attn1)
            labels = labels.to(rank)

            logits = model(feats, attention_mask=attn)
            loss = loss_fn(logits, labels.float())
            running_loss += loss.detach()
            
            tm_metrics(active_fn(logits).float().detach(), labels.int())
            
        if ((i + 1) % params.report_step == 0):
            logg.info(f"[Validation] [Epoch {epoch}] Step {i+1}/{len(valid_loader)}")
    
    
    metrics = compute_metrics(tm_metrics)
    metrics["valid/loss"] = running_loss / len(valid_loader)
    
    for k, v in metrics.items():
        logg.metric(k, v.item(), step = epoch)
    return model, metrics

def get_metric_collection(_, prefix: str = "valid/"):
    collection = tm.MetricCollection({
        "accuracy": tm.Accuracy(task = "binary"),
        "aupr": tm.AveragePrecision(task = "binary"),
        "precision": tm.Precision(task = "binary"),
        "recall": tm.Recall(task = "binary"),
        "f1": tm.F1Score(task = "binary"),
        "mcc": tm.MatthewsCorrCoef(task = "binary"),
        "specificity": tm.Specificity(task = "binary"),
    }, prefix = prefix)
    return collection

def compute_metrics(metric_collection: tm.MetricCollection):
    
    metrics = metric_collection.compute()
                
    return metrics

def save_model(checkpoint_key, model, optimizer, scheduler, epoch, best_valid_loss, best_valid_aupr, params, logg):
    
    rank = params.rank
    
    state_dict = model.module.state_dict()
    
    for k,v in state_dict.items():
        if isinstance(v, torch.Tensor):
            state_dict[k] = v.cpu()

    save_dict = {
        "epoch": epoch,
        "model_state_dict": state_dict,
        "optimizer_state_dict:": optimizer.state_dict(),
        "scheduler_state_dict": scheduler.state_dict(),
        "best_valid_loss": best_valid_loss, 
        "best_valid_aupr": best_valid_aupr, 
    }

    if rank == 0:
        checkpoint_name = f"./models/{params.run_name}/{params.run_name}_{params.esm_pretrained}_finetune_{checkpoint_key}.pt"
        torch.save(save_dict, checkpoint_name)

def mp_main(rank, world_size, params):
    params.rank = rank
    params.world_size = world_size
    
    logg = BaseLogger(name = params.run_name, level = params.log_level, rank = params.rank, use_tensorboard = True)
    if params.debug:
        logg.set_level(logging.DEBUG)
    else:
        logg.set_level(params.log_level)
    
    logg.debug(f"Running DDP on rank {rank}")
    setup(params, logg)
    
    logg.info("Loading datasets")
    train_dataset, valid_dataset = get_datasets(params, logg)
    logg.info(f"Train dataset size: {len(train_dataset)} | Valid dataset size: {len(valid_dataset)}")
    train_sampler, valid_sampler, train_loader, valid_loader = get_dataloaders(train_dataset, valid_dataset, params, logg)
    
    logg.info("Building model")
    model = get_model(params, logg)
    
    if not hasattr(params, "lora_l2_coeff"):
            params.lora_l2_coeff = params.l2_coeff
    opt_params = add_weight_decay(model, params.l2_coeff, params.lora_l2_coeff)
    optimizer = optim.AdamW(opt_params, lr=params.lr)
    scheduler = get_cosine_decay_schedule(optimizer)
    
    if hasattr(params, "checkpoint"):
        logg.info(f"Loading checkpoint: {params.checkpoint}")
        
        checkpoint = torch.load(params.checkpoint, map_location = torch.device(params.rank))
        if "optimizer_state_dict:" in checkpoint:
            checkpoint["optimizer_state_dict"] = checkpoint["optimizer_state_dict:"]
        
        model.module.load_state_dict(checkpoint["model_state_dict"])
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
        start_epoch = checkpoint["epoch"] + 1
        try:
            best_valid_loss = checkpoint["best_valid_loss"]
            best_valid_aupr = checkpoint["best_valid_aupr"]
        except KeyError:
            best_valid_loss = torch.inf
            best_valid_aupr = 0
    else:
        start_epoch = 0
        best_valid_loss = torch.inf
        best_valid_aupr = 0
    for epoch in range(start_epoch, params.n_epoch):
        
        train_sampler.set_epoch(epoch)
        valid_sampler.set_epoch(epoch)
        
        with timer(f"[Train] [Epoch {epoch}]", logg, rank):
            model, optimizer, scheduler, _ = train_epoch(epoch, model, optimizer, scheduler, train_loader, params, logg)
        
        with timer(f"[Validation] [Epoch {epoch}]", logg, rank):
            model, valid_metrics = valid_epoch(epoch, model, valid_loader, params, logg)
        
        if valid_metrics["valid/loss"] < best_valid_loss:
            logg.info(f"New best validation loss: {valid_metrics['valid/loss']:.6f}")
            logg.info("Saving new best model")
            best_valid_loss = valid_metrics["valid/loss"]
            save_model("best_loss", model, optimizer, scheduler, epoch, best_valid_loss, best_valid_aupr, params, logg)
            
        if valid_metrics["valid/aupr"] > best_valid_aupr:
            logg.info(f"New best validation AUPR: {valid_metrics['valid/aupr']:.6f}")
            logg.info("Saving new best model")
            best_valid_aupr = valid_metrics["valid/aupr"]
            save_model("best_aupr", model, optimizer, scheduler, epoch, best_valid_loss, best_valid_aupr, params, logg)
            
        if (epoch + 1) % params.save_every == 0:
            logg.info(f"Saving checkpoint (Epoch {epoch})")
            save_model(f"checkpoint{epoch}", model, optimizer, best_valid_loss, best_valid_aupr, scheduler, epoch, params, logg)
    
    save_model("last", model, optimizer, scheduler, epoch, best_valid_loss, best_valid_aupr, params, logg)
            
    cleanup()