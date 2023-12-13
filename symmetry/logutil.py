# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
import os
import re
import logging
import json
import pandas as pd
import typing as T
from io import StringIO
from torch.utils.tensorboard import SummaryWriter
from time import time

from omegaconf import OmegaConf

RELEVANT_HPARAMS = [
    "do_peft"
    "n_epoch",
    "epoch_size",
    "lr",
    "l2_coeff",
    "lora_l2_coeff",
    "ema",
    "lora_r",
    "lora_alpha",
    "lora_dropout",
    "lora_bias",
    "esm_pretrained",
    "esm_clsf_hidden_size",
]

CFG_START_REGEX = re.compile("^.*Loaded params: \{$")
CFG_END_REGEX = re.compile("^}.* from .*\.yaml$")
TRAIN_LOSS_REGEX = re.compile(".*\[Train\](?: \[Epoch \d+\])? Step (\d+)/(\d+) \| Loss: (\d+\.\d+)$")
METRIC_REGEX = re.compile(".*\[Metric\](?: \[Step \d+\])? ([a-z]+)/([a-zA-z_\d]+) = (\d+\.\d+)$")
STARTING_EPOCH_REGEX = re.compile(".*\[Epoch (\d+)\] starting...")

def check_rank(f):
    def wrapper(*args, **kwargs):
        if args[0].rank == 0:
            return f(*args, **kwargs)
    return wrapper

def add_rank(f):
    def wrapper(obj, msg, *args, **kwargs):
        rank = obj.rank
        msg = f"[rank:{rank}] {msg}"
        return f(obj, msg, *args, **kwargs)
    return wrapper

class BaseLogger:
    def __init__(
            self, 
            name: str,
            level: int = logging.INFO, 
            rank: int = 0,
            use_tensorboard: bool = False,
        ):

        self.level_ = level
        self.rank = rank
        self.name = name
        self.logger = logging.getLogger(name)
        self.logger.setLevel(level)
        self.logger.propagate = False
        self.logger.handlers = []
        self.logger.addHandler(logging.StreamHandler())

        self.log_dir = f'logs/{name}'

        if not os.path.exists(self.log_dir):
            os.makedirs(self.log_dir, exist_ok=True)

        self.add_file_logger()
        if use_tensorboard and (self.rank == 0):
            self.add_tb()
            
        self._format(self.logger)
        self._format(self.file_only_logger)

    @property
    def level(self):
        return self.level_
    
    @level.setter
    def level(self, level: int):
        self.level_ = level
        self.logger.setLevel(level)
        
        if hasattr(self, "file_only_logger"):
            self.file_only_logger.setLevel(level)
    
    def _format(
            self,
            logger,
            fmt: str = "[%(asctime)s.%(msecs)03d] [%(name)s:%(levelname)s] %(message)s",
            datefmt: str = '%Y-%m-%d %H:%M:%S',
        ):
        formatter = logging.Formatter(fmt, datefmt)

        for h in logger.handlers:
            h.setFormatter(formatter)
    
    def set_rank(self, rank: int):
        self.rank = rank
    
    def add_file_logger(self, fname: str = "log.txt"):
        self.logger.addHandler(logging.FileHandler(
            filename=os.path.join(self.log_dir, fname),
            mode='a+',
        ))
        
        self.file_only_logger = logging.getLogger(f"{self.name}_file_only")
        self.file_only_logger.setLevel(logging.INFO)
        self.file_only_logger.propagate = False
        self.file_only_logger.handlers = []
        self.file_only_logger.addHandler(logging.FileHandler(
            filename=os.path.join(self.log_dir, fname),
            mode='a+',
        ))
        
    @check_rank
    def add_tb(self):
        self.writer = SummaryWriter(
            log_dir=self.log_dir,
        )
    
    def set_level(self, level: int):
        self.level = level
    
    @add_rank
    def debug(self, msg: str):
        self.logger.debug(msg)

    @check_rank
    @add_rank
    def info(self, msg: str, file_only = False):
        if file_only and hasattr(self, "file_only_logger"):
            self.file_only_logger.info(msg)
        else:
            self.logger.info(msg)
        

    @add_rank
    def warning(self, msg: str):
        self.logger.warning(msg)

    @add_rank
    def error(self, msg: str):
        self.logger.error(msg)

    @add_rank
    def critical(self, msg: str):
        self.logger.critical(msg)

    @check_rank
    def metric(self, k: str, v: float, step: int, log_also = True):
        if log_also:
            self.info(f"[Metric] [Step {step}] {k} = {v:.6f}")
        if hasattr(self, "writer"):
            self.writer.add_scalar(k, v, step or 0)
        
def parse_log(
    log_path: str,
):

    train_history = []
    metric_df = []
    cfg_lines = []

    in_config = False
    with open(log_path, "r") as f:
        for line in f:
            line = line.strip()
            
            if CFG_END_REGEX.match(line):
                cfg_lines.append("}")
                in_config = False
            elif in_config:
                cfg_lines.append(line)
            if CFG_START_REGEX.match(line):
                cfg_lines.append("{")
                in_config = True
            
            match_e = STARTING_EPOCH_REGEX.match(line)
            match_train = TRAIN_LOSS_REGEX.match(line)
            match_met = METRIC_REGEX.match(line)
            
            if match_e:
                curr_epoch = int(match_e.group(1))
            
            if match_train:
                train_step, epoch_size, train_loss = match_train.group(1,2,3)
                step = (int(curr_epoch) * int(epoch_size)) + int(train_step)
                train_history.append((step, float(train_loss)))
            
            if match_met:
                metric_group, metric_name, metric_value = match_met.group(1,2,3)
                metric_df.append((metric_group, metric_name, curr_epoch, float(metric_value)))
                
    train_history = pd.DataFrame(train_history, columns = ["Step", "Loss"])
    metric_df = pd.DataFrame(metric_df, columns = ["Subset", "Metric", "Epoch", "Value"])
    cfg = OmegaConf.create(json.load(StringIO("\n".join(cfg_lines))))
    
    return train_history, metric_df, cfg

class timer:
    
    def __init__(self, name: str, logg: T.Optional[BaseLogger] = None, rank: int = 0):
        self.name = name
        self.logg = logg
        self.rank = rank
    
    @check_rank
    def _write(self, msg: str):
        if self.logg is None:
            print(msg)
        else:
            self.logg.info(msg)
    
    def __enter__(self):
        self._write(f"{self.name} starting...")
        self.start = time()
        
    def __exit__(self, *args):
        self.end = time()
        interval = self.end - self.start
        msg = f"{self.name} elapsed time: {interval:4f} seconds"
        
        self._write(msg)