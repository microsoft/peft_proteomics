# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

import torch
import torch.multiprocessing as mp
import argparse
import os
import json
from omegaconf import OmegaConf

from logutil import BaseLogger
from multi import find_free_port
from modeling import mp_main

if __name__ == "__main__":

    parser = argparse.ArgumentParser("Homooligomer training script")

    parser.add_argument("--run_name", type=str, required=True, help="Unique ID for run")
    parser.add_argument("--config", type=str,  required=True, help="Path to config file")
    parser.add_argument("--debug", action="store_true", help="Debug mode")
    parser.add_argument("--devices", nargs="+", type=int, help="GPU devices to use")
    parser.add_argument("--checkpoint", type=str, help="Path to checkpoint to resume training")

    args = parser.parse_args()
    params = OmegaConf.load(args.config)
    params.run_name = args.run_name
    
    os.makedirs(f"./logs/{params.run_name}", exist_ok=True)
    os.makedirs(f"./models/{params.run_name}", exist_ok=True)
    
    params.debug = args.debug
    if args.checkpoint is not None:
        params.checkpoint = args.checkpoint

    logg = BaseLogger(params.run_name, level=params.log_level)
    logg.info(f"Loaded params: {json.dumps(OmegaConf.to_container(params),indent=2)} from {args.config}")
    
    logg.info(f"Running with CUDA_VISIBLE_DEVICES={os.getenv('CUDA_VISIBLE_DEVICES')}")
    
    world_size = torch.cuda.device_count()
    params.MASTER_PORT = find_free_port(params.MASTER_PORT, logg)
    
    mp.freeze_support()
    try:
        mp.spawn(mp_main, args=(world_size,params), nprocs=world_size, join=True)
    except Exception as e:
        logg.error(f"Exception: {e}")
        raise e