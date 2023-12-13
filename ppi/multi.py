# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
import os
import socket
import torch.distributed as dist

from omegaconf import OmegaConf
from logutil import BaseLogger

MAXPORT = 12500

def is_port_in_use(port: int) -> bool:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        return s.connect_ex(('localhost', port)) == 0
    
def find_free_port(orig_port: int, logg: BaseLogger) -> int:
    port = orig_port
    while is_port_in_use(port) and port < MAXPORT:
        logg.debug(f"Port {port} in use, trying next port")
        port += 1
    logg.info(f"Port {orig_port} passed, using port {port}")
    return port

def setup(params: OmegaConf):
    
    rank = params.rank
    world_size = params.world_size
    
    os.environ['MASTER_ADDR'] = params.MASTER_ADDR
    os.environ['MASTER_PORT'] = str(params.MASTER_PORT)

    # initialize the process group
    dist.init_process_group("nccl", rank=rank, world_size=world_size)

def cleanup():
    dist.destroy_process_group()