import os
import sys
import tempfile
import torch
import torch.distributed as dist
import torch.distributed.rpc as rpc
import torch.nn as nn
import torch.optim as optim
import torch.multiprocessing as mp

from torch.nn.parallel import DistributedDataParallel as DDP

def setup(rank, world_size):
    dist.init_process_group("nccl", rank=rank, world_size=world_size)
    torch.cuda.set_device(0)

def ddp_worker(rank, world_size):
    os.environ["RANK"] = str(rank)
    os.environ["WORLD_SIZE"] = str(world_size)
    
if __name__ == "__main__":
    setup()

    rpc.init_rpc(
        name=f"worker{rank}",
        rank=rank,
        world_size=5
    )