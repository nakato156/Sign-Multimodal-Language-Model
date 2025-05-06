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

if __name__ == "__main__":
    rpc.init_rpc(
        name = "data_provider",
        rank="rank",
        world_size=5
    )