import torch.nn as nn
from .components import STGCN
import torch.nn.functional as F

class Imitator(nn.Module):
    def __init__(
        self,
        input_size: int = 1086,
        hidden_size: int = 512,
        T_size: int = 525,
        output_size: int = 3072,
        nhead: int = 32,
        ff_dim: int = 1024,
        n_layers: int = 8,
        pool_dim: int = 128,
    ):
        super().__init__()

        self.cfg = {
            "input_size": input_size,
            "hidden_size": hidden_size,
            "T_size": T_size,
            "output_size": output_size,
            "nhead": nhead,
            "ff_dim": ff_dim,
            "n_layers": n_layers,
            "pool_dim": pool_dim
        }
        
        self.stgcn = STGCN(
            in_channels=input_size,
            out_channels=output_size,
            num_blocks=2,
            kernel_size_spatial=25,
            kernel_size_temporal=9
        )
        
        self.temporal_adjuster = nn.Sequential(
            nn.Linear(T_size, pool_dim),
            nn.ReLU()
        )        

        self.linear_out = nn.Linear(output_size, output_size)

    def forward(self, x):
        # x -> [batch_size, T, input_size]
        B, T, D, C = x.shape
        x = x.view(B, T, D * C)     # [B, T, 1086]
        x = x.unsqueeze(1)          # [B, 1, T, 1086]
        x = x.permute(0, 3, 1, 2)   # [B, 1086, T, 1]

        x = self.stgcn(x)
        x = F.relu(x)
        # print("stgcn: ", x.shape)
        
        x = x.view(B, -1, T)
        # print("view: ", x.shape)        

        x = self.temporal_adjuster(x)  # [B, hidden, 128]
        x = x.transpose(1, 2)
        # print("temporal_adjuster: ", x.shape)

        x = self.linear_out(x)
        # print("linear_out: ", x.shape)

        return x