import torch.nn as nn
from .PositionalEncoding import PositionalEncoding
import torch.nn.functional as F

class Imitator(nn.Module):
    def __init__(self, input_size=1086, hidden_size=512, T_size=525, output_size=3072):
        super().__init__()

        self.linear = nn.Linear(input_size, hidden_size)
        nn.init.xavier_uniform_(self.linear.weight)
        
        self.norm1 = nn.LayerNorm(hidden_size)

        encoder_layer = nn.TransformerEncoderLayer(d_model=hidden_size, nhead=32, dim_feedforward=4096, batch_first=True)
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=8)
        self.pe = PositionalEncoding(d_model=hidden_size)
        
        self.linear2 = nn.Linear(hidden_size, output_size)
        nn.init.xavier_uniform_(self.linear2.weight)
        
        self.pooling = nn.Linear(T_size, 128)
        nn.init.xavier_uniform_(self.pooling.weight)

        self.norm_out = nn.LayerNorm(output_size)

    def forward(self, x):
        # x -> [batch_size, T, input_size]
        B, T, D, C = x.shape
        x = x.view(B, T,  D * C)
        x = F.relu(self.linear(x))
        x = self.norm1(x)

        x = self.pe(x)
        x = self.transformer(x)

        x = F.relu(self.linear2(x))

        x = x.transpose(1, 2)
        x = F.relu(self.pooling(x))
        x = x.transpose(1, 2)

        return self.norm_out(x)