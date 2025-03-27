import torch.nn as nn
from .PositionalEncoding import PositionalEncoding
import torch.nn.functional as F

class Imitator(nn.Module):
    def __init__(self, input_size=1086, output_size=128256, d_model=2048):
        super().__init__()
        self.linear = nn.Linear(input_size, 512)
        nn.init.xavier_uniform_(self.linear.weight)
        
        self.norm1 = nn.LayerNorm(512)

        encoder_layer = nn.TransformerEncoderLayer(d_model=512, nhead=32, dim_feedforward=4096, batch_first=True)
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=8)
        self.pe = PositionalEncoding(d_model=512)
        
        self.linear2 = nn.Linear(512, output_size)
        nn.init.xavier_uniform_(self.linear2.weight)
        
        self.pooling = nn.Linear(525, 128)
        nn.init.xavier_uniform_(self.pooling.weight)

        self.norm_out = nn.LayerNorm(3072)

    def forward(self, x):
        # x -> [batch_size, T, input_size]
        x = x.view(x.shape[0], x.shape[1], 543*2)
        x = F.relu(self.linear(x))
        x = self.norm1(x)

        x = self.pe(x)
        x = self.transformer(x)

        x = F.relu(self.linear2(x))

        x = x.transpose(1, 2)
        x = F.relu(self.pooling(x))
        x = x.transpose(1, 2)

        return self.norm_out(x)