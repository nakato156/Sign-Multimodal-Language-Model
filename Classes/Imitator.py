import torch
import torch.nn as nn

from .PositionalEncoding import PositionalEncoding

class Imitator(nn.Module):
    def __init__(self, input_size=1086, output_size=128256, d_model=2048):
        super().__init__()
        self.linear = nn.Linear(input_size, 512)
        encoder_layer = nn.TransformerEncoderLayer(d_model=512, nhead=16, dim_feedforward=4096, batch_first=True)
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=16)
        self.pe = PositionalEncoding(d_model=512)
        self.linear2 = nn.Linear(512, output_size)
        self.pooling = nn.Linear(525,128)

    def forward(self, x):
        # x -> [batch_size, T, input_size]
        x = x.view(x.shape[0],x.shape[1],543*2)
        x = self.linear(x) 
        x = self.pe(x)
        x = self.transformer(x)
        x = self.linear2(x)
        x = x.transpose(1,2)
        x = self.pooling(x)
        x = x.transpose(1,2)
        x = x.to(torch.bfloat16)
        return x