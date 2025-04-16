import torch
from torch.nn.utils.rnn import pad_sequence

def collate_fn(batch, LOG=False):
    data = pad_sequence([item[0] for item in batch])
    data = data.permute(1, 0, 2, 3)

    embeddings = torch.stack([item[1] for item in batch])

    if LOG:
        print(f"Data: {data.size()}, Embeddings: {embeddings.size()}")

    return data, embeddings