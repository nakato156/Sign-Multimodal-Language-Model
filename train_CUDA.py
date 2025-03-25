import pandas as pd
from tqdm import tqdm

from unsloth import FastLanguageModel

import numpy as np
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.optim import Adam
from torch.utils.tensorboard import SummaryWriter
from torch.nn.utils.rnn import pad_sequence
import torch

from Classes.SignDataLoader import SignDataLoader
from Classes.Imitator import Imitator
from Classes.KeypointDataset import KeypointDataset

import gc
import os
import nvtx
from torch.profiler import profile, record_function, ProfilerActivity

#sudo env "PATH=$PATH" nsys profile --show-output=true --gpu-metrics-devices=all --gpu-metrics-frequency=1000 -o profile_output python train.py

@nvtx.annotate("Start Training", color="green")
def train(model, train_loader, epochs=100, log_interval=10, learning_rate=1e-4):
    model.train()

    optimizer = Adam(model.parameters(), lr=learning_rate)
    criterion = nn.MSELoss()
    writer = SummaryWriter("imitator_report")

    df = pd.DataFrame(columns=["epoch", "loss"])

    for epoch in tqdm(range(epochs), desc="Entrenando", colour="green"):
        total_loss = 0
        for data, embeddings in train_loader:
            #print(data.shape) #[12, 1050, 543, 2]
            #print(embeddings.shape) #[12, 128, 3072]
            data = data.to("cuda")
            embeddings = embeddings.to("cuda")

            output = model(data)
            #print(output.shape)
            loss = criterion(output, embeddings)

            #print("Model Output: ", output)
            #print("Model Embeddings: ", embeddings)

            total_loss += loss
            writer.add_scalar("Loss/train", loss, epoch)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        
        if epoch % log_interval == 0:
            df.loc[len(df)] = [epoch, f"{total_loss/len(train_loader):.4f}"]
            print("Epoch: ", epoch, ".\t Total loss: ", total_loss/len(train_loader))
    
    writer.flush()
    writer.close()

def returnLLM():
    max_seq_length = 2048 * 2
    dtype = None
    load_in_4bit = True

    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name = "unsloth/Llama-3.2-3B-Instruct",
        max_seq_length = max_seq_length,
        dtype = dtype,
        load_in_4bit = load_in_4bit,
    )

    # Obtener capa de embeddings
    embedding_layer = model.get_input_embeddings()

    del model
    
    while True:
        torch.cuda.empty_cache()
        if gc.collect() == 0:
            break

    return embedding_layer, tokenizer

def collate_fn(batch):
    data = pad_sequence(item[0] for item in batch)
    data = data.permute(1, 0, 2, 3)

    embeddings = pad_sequence(item[1] for item in batch)
    embeddings = embeddings.permute(1, 0, 2)

    print(f"Data: {data.size()}, Embeddings: {embeddings.size()}")

    return data, embeddings

def trace_handler(p):
    output = p.key_averages().table(sort_by=sort_by_keyword, row_limit=10)
    print(output)
    p.export_chrome_trace("/tmp/trace_" + str(p.step_num) + ".json")

if __name__ == "__main__":
    embedding_layer, tokenizer = returnLLM()
    device_embed = "cpu"

    all_embeddings = embedding_layer.weight.data.to(device_embed)  # Tensor de forma [vocab_size, d_model]
    vocab_size, d_model = all_embeddings.shape

    print(f"Vocab size: {vocab_size}, d_model: {d_model}")

    DataPath = os.path.join(os.getcwd(), os.pardir, "data", "dataset2")
    h5File = os.path.join(DataPath, "keypoints.h5")
    csvFile = os.path.join(DataPath, "meta.csv")

    LIMITS_SECONDS = 30

    # parameters
    input_size = 543*2 # cantidad de puntos x 2
    output_size = 3072
    learning_rate = 2e-4
    device = "cuda" if torch.cuda.is_available() else "cpu"

    keypointReader = KeypointDataset(h5Path=h5File, labelsCSV=csvFile, max_seq_len=LIMITS_SECONDS * 35)
    dataset = SignDataLoader(tokenizer, embedding_layer, keypointReader, device)
    dataloader = DataLoader(dataset, batch_size=12, shuffle=True, collate_fn=collate_fn)

    # model
    model = Imitator(input_size=input_size, output_size=output_size, d_model=d_model).to(device)

    print(model)
    
    sort_by_keyword = 'cuda_time_total'

    with profile(
        activities=[ProfilerActivity.CUDA, ProfilerActivity.CPU],
        schedule=torch.profiler.schedule(
            wait=1,
            warmup=1,
            active=2),
        on_trace_ready=trace_handler,
        record_shapes=True, 
        with_stack=True
    ) as prof:
        train(model, dataloader, epochs=1, log_interval=10, learning_rate=learning_rate)
        
