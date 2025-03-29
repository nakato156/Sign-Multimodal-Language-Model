import os
import pandas as pd
from tqdm import tqdm
import torch

import torch.multiprocessing as mp
import torch.nn as nn
from torch import autocast, GradScaler
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader, random_split
from torch.optim import AdamW
from torch.utils.tensorboard import SummaryWriter

#Imported Classes
from Classes.SignDataLoader import SignDataLoader
from Classes.Imitator import Imitator
from Classes.KeypointDataset import KeypointDataset
from Classes.Tools import Tools

#Profilers
from torch.profiler import profile, ProfilerActivity
import memory_profiler
import nvtx

# sudo env "PATH=$PATH" nsys profile --trace cuda,osrt,nvtx --gpu-metrics-device=all --cuda-memory-usage true --force-overwrite true --output profile_run_v1 --gpu-metrics-frequency=500 python train_CUDA.py

PROFILE = False
LOG = False

@nvtx.annotate("Start Training", color="green")
@memory_profiler.profile
def train(
    model,
    train_loader,
    modelVersions,
    modelDir, 
    epochs=100,
    log_interval=10,
    checkpoint_interval=5,
    learning_rate=1e-4,
    device = "cuda"
):
    model.train()

    optimizer = AdamW(model.parameters(), lr=learning_rate)
    criterion = nn.CosineSimilarity(dim=2, eps=1e-6)
    writer = SummaryWriter("imitator_report")
    scaler = GradScaler(device=device)

    df = pd.DataFrame(columns=["epoch", "loss"])

    for epoch in tqdm(range(epochs), desc="Entrenando", colour="green"):
        total_loss = 0
        for data, embeddings in train_loader:
            optimizer.zero_grad(set_to_none=True)

            with torch.autograd.profiler.record_function("Data to CUDA"):
                data = data.to(device)
                embeddings = embeddings.to(device)
            
            #Change to bfloat16 if the GPU used is with Ampere Architecture or Higher
            if torch.cuda.get_device_capability()[0] >= 8:
                with autocast(device_type=device, dtype=torch.bfloat16):
                    output = model(data)
            else:
                with autocast(device_type=device):
                    output = model(data)
            
            cos_sim = criterion(output, embeddings)
            loss = (1 - cos_sim).mean()
            total_loss += loss.item()

            writer.add_scalar("Loss/train", loss, epoch)

            scaler.scale(loss).backward()
            
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 0.5)

            scaler.step(optimizer)
            scaler.update()

            #loss.backward()
            #optimizer.step()

            del output, loss, data, embeddings, cos_sim
            torch.cuda.empty_cache()

        if epoch % log_interval == 0:
            df.loc[len(df)] = [epoch, f"{total_loss/len(train_loader):.4f}"]
            
        if epoch % checkpoint_interval == 0 and epoch != 0 and epoch == epochs-1: 
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': total_loss
            }, os.path.join(modelDir, "checkpoints", str(modelVersions["version"]), str(modelVersions["checkpoint"]), str(epoch)))

        print("\nEpoch: ", epoch, ".\t Total loss: ", total_loss/len(train_loader))
    
    writer.flush()
    writer.close()

def collate_fn(batch):
    data = pad_sequence([item[0] for item in batch])
    data = data.permute(1, 0, 2, 3)

    embeddings = pad_sequence([item[1] for item in batch], padding_value=128004)
    embeddings = embeddings.permute(1, 0, 2)

    # print(f"Data has NaN: {torch.isnan(data).any()}")
    # print(f"Embeddings has NaN: {torch.isnan(embeddings).any()}")

    if LOG:
        print(f"Data: {data.size()}, Embeddings: {embeddings.size()}")

    return data, embeddings

def trace_handler(p):
    output = p.key_averages().table(sort_by=sort_by_keyword, row_limit=10)
    print(output)
    p.export_chrome_trace("/tmp/trace_" + str(p.step_num) + ".json")

if __name__ == "__main__":
    tools = Tools()

    embedding_layer, tokenizer = tools.getLLM()
    vocab_size, d_model = embedding_layer.weight.size()

    print(f"Vocab size: {vocab_size}, d_model: {d_model}")

    DataPath = os.path.join(os.getcwd(), os.pardir, "data", "dataset2")
    ModelPath = os.path.join(os.getcwd(), "model")
    h5File = os.path.join(DataPath, "keypoints.h5")
    csvFile = os.path.join(DataPath, "meta.csv")

    # parameters
    modelParameters = {
        "model": {
            "version": 2,
            "checkpoint": 1
        },
        "input_size": 543*2,
        "output_size": 3072,
        "learning_rate": 2e-4,
        "device": "cuda" if torch.cuda.is_available() else "cpu",
        "epochs": 100,
        "logIntervals": 1,
        "checkpointIntervals": 20,
        "batchSize": 16,
        "frameClips": 15 * 35,
        "train_ratio": 0.8,
        "validation_ratio": 0.2
    }

    keypointReader = KeypointDataset(h5Path=h5File, labelsCSV=csvFile, max_seq_len=modelParameters["frameClips"])
    dataset = SignDataLoader(tokenizer, embedding_layer, keypointReader, modelParameters["device"])

    keypointReaderSize = len(keypointReader)
    train_size = int(keypointReaderSize * modelParameters["train_ratio"])
    validation_size = keypointReaderSize - train_size


    train_dataset, validation_dataset = random_split(dataset, [train_size, validation_size])
    train_dataloader = DataLoader(train_dataset, batch_size=modelParameters["batchSize"], shuffle=True, collate_fn=collate_fn)
    validation_dataloader = DataLoader(validation_dataset, batch_size=modelParameters["batchSize"], shuffle=True, collate_fn=collate_fn)

    # model
    model = Imitator(input_size=modelParameters["input_size"], output_size=modelParameters["output_size"], d_model=d_model).to(modelParameters["device"])

    print(model)
    
    sort_by_keyword = 'cuda_time_total'
 
    if PROFILE:
        with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA], record_shapes=True, with_stack=True, profile_memory=True) as p:
            train(model, train_dataloader, epochs=modelParameters["epochs"], log_interval=modelParameters["logIntervals"], learning_rate=modelParameters["learning_rate"], modelVersions=modelParameters["model"], modelDir=ModelPath, checkpoint_interval=modelParameters["checkpointIntervals"])
    else:
        train(model, train_dataloader, epochs=modelParameters["epochs"], log_interval=modelParameters["logIntervals"], learning_rate=modelParameters["learning_rate"], modelVersions=modelParameters["model"], modelDir=ModelPath, checkpoint_interval=modelParameters["checkpointIntervals"])

    #p.export_chrome_trace("profile_trace.json")
    #print(p.key_averages().table(sort_by="cuda_time_total", row_limit=10))    

    