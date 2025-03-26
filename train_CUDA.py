import pandas as pd
from tqdm import tqdm

import torch.nn as nn
from torch.utils.data import DataLoader
from torch.optim import Adam
from torch.utils.tensorboard import SummaryWriter
from torch.nn.utils.rnn import pad_sequence
import torch

from Classes.SignDataLoader import SignDataLoader
from Classes.Imitator import Imitator
from Classes.KeypointDataset import KeypointDataset
from Classes.Tools import Tools
from torch.profiler import profile, ProfilerActivity

import os
import nvtx

# sudo env "PATH=$PATH" nsys profile --trace cuda,osrt,nvtx --gpu-metrics-device=all --cuda-memory-usage true --force-overwrite true --output profile_run_v1 --gpu-metrics-frequency=500 python train_CUDA.py

LOG = False

@nvtx.annotate("Start Training", color="green")
def train(
    model,
    train_loader,
    modelVersions,
    modelDir, 
    epochs=100,
    log_interval=10,
    checkpoint_interval=5,
    learning_rate=1e-4
):
    model.train()

    optimizer = Adam(model.parameters(), lr=learning_rate)
    criterion = nn.MSELoss()
    writer = SummaryWriter("imitator_report")

    df = pd.DataFrame(columns=["epoch", "loss"])

    for epoch in tqdm(range(epochs), desc="Entrenando", colour="green"):
        total_loss = 0
        for data, embeddings in train_loader:
            if LOG: 
                print(data.shape) #[12, 1050, 543, 2]
            if LOG: 
                print(embeddings.shape) #[12, 128, 3072]

            with torch.autograd.profiler.record_function("Data to CUDA"):
                data = data.to("cuda")
                embeddings = embeddings.to("cuda")

            with torch.autograd.profiler.record_function("Data to Model"):
                output = model(data)
                if torch.cuda.get_device_capability()[0] >= 8:
                    output = output.to(torch.bfloat16)
            with torch.autograd.profiler.record_function("Loss Computation"):
                loss = criterion(output, embeddings)

            if LOG: 
                print(output.shape)
                print(output[0][0])

                print("Model Output: ", output)
                print("Model Embeddings: ", embeddings)

            total_loss += loss
            writer.add_scalar("Loss/train", loss, epoch)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        
        if epoch % log_interval == 0:
            df.loc[len(df)] = [epoch, f"{total_loss/len(train_loader):.4f}"]
            
        if epoch % checkpoint_interval == 0 and model != 0: 
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': total_loss
            }, os.path.join(modelDir, "checkpoints", str(modelVersions["version"]), str(modelVersions["checkpoint"]), str(epoch)))

        print("Epoch: ", epoch, ".\t Total loss: ", total_loss/len(train_loader))
    
    writer.flush()
    writer.close()

def collate_fn(batch):
    data = pad_sequence(item[0] for item in batch)
    data = data.permute(1, 0, 2, 3)

    embeddings = pad_sequence((item[1] for item in batch), padding_value=128004)
    embeddings = embeddings.permute(1, 0, 2)

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
            "version": 1,
            "checkpoint": 1
        },
        "input_size": 543*2,
        "output_size": 3072,
        "learning_rate": 2e-4,
        "device": "cuda" if torch.cuda.is_available() else "cpu",
        "epochs": 1,
        "logIntervals": 10,
        "checkpointIntervals": 5,
        "batchSize": 12,
        "frameClips": 15 * 35
    }

    keypointReader = KeypointDataset(h5Path=h5File, labelsCSV=csvFile, max_seq_len=modelParameters["frameClips"])
    dataset = SignDataLoader(tokenizer, embedding_layer, keypointReader, modelParameters["device"])
    dataloader = DataLoader(dataset, batch_size=modelParameters["batchSize"], shuffle=True, collate_fn=collate_fn)

    # model
    model = Imitator(input_size=modelParameters["input_size"], output_size=modelParameters["output_size"], d_model=d_model).to(modelParameters["device"])

    print(model)
    
    sort_by_keyword = 'cuda_time_total'

    with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA], record_shapes=True, with_stack=True, profile_memory=True) as p:
        train(model, dataloader, epochs=modelParameters["epochs"], log_interval=modelParameters["logIntervals"], learning_rate=modelParameters["learning_rate"], modelVersions=modelParameters["model"], modelDir=ModelPath)
    
    p.export_chrome_trace("profile_trace.json")
    print(p.key_averages().table(sort_by="cuda_time_total", row_limit=10))    