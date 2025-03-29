import torch
from unsloth import FastLanguageModel
import gc
import os

import pandas as pd
from tqdm import tqdm

import torch.multiprocessing as mp
import torch.nn as nn
from torch import autocast, GradScaler
from torch.nn.utils.rnn import pad_sequence
from torch.optim import AdamW
from torch.utils.tensorboard import SummaryWriter

import memory_profiler
import nvtx

class Tools:
    def __init__(self, LOG=False):
        super().__init__()
        self.executeLLM()
        self.LOG = LOG


    def getLLM(self):
        return self.embedding_layer, self.tokenizer

    def executeLLM(self):
        max_seq_length = 2048 * 2

        dtype = None
        load_in_4bit = True

        model, self.tokenizer = FastLanguageModel.from_pretrained(
            model_name="unsloth/Llama-3.2-3B-Instruct",
            max_seq_length=max_seq_length,
            dtype=dtype,
            load_in_4bit=load_in_4bit,
        )

        # Obtener capa de embeddings
        self.embedding_layer = model.get_input_embeddings()

        del model

        while True:
            torch.cuda.empty_cache()
            if gc.collect() == 0:
                break

    @nvtx.annotate("Start Training", color="green")
    @memory_profiler.profile
    def train(
        self,
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

                with nvtx.annotate("Data to CUDA", color="yellow"):
                    data = data.to(device)
                    embeddings = embeddings.to(device)

                with nvtx.annotate("Training", color="blue"):            
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

    def collate_fn(self, batch):
        data = pad_sequence([item[0] for item in batch])
        data = data.permute(1, 0, 2, 3)

        embeddings = pad_sequence([item[1] for item in batch], padding_value=128004)
        embeddings = embeddings.permute(1, 0, 2)

        # print(f"Data has NaN: {torch.isnan(data).any()}")
        # print(f"Embeddings has NaN: {torch.isnan(embeddings).any()}")

        if self.LOG:
            print(f"Data: {data.size()}, Embeddings: {embeddings.size()}")

        return data, embeddings
