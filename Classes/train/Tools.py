import torch
from unsloth import FastLanguageModel
import gc
import os
import json

import pandas as pd
from tqdm import tqdm

import torch.nn as nn
from torch import autocast, GradScaler
from torch.nn.utils.rnn import pad_sequence
from torch.optim import AdamW
from torch.utils.tensorboard import SummaryWriter
import torch.nn.functional as F

import nvtx
from .EarlyStopping import EarlyStopping

class ImitatorLoss(nn.Module):

    def __init__(self, alpha=1.0, beta=1.0):
        """
        Combina L2 (MSE) y CosineEmbeddingLoss:
        total_loss = alpha * L2 + beta * (1 - cos_similarity)
        """
        super().__init__()

        self.alpha = alpha
        self.beta = beta
        self.cos_sim = nn.CosineSimilarity(dim=-1)

    def forward(self, pred, target):
        l2 = F.mse_loss(pred, target)

        cosine_sim = self.cos_sim(pred, target)
        cosine_loss = 1 - cosine_sim.mean()

        return (self.alpha * l2) + (self.beta * cosine_loss)

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
    
    def get_logits_from_embedding(self, embeddings) -> str:
        logits_from_embeddings = embeddings @ self.embedding_layer.weight.T
        return F.log_softmax(logits_from_embeddings)

    @torch.compile
    @nvtx.annotate("Start Training", color="green")
    def train(
        self,
        model,
        train_loader,
        val_loader,
        modelVersions,
        modelDir, 
        epochs=100,
        log_interval=10,
        checkpoint_interval=5,
        learning_rate=1e-4,
        device = "cuda"
    ):
        
        optimizer = AdamW(model.parameters(), lr=learning_rate)
        criterion = ImitatorLoss(alpha=1.0, beta=1.0)
        criterion_compiled = torch.compile(
            criterion, backend="inductor", mode="default"
        )

        writer = SummaryWriter("imitator_report")
        scaler = GradScaler(device=device)

        modelPath =  os.path.join(modelDir, "checkpoints", str(modelVersions["version"]), str(modelVersions["checkpoint"]))

        df = pd.DataFrame(columns=["epoch", "loss"])
        early_stopping = EarlyStopping(modelPath)

        for epoch in tqdm(range(epochs), desc="Entrenando", colour="green"):
            model.train()
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
                            loss = criterion_compiled(output, embeddings)
                    else:
                        with autocast(device_type=device):
                            output = model(data)
                            loss = criterion_compiled(output, embeddings)
                
                with nvtx.annotate("Backward Pass", color="blue"):            
                    total_loss += loss.detach()
                    final_loss = total_loss.item()

                writer.add_scalar("Loss/train", loss, epoch)

                with nvtx.annotate("Update", color="blue"):
                    scaler.scale(loss).backward()
                    
                    # scaler.unscale_(optimizer)
                    # torch.nn.utils.clip_grad_norm_(model.parameters(), 0.5)

                    scaler.step(optimizer)
                    scaler.update()

                    #loss.backward()
                    #optimizer.step()

                #del output, loss, data, embeddings
            torch.cuda.empty_cache()
                #del output, loss, data, embeddings
            torch.cuda.empty_cache()

            if epoch % log_interval == 0:
                df.loc[len(df)] = [epoch, f"{final_loss/len(train_loader):.4f}"]
                
            if (epoch % checkpoint_interval == 0 and epoch != 0) or (epoch == epochs-1):
                self.saveModel(model, os.path.join(modelPath, str(epoch)))
                
            print("\nEpoch: ", epoch, ".\t Total loss: ", final_loss/len(train_loader))
        
            with nvtx.annotate("Prueba de Validacion", color="blue"):
                with torch.no_grad():
                    model.eval()
                    val_loss = 0
                    for data, input_ids in val_loader:
                        data = data.to(device)
                        input_ids = input_ids.to(device)

                        output = model(data)
                        loss = criterion_compiled(output.to(dtype=torch.bfloat16), input_ids)
                        val_loss += loss.detach()

                        #del output, data, embeddings, cos_sim
                        #del output, data, embeddings, cos_sim
                    torch.cuda.empty_cache()
                        
                        
                    final_val_loss = val_loss.item() / len(val_loader)
                    print(f"Validation Loss: {final_val_loss}" )
                    
                    early_stopping(final_val_loss)
                    if early_stopping.stop:
                        self.saveModel(model, path=os.path.join(modelPath, str(epoch + 1)))
                        return
        
        writer.flush()
        writer.close()

    def collate_fn(self, batch):
        data = pad_sequence([item[0] for item in batch])
        data = data.permute(1, 0, 2, 3)

        embeddings = torch.stack([item[1] for item in batch])

        # embeddings = pad_sequence([item[1] for item in batch], padding_value=128004)
        # embeddings = embeddings.permute(1, 0, 2)

        # print(f"Data has NaN: {torch.isnan(data).any()}")
        # print(f"Embeddings has NaN: {torch.isnan(embeddings).any()}")

        if self.LOG:
            print(f"Data: {data.size()}, Embeddings: {embeddings.size()}")

        return data, embeddings

    def saveModel(self, model, path):
        if not os.path.exists(path):
            os.makedirs(path)
        torch.save(model, f"{path}/model.pt")

    def loadModel(self, path):
        torch.load(path)

    def saveParameters(self, path, parameters):
        parameterPath = os.path.join(
            path,
            "checkpoints",
            str(parameters["model"]["version"]),
            str(parameters["model"]["checkpoint"]),
            str(parameters["model"]["version"]),
            str(parameters["model"]["checkpoint"]),
        )

        if (
            not os.path.exists(
                parameterPath
            )
        ):
            os.makedirs(parameterPath)

        with open(os.path.join(parameterPath, "parameters.json"), 'w') as f:
            json.dump(parameters, f)