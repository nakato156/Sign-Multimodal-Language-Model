import os
from pathlib import Path
import gc

from setup_train import *

from unsloth import FastLanguageModel

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch import autocast # , GradScaler
import torch.nn.functional as F

from Classes.dataloader import KeypointDataset, SignDataLoader, collate_fn
from Classes.train.Imitator import Imitator
from Classes.train.PositionalEncoding import PositionalEncoding
from Classes.utils.early_stopping import EarlyStopping
from tqdm import tqdm

LOG = False

llama_model, tokenizer = FastLanguageModel.from_pretrained(
    model_name="unsloth/Llama-3.2-3B-Instruct",
    max_seq_length=4096,
    dtype=None,
    load_in_4bit=True,
)


torch.serialization.add_safe_globals([Imitator, PositionalEncoding])
checkpoint_name = "./model/checkpoints/8/1/bueno/model.pt"

def train(mslm, lm_head, train_dataloader, val_dataloader, llama_embed_layer, device="cuda", epochs=50, learning_rate=5e-5, 
          checkpoint_dir=None, checkpoint_interval=5, early_stopping=None):
    
    # scaler = GradScaler(device=device)
    criterion = nn.KLDivLoss(reduction="batchmean")
    optimizer = torch.optim.AdamW(mslm.parameters(), lr=learning_rate)
    
    total_steps  = epochs * len(train_dataloader)
    warmup_steps = int(0.1 * total_steps) 
    
    scheduler = CosineAnnealingLR(
        optimizer,
        T_max=total_steps - warmup_steps,
        eta_min=learning_rate * 1e-2
    )

    for epoch in tqdm(range(epochs), desc="Training Epochs", colour="green"):
        epoch_loss = 0.0
        mslm.train()
        for keypoints, input_ids in train_dataloader:
            keypoints = keypoints.to(device, non_blocking=True)
            input_ids = input_ids.to(device, non_blocking=True)
            embeddings = llama_embed_layer(input_ids)

            optimizer.zero_grad()
            with autocast(device_type=device, dtype=torch.bfloat16):
                pred_embed = mslm(keypoints).to(dtype=torch.bfloat16)
                pred_logits = lm_head(pred_embed)
                pred_log_probs = F.log_softmax(pred_logits, dim=-1)

                with torch.no_grad():
                    true_logits = lm_head(embeddings)
                    true_log_probs = F.softmax(true_logits, dim=-1)
                
                loss = criterion(
                    pred_log_probs.view(-1),
                    true_log_probs.view(-1)
                )
                
                epoch_loss += loss.detach()
                loss.backward()
                
                optimizer.step()
                scheduler.step()
                
                # scaler.scale(loss).backward()
                # scaler.step(optimizer)
                # scaler.update()
        
        epoch_loss /= len(train_dataloader)
        
        # Validation loss
        val_loss = 0.0
        mslm.eval()
        with torch.no_grad():
            for keypoints, input_ids in val_dataloader:
                keypoints = keypoints.to(device, non_blocking=True)
                input_ids = input_ids.to(device, non_blocking=True)
                embeddings = llama_embed_layer(input_ids)

                pred_embed = mslm(keypoints).to(dtype=torch.bfloat16, non_blocking=True)
                pred_logits = lm_head(pred_embed)
                pred_log_probs = F.log_softmax(pred_logits, dim=-1)

                with torch.no_grad():
                    true_logits = lm_head(embeddings)
                    true_log_probs = F.softmax(true_logits, dim=-1)
                
                loss = criterion(
                    pred_log_probs.view(-1),
                    true_log_probs.view(-1)
                ).item()
                val_loss += loss
        val_loss /= len(val_dataloader)
        # scheduler.step(val_loss)
        
        if epoch % 2 == 0:
            print(f"Epoch [{epoch+1}/{epochs}], Loss: {epoch_loss:.4f}\tValidation Loss: {val_loss:.4f}")

        # Early stopping
        if early_stopping is not None:
            early_stopping(val_loss)
            if early_stopping.stop:
                print(f"Early stopping at epoch {epoch+1}")
                break

        # Periodic checkpoint saving
        if checkpoint_dir and (epoch + 1) % checkpoint_interval == 0:
            checkpoint_path = Path(checkpoint_dir) / str(epoch + 1) 
            
            if not checkpoint_path.exists():
                checkpoint_path.mkdir(parents=True)

            checkpoint_file = checkpoint_path / f"model.pt"
            torch.save(mslm, checkpoint_file)
            print(f"Checkpoint saved at {checkpoint_file}")

def main(modelParameters, mslm, llama_embed_layer, lm_head):
    # Paths
    DataPath, ModelPath, h5File, csvFile = setup_paths()

    train_dataset, validation_dataset = prepare_datasets(
        h5File,
        csvFile,
        tokenizer,
        max_seq_len=modelParameters["frameClips"],
        train_ratio=modelParameters["train_ratio"],
        device=modelParameters["device"],
    )

    # Create DataLoaders
    train_dataloader, val_dataloader = create_dataloaders(
        train_dataset, validation_dataset, modelParameters["batchSize"]
    )

    # Early stopping and checkpoint setup
    checkpoint_dir = Path("model/checkpoints") / str(modelParameters["model"]["version"]) / str(modelParameters["model"]["checkpoint"])
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    early_stopping = EarlyStopping(patience=5, threshold=0.002, verbose=True)

    while True:
        torch.cuda.empty_cache()
        if not gc.collect():
            break

    train(
        mslm, lm_head, train_dataloader, val_dataloader,
        llama_embed_layer=llama_embed_layer,
        device=modelParameters["device"],
        epochs=modelParameters["epochs"],
        learning_rate=modelParameters["learning_rate"],
        checkpoint_dir=checkpoint_dir,
        checkpoint_interval=modelParameters["checkpointIntervals"],
        early_stopping=early_stopping
    )

if __name__ == '__main__':
    torch.backends.cudnn.benchmark = True

    ModelPath = Path(os.path.join(os.getcwd(), "model"))
    mslm = torch.load(checkpoint_name, weights_only=False).to("cuda")
    
    llama_embed_layer = llama_model.get_input_embeddings()
    lm_head = llama_model.lm_head.to("cuda")
    for param in lm_head.parameters():
        param.requires_grad = False

    del llama_model

    modelParameters = {
        "model": {
            "version": "fine-tuning",
            "checkpoint": 8,
            "from_checkpoint": checkpoint_name
        },
        "input_size": 543*2,
        "output_size": 3072,
        "learning_rate": 5e-5,
        "device": "cuda" if torch.cuda.is_available() else "cpu",
        "epochs": 100,
        "logIntervals": 20,
        "checkpointIntervals": 5,
        "batchSize": 16,
        "frameClips": 15 * 35,
        "train_ratio": 0.8,
        "validation_ratio": 0.2
    }

    try:
        main(modelParameters, mslm, llama_embed_layer, lm_head)
    finally:
        model_params = modelParameters["model"]
        folder = ModelPath / "checkpoints" / str(model_params["version"]) / str(model_params["checkpoint"]) 
        if not folder.exists():
            folder.mkdir(parents=True)
        torch.save(mslm, folder / "final_model.pt")