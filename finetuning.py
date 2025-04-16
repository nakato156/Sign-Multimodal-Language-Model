import os
from pathlib import Path
import gc

from unsloth import FastLanguageModel
import torch
import torch.nn as nn
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader, random_split

from Classes.train.KeypointDataset import KeypointDataset
from Classes.train.SignDataLoader import SignDataLoader
from Classes.train.Imitator import Imitator
from Classes.train.PositionalEncoding import PositionalEncoding
from Classes.train.EarlyStopping import EarlyStopping
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

def get_logits(lm_head, embeddings):
    return lm_head(embeddings)

def collate_fn(batch):
    data = pad_sequence([item[0] for item in batch])
    data = data.permute(1, 0, 2, 3)

    embeddings = torch.stack([item[1] for item in batch])

    if LOG:
        print(f"Data: {data.size()}, Embeddings: {embeddings.size()}")
    return data, embeddings

def train(mslm, lm_head, train_dataloader, val_dataloader, device="cuda", epochs=50, learning_rate=5e-5, 
          checkpoint_dir=None, checkpoint_interval=5, early_stopping=None):
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(mslm.parameters(), lr=learning_rate)
    mslm = mslm.to(device)
    mslm.train()

    for epoch in tqdm(range(epochs), desc="Training Epochs", colour="green"):
        epoch_loss = 0.0
        for keypoints, embeddings in train_dataloader:
            keypoints = keypoints.to(device)
            embeddings = embeddings.to(device)

            optimizer.zero_grad()
            pred_embed = mslm(keypoints).to(dtype=torch.bfloat16)
            pred_logits = get_logits(lm_head, pred_embed)
            true_logits = get_logits(lm_head, embeddings)
            true_targets = torch.argmax(true_logits, dim=-1)
            loss = criterion(pred_logits.view(-1, pred_logits.size(-1)), true_targets.view(-1))
            epoch_loss += loss.detach()
            loss.backward()
            optimizer.step()
        
        epoch_loss /= len(train_dataloader)
        
        # Validation loss
        val_loss = 0.0
        mslm.eval()
        with torch.no_grad():
            for keypoints, embeddings in val_dataloader:
                keypoints = keypoints.to(device)
                embeddings = embeddings.to(device)
                pred_embed = mslm(keypoints).to(dtype=torch.bfloat16)
                pred_logits = get_logits(lm_head, pred_embed)
                true_logits = get_logits(lm_head, embeddings)
                true_targets = torch.argmax(true_logits, dim=-1)
                loss = criterion(pred_logits.view(-1, pred_logits.size(-1)), true_targets.view(-1))
                val_loss += loss.detach()
        val_loss /= len(val_dataloader)
        
        if epoch % 2 == 0:
            print(f"Epoch [{epoch+1}/{epochs}], Loss: {epoch_loss:.4f}\tValidation Loss: {val_loss:.4f}")

        mslm.train()

        # Early stopping
        if early_stopping is not None:
            early_stopping(val_loss.item())
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
    DataPath = os.path.join(os.getcwd(), os.pardir, "data", "dataset2")
    h5File = os.path.join(DataPath, "keypoints.h5")
    csvFile = os.path.join(DataPath, "meta.csv")

    keypointReader = KeypointDataset(h5Path=h5File, labelsCSV=csvFile, max_seq_len=modelParameters["frameClips"])
    dataset = SignDataLoader(tokenizer, llama_embed_layer, keypointReader, modelParameters["device"])

    keypointReaderSize = len(keypointReader)
    train_size = int(keypointReaderSize * modelParameters["train_ratio"])
    validation_size = keypointReaderSize - train_size

    train_dataset, validation_dataset = random_split(dataset, [train_size, validation_size])
    train_dataloader = DataLoader(train_dataset, batch_size=modelParameters["batchSize"], shuffle=True, collate_fn=collate_fn)
    val_dataloader = DataLoader(validation_dataset, batch_size=modelParameters["batchSize"], shuffle=False, collate_fn=collate_fn)

    del llama_embed_layer
    del keypointReader
    del dataset

    # Early stopping and checkpoint setup
    checkpoint_dir = Path("model/checkpoints") / str(modelParameters["model"]["version"]) / str(modelParameters["model"]["checkpoint"])
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    early_stopping = EarlyStopping(path_checkpoints=checkpoint_dir, patience=10, threshold=0.002, verbose=True)

    while True:
        torch.cuda.empty_cache()
        if not gc.collect():
            break

    train(
        mslm, lm_head, train_dataloader, val_dataloader,
        device=modelParameters["device"],
        epochs=modelParameters["epochs"],
        learning_rate=modelParameters["learning_rate"],
        checkpoint_dir=checkpoint_dir,
        checkpoint_interval=modelParameters["checkpointIntervals"],
        early_stopping=early_stopping
    )

if __name__ == '__main__':
    ModelPath = Path(os.path.join(os.getcwd(), "model"))
    mslm = torch.load(checkpoint_name, weights_only=False)
    llama_embed_layer = llama_model.get_input_embeddings()
    lm_head = llama_model.lm_head

    del llama_model

    modelParameters = {
        "model": {
            "version": 20,
            "checkpoint": 4,
            "from_checkpoint": checkpoint_name
        },
        "input_size": 543*2,
        "output_size": 3072,
        "learning_rate": 5e-4,
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
        torch.save(mslm, folder / "model.pt")