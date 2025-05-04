import os
import torch

from torch.utils.data import DataLoader, random_split

#Imported Classes
from mslm.models import Imitator
from mslm.training import Trainer
from mslm.dataloader import KeypointDataset, SignDataLoader, collate_fn
from mslm.utils.llm_tools import Tools
from mslm.utils.paths import path_vars

#Profilers
from torch.profiler import profile, ProfilerActivity

def setup_paths():
    """Define y retorna las rutas necesarias para datos y modelos."""
    data_path = path_vars.data_path
    model_path = path_vars.model_path
    h5_file = path_vars.h5_file
    csv_file = path_vars.csv_file
    return data_path, model_path, h5_file, csv_file

def load_llm_components():
    """Carga el embedding layer y el tokenizer usando llm_tools."""
    llm_tools = Tools()
    embedding_layer, tokenizer = llm_tools.getLLM()
    vocab_size, d_model = embedding_layer.weight.size()
    print(f"Vocab size: {vocab_size}, d_model: {d_model}")
    return embedding_layer, tokenizer, vocab_size, d_model

def prepare_datasets(h5File, csvFile, tokenizer, max_seq_len, train_ratio, device):
    """Carga el dataset base, lo envuelve y lo divide en entrenamiento y validación."""
    keypoint_reader = KeypointDataset(h5Path=h5File, labelsCSV=csvFile, max_seq_len=max_seq_len)
    dataset = SignDataLoader(tokenizer, keypoint_reader, device)

    keypoint_readerSize = len(keypoint_reader)
    train_size = int(keypoint_readerSize * train_ratio)
    validation_size = keypoint_readerSize - train_size

    train_dataset, validation_dataset = random_split(dataset, [train_size, validation_size])
    print(f"Train size:\t{len(train_dataset)}\nValidation size:\t{len(validation_dataset)}")
    return train_dataset, validation_dataset

def create_dataloaders(train_dataset, validation_dataset, batch_size, num_workers=4):
    """Crea y retorna los DataLoaders para entrenamiento y validación."""
    train_dataloader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
        persistent_workers=True,
        collate_fn=collate_fn
    )
    val_dataloader = DataLoader(
        validation_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
        persistent_workers=True,
        collate_fn=collate_fn
    )
    return train_dataloader, val_dataloader

def build_model(input_size, T_size, output_size, device, compile=True, **kwargs):
    """Construye, compila y retorna el modelo Imitator."""
    model = Imitator(input_size=input_size, T_size=T_size, output_size=output_size, **kwargs).to(device)
    if compile:
        model = torch.compile(model, backend="inductor", mode="reduce-overhead")
    print(model)
    print(f"{sum(p.numel() for p in model.parameters())/1e6:.2f} M parameters")
    return model

def run_training(params, train_dataloader, val_dataloader, embedding_layer, model, PROFILE=False):
    """Configura y ejecuta el entrenamiento."""
    trainer = Trainer(model, train_dataloader, val_dataloader, embedding_layer, **params)
    trainer.ckpt_mgr.save_params(params)

    if PROFILE:
        print("Starting training with profiling...")
        with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
                    record_shapes=True,
                    with_stack=True,
                    profile_memory=True) as p:
            trainer.train()
    else:
        print("Starting training...")
        return trainer.train()