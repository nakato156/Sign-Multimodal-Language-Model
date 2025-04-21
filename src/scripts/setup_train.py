import os
import torch

from torch.utils.data import DataLoader, random_split

#Imported Classes
from src.train import Imitator
from src.train.trainer import Trainer
from src.dataloader import KeypointDataset, SignDataLoader, collate_fn
from src.utils.llm_tools import Tools
#Profilers
from torch.profiler import profile, ProfilerActivity

PROFILE = False
LOG = False
sort_by_keyword = "cuda_time_total"

def trace_handler(p):
    output = p.key_averages().table(sort_by=sort_by_keyword, row_limit=10)
    print(output)
    p.export_chrome_trace("/tmp/trace_" + str(p.step_num) + ".json")

def setup_paths():
    """Define y retorna las rutas necesarias para datos y modelos."""
    cwd = os.getcwd()
    DataPath = os.path.join(cwd, os.pardir, "data", "dataset2")
    ModelPath = os.path.join(cwd, "model")
    h5File = os.path.join(DataPath, "keypoints.h5")
    csvFile = os.path.join(DataPath, "meta.csv")
    return DataPath, ModelPath, h5File, csvFile

def load_llm_components():
    """Carga el embedding layer y el tokenizer usando llm_tools."""
    llm_tools = Tools()
    embedding_layer, tokenizer = llm_tools.getLLM()
    vocab_size, d_model = embedding_layer.weight.size()
    print(f"Vocab size: {vocab_size}, d_model: {d_model}")
    return embedding_layer, tokenizer, vocab_size, d_model

def prepare_datasets(h5File, csvFile, tokenizer, max_seq_len, train_ratio, device):
    """Carga el dataset base, lo envuelve y lo divide en entrenamiento y validación."""
    keypointReader = KeypointDataset(h5Path=h5File, labelsCSV=csvFile, max_seq_len=max_seq_len)
    dataset = SignDataLoader(tokenizer, keypointReader, device)

    keypointReaderSize = len(keypointReader)
    train_size = int(keypointReaderSize * train_ratio)
    validation_size = keypointReaderSize - train_size

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
        model = torch.compile(model, backend="inductor", mode="default")
    print(model)
    print(f"{sum(p.numel() for p in model.parameters())/1e6:.2f} M parameters")
    return model

def run_training(params, train_dataloader, val_dataloader, embedding_layer, model):
    """Configura y ejecuta el entrenamiento."""
    trainer = Trainer(model, train_dataloader, val_dataloader, embedding_layer, **params)
    trainer.ckpt_mgr.save_params(params)

    if PROFILE:
        print("Starting training with profiling...")
        with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
                     record_shapes=True,
                     with_stack=True,
                     profile_memory=True,
                     on_trace_ready=trace_handler) as p: # Usar on_trace_ready es más eficiente según el gemini
            trainer.train()
    else:
        print("Starting training...")
        return trainer.train()