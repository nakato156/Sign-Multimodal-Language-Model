import os
import torch

from torch.utils.data import DataLoader, random_split

#Imported Classes
from Classes.train import Imitator
from Classes.train.trainer import Trainer
from Classes.dataloader import KeypointDataset, SignDataLoader, collate_fn
from Classes.utils.llm_tools import Tools
#Profilers
from torch.profiler import profile, ProfilerActivity

PROFILE = False
LOG = False

def trace_handler(p):
    output = p.key_averages().table(sort_by=sort_by_keyword, row_limit=10)
    print(output)
    p.export_chrome_trace("/tmp/trace_" + str(p.step_num) + ".json")

if __name__ == "__main__":
    #mp.set_start_method("spawn", force=True)
    llm_tools = Tools()

    embedding_layer, tokenizer = llm_tools.getLLM()
    vocab_size, d_model = embedding_layer.weight.size()

    print(f"Vocab size: {vocab_size}, d_model: {d_model}")

    DataPath = os.path.join(os.getcwd(), os.pardir, "data", "dataset2")
    ModelPath = os.path.join(os.getcwd(), "model")
    h5File = os.path.join(DataPath, "keypoints.h5")
    csvFile = os.path.join(DataPath, "meta.csv")

    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Parameters and Saving Parameteres
    model_parameters = {
        "input_size": 543*2,
        "output_size": 3072,
        "ff_dim": 1792,
        "n_layers": 12,
        "T_size": 15 * 35,
        "device": device,
    }

    train_config = {
        "learning_rate": 4.6e-5,
        "model_version": 34,
        "checkpoint": 1,
        "model_dir": ModelPath,
        "epochs": 100,
        "log_interval": 2,
        "checkpoint_interval": 5,
        "batch_size": 32,
        "train_ratio": 0.8,
        "validation_ratio": 0.2,
        "device": device,
    }

    keypointReader = KeypointDataset(h5Path=h5File, labelsCSV=csvFile, max_seq_len=model_parameters["T_size"])
    dataset = SignDataLoader(tokenizer, keypointReader, device)

    keypointReaderSize = len(keypointReader)
    train_size = int(keypointReaderSize * train_config["train_ratio"])
    validation_size = keypointReaderSize - train_size

    train_dataset, validation_dataset = random_split(dataset, [train_size, validation_size])
    print(f"Train size:\t{len(train_dataset)}\nValidation size:\t{len(validation_dataset)}")
    
    train_dataloader = DataLoader(
        train_dataset,
        batch_size=train_config["batch_size"],
        shuffle=True,
        num_workers=4,
        pin_memory=True,
        persistent_workers=True,
        collate_fn=collate_fn
    )
    val_dataloader = DataLoader(
        validation_dataset,
        batch_size=train_config["batch_size"],
        shuffle=False,
        num_workers=4,
        pin_memory=True,
        persistent_workers=True,
        collate_fn=collate_fn
    )
    
    # model
    model = Imitator(**model_parameters).to(device)
    model = torch.compile(model, backend="inductor", mode="default")
    
    trainer = Trainer(model, train_dataloader, val_dataloader, embedding_layer, **train_config)
    trainer.ckpt_mgr.save_params(model_parameters)

    print(model)
    print(f"{(sum(p.numel() for p in model.parameters())/1e6):2f}", 'M parameters')
    
    sort_by_keyword = 'cuda_time_total'
 
    if PROFILE:
        with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA], record_shapes=True, with_stack=True, profile_memory=True) as p:
            trainer.train()
    else:
        trainer.train()