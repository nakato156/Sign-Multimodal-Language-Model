import torch
from .setup_train import *
from argparse import ArgumentParser

def parse_arg():
    parser = ArgumentParser(description="Train the model.")
    parser.add_argument("--epochs", type=int, default=100, help="Number of epochs to train.")
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size for training.")
    parser.add_argument("--checkpoint_interval", type=int, default=5, help="Interval for saving checkpoints.")
    parser.add_argument("--log_interval", type=int, default=2, help="Interval for logging.")
    return parser.parse_args()

if __name__ == "__main__":
    args = parse_arg()

    DataPath, ModelPath, h5File, csvFile = setup_paths()
    embedding_layer, tokenizer, vocab_size, d_model = load_llm_components()

    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Parameters and Saving Parameteres
    model_parameters = {
        "device": device,
        "input_size": 543*2,
        "output_size": 3072,
        "T_size": 15 * 35,
        "hidden_size": 1024,
        "nhead": 16, 
        "ff_dim": 2048,
        "n_layers": 6, 
    }

    train_config = {
        "learning_rate": 0.0023818221335158807,
        "model_version": 35,
        "checkpoint": 1,
        "model_dir": ModelPath,
        "epochs": args.epochs,
        "log_interval": args.log_interval,
        "checkpoint_interval": args.checkpoint_interval,
        "batch_size": args.batch_size,
        "train_ratio": 0.8,
        "validation_ratio": 0.2,
        "device": device,
    }

    train_dataset, validation_dataset = prepare_datasets(h5File, csvFile, tokenizer, model_parameters["T_size"], train_config["train_ratio"],device)
    train_dataloader, val_dataloader = create_dataloaders(train_dataset, validation_dataset, train_config["batch_size"], 4)
    model = build_model(**model_parameters)  
    
    sort_by_keyword = 'cuda_time_total'
 
    run_training(train_config, train_dataloader, val_dataloader, embedding_layer, model)