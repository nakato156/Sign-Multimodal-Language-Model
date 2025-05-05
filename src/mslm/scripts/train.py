import torch
from mslm.utils.setup_train import load_llm_components, setup_paths
from mslm.utils import create_dataloaders, build_model, run_training, prepare_datasets, ConfigLoader

def run(
    epochs: int,
    batch_size: int,
    checkpoint_interval: int,
    log_interval: int,
    train_ratio: float = 0.8,
):
    _, _, h5_file, csv_file = setup_paths()
    embedding_layer, tokenizer, _, _ = load_llm_components()
    device = "cuda" if torch.cuda.is_available() else "cpu"


    model_parameters = ConfigLoader("config/model/config.toml").load_config()
    model_parameters.update({
        "device": device if model_parameters.get("device") == "auto" else model_parameters.get("device", device),
        "input_size": 543 * 2,
        "output_size": 3072,
        "T_size": 15 * 35,
    })
    
    # --- config de entrenamiento ---
    train_config = ConfigLoader("config/training/train_config.toml").load_config()
    train_ratio = train_config.get("train_ratio", train_ratio)
    train_config.update({
        "checkpoint": 1,
        "learning_rate": train_config.get("learning_rate", 0.00238),
        "epochs": epochs if epochs else train_config.get("epochs", 100),
        "batch_size": batch_size if batch_size else train_config.get("batch_size", 32),
        "checkpoint_interval": checkpoint_interval if checkpoint_interval else train_config.get("checkpoint_interval", 5),
        "log_interval": log_interval if log_interval else train_config.get("log_interval", 2),
        "train_ratio": train_ratio,
        "validation_ratio": round(1 - train_ratio, 2),
        "device": device if model_parameters.get("device") == "auto" else model_parameters.get("device", device),
    })
    
    tr_ds, val_ds = prepare_datasets(h5_file, csv_file, tokenizer,
                                     model_parameters["T_size"],
                                     train_ratio, device)
    tr_dl, val_dl = create_dataloaders(tr_ds, val_ds, batch_size, num_workers=4)

    model = build_model(**model_parameters)
    run_training(train_config, tr_dl, val_dl, embedding_layer, model)

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Train a model.")
    parser.add_argument("--epochs", type=int, default=100, help="Number of epochs to train.")
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size for training.")
    parser.add_argument("--checkpoint_interval", type=int, default=5, help="Interval for saving checkpoints.")
    parser.add_argument("--log_interval", type=int, default=2, help="Interval for logging training progress.")
    args = parser.parse_args()

    run(args.epochs, args.batch_size, args.checkpoint_interval, args.log_interval)