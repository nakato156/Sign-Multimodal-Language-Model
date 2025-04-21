import torch

from setup_train import *

if __name__ == "__main__":
    llm_tools = Tools()

    DataPath, ModelPath, h5File, csvFile = setup_paths()
    embedding_layer, tokenizer, vocab_size, d_model = load_llm_components()
    print(f"Vocab size: {vocab_size}, d_model: {d_model}")

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

    train_dataset, validation_dataset = prepare_datasets(h5File, csvFile, tokenizer, model_parameters["T_size"], train_config["train_ratio"],device)
    train_dataloader, val_dataloader = create_dataloaders(train_dataset, validation_dataset, train_config["batch_size"], 4)
    model = build_model(model_parameters["input_size"], model_parameters["T_size"], model_parameters["output_size"], device)  
    
    sort_by_keyword = 'cuda_time_total'
 
    run_training(model_parameters, train_dataloader, val_dataloader, embedding_layer, model)