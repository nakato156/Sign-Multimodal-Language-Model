from src.study.studies import complete_objetive
from .setup_train import *
import optuna

if __name__ == "__main__":
    # Load LLM components
    embedding_layer, tokenizer, vocab_size, d_model = load_llm_components()

    # Paths
    DataPath, ModelPath, h5File, csvFile = setup_paths()

    # Parameters
    modelParameters = {
        "input_size": 543*2,
        "output_size": 3072,
        "model_version": 32,
        "checkpoint": 1,
        "model_dir": ModelPath,
        "device": "cuda" if torch.cuda.is_available() else "cpu",
        "epochs": 10,
        "log_interval": 2,
        "checkpoint_interval": 5,
        "batch_size": 32,
        "frame_clips": 15 * 35,
        "train_ratio": 0.8,
        "validation_ratio": 0.2
    }

    train_dataset, validation_dataset = prepare_datasets(
        h5File,
        csvFile,
        tokenizer,
        max_seq_len=modelParameters["frame_clips"],
        train_ratio=modelParameters["train_ratio"],
        device=modelParameters["device"]
    )
    
    # Create DataLoaders
    train_dataloader, val_dataloader = create_dataloaders(train_dataset, validation_dataset, modelParameters["batch_size"])


    args_train = {
        "modelParameters": modelParameters,
        "train_dataloader": train_dataloader,
        "val_dataloader": val_dataloader,
        "embedding_layer": embedding_layer,
    }

    # Define the study name and storage
    study_name = "complete_study"
    storage_name = "sqlite:///{}.db".format(study_name)
    
    study = optuna.create_study(study_name=study_name, storage=storage_name, direction="minimize", load_if_exists=True)

    # Optimize the objective function
    study.optimize(lambda trial: complete_objetive(trial, **args_train), n_trials=30)

    # Print the best trial
    print("Best trial:")
    print("\tValue: {}".format(study.best_trial.value))
    print("\tBest Hyperparameters:", study.best_params)
    print("\tParams: ")
    for key, value in study.best_trial.params.items():
        print("\t{}: {}".format(key, value))