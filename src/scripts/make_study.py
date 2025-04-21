from src.study.studies import complete_objetive
from setup_train import *
import optuna

if __name__ == "__main__":
    # Load LLM components
    embedding_layer, tokenizer, vocab_size, d_model = load_llm_components()

    # Paths
    DataPath, ModelPath, h5File, csvFile = setup_paths()

    # Parameters
    modelParameters = {
        "model_version": 32,
        "checkpoint": 1,
        "model_dir": ModelPath,
        "input_size": 543*2,
        "output_size": 3072,
        "learning_rate": 5e-4,
        "device": "cuda" if torch.cuda.is_available() else "cpu",
        "epochs": 20,
        "logIntervals": 2,
        "checkpointIntervals": 5,
        "batchSize": 32,
        "frameClips": 15 * 35,
        "train_ratio": 0.8,
        "validation_ratio": 0.2
    }

    train_dataset, validation_dataset = prepare_datasets(
        h5File,
        csvFile,
        tokenizer,
        max_seq_len=modelParameters["frameClips"],
        train_ratio=modelParameters["train_ratio"],
        device=modelParameters["device"]
    )
    
    # Create DataLoaders
    train_dataloader, val_dataloader = create_dataloaders(train_dataset, validation_dataset, modelParameters["batchSize"])


    args_train = {
        "modelParameters": modelParameters,
        "train_dataloader": train_dataloader,
        "val_dataloader": val_dataloader,
        "embedding_layer": embedding_layer,
    }

    # Define the study name and storage
    study_name = "complete_study_2"
    storage_name = "sqlite:///{}.db".format(study_name)
    
    study = optuna.create_study(study_name=study_name, storage=storage_name, direction="minimize", load_if_exists=True)

    # Optimize the objective function
    study.optimize(lambda trial: complete_objetive(trial, **args_train), n_trials=20)

    # Print the best trial
    print("Best trial:")
    print("\tValue: {}".format(study.best_trial.value))
    print("\tBest Hyperparameters:", study.best_params)
    print("\tParams: ")
    for key, value in study.best_trial.params.items():
        print("\t{}: {}".format(key, value))