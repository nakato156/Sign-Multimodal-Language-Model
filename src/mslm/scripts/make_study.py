import optuna
from mslm.utils.setup_train import load_llm_components, setup_paths, prepare_datasets, create_dataloaders
from mslm.studies import complete_objective

def run(
    n_trials: int = 30,
    batch_size: int = 32,
    train_ratio: float = 0.8,
):
    # setup
    embedding_layer, tokenizer, _, _ = load_llm_components()
    _, Model_path, h5_file, csv_file = setup_paths()
    
    # params
    validation_ratio = 1 - train_ratio
    model_parameters = {
        "input_size": 543*2,
        "output_size": 3072,
        "model_version": 32,
        "checkpoint": 1,
        "model_dir": Model_path,
        "device": "cuda",
        "epochs": 10,
        "log_interval": 2,
        "checkpoint_interval": 5,
        "batch_size": 32,
        "frame_clips": 15 * 35,
        "train_ratio": train_ratio,
        "validation_ratio": validation_ratio
    }

    # datasets
    tr_ds, val_ds = prepare_datasets(h5_file, csv_file, tokenizer,
                                     model_parameters["frame_clips"],
                                     train_ratio,
                                     model_parameters["device"])
    tr_dl, val_dl = create_dataloaders(tr_ds, val_ds, batch_size)
    # optuna
    storage = f"sqlite:///../outputs/studies/complete_study.db"
    study = optuna.create_study(study_name="complete_study",
                                storage=storage,
                                direction="minimize",
                                load_if_exists=True)
    study.optimize(
        lambda t: complete_objective(t,
            model_parameters=model_parameters,
            train_dataloader=tr_dl,
            val_dataloader=val_dl,
            embedding_layer=embedding_layer
        ),
        n_trials=n_trials
    )
    print(study.best_trial)
