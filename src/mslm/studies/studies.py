from ..training import Trainer
from ..models import Imitator

def lr_objetive(trial, train_dataloader, val_dataloader, embedding_layer, **params):
    learning_rate = trial.suggest_float("lr", 1e-5, 1e-1, log=True)
    params["learning_rate"] = learning_rate
    modelParameters = params["modelParameters"]

    model = Imitator(
        input_size=modelParameters["input_size"],
        T_size=modelParameters["frame_clips"],
        output_size=modelParameters["output_size"],
        nhead=32,
        ff_dim=4096,
        n_layers=8,
        pool_dim=128
    ).to(modelParameters["device"])

    trainer = Trainer(model, train_dataloader, val_dataloader, embedding_layer, **params)
    _, val_loss = trainer.train()
    return val_loss

def complete_objective(trial, train_dataloader, val_dataloader, embedding_layer, modelParameters):
    hidden_size = trial.suggest_categorical("hidden_size", [512, 1024])
    nhead = trial.suggest_categorical("nhead", [16, 32, 64])
    ff_dim = trial.suggest_int("ff_dim", 1024, 3072, step=256)
    n_layers = trial.suggest_categorical("n_layers", [6, 8, 10, 12])
    learning_rate = trial.suggest_float("lr", 1e-5, 1e-1, log=True)
    modelParameters["learning_rate"] = learning_rate
    
    model = Imitator(
        input_size=modelParameters["input_size"],
        T_size=modelParameters["frame_clips"],
        output_size=modelParameters["output_size"],
        hidden_size=hidden_size,
        nhead=nhead,
        ff_dim=ff_dim,
        n_layers=n_layers,
        pool_dim=128
    ).to(modelParameters["device"])

    trainer = Trainer(model, train_dataloader, val_dataloader, embedding_layer, **modelParameters)
    _, val_loss = trainer.train()
    return val_loss