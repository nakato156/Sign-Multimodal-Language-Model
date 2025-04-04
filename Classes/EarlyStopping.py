import numpy as np
import torch 
from pathlib import Path

class EarlyStopping:
    def __init__(self, path_checkpoints:Path, patience:int=10, verbose:bool=False):
        self.best_loss = float("inf")
        self.patience = patience
        self.verbose = verbose
        self.path_checkpoints = Path(path_checkpoints)
        self.stop = False

    def __call__(self,val_loss, model, checkpoint_name:str):
        if val_loss == np.nan:
            if self.verbose: print("Giorgio desgraciado")
        
        if val_loss < self.best_loss:
            self.best_loss = val_loss
            self.patience -= 1
        
        if self.patience == 0:
            if self.verbose: print("Early Stopping")
            torch.save(model.state_dict(), self.path_checkpoints / f"best_early_{checkpoint_name}")
            self.stop = True