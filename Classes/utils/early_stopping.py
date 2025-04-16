import numpy as np
from pathlib import Path

class EarlyStopping:
    def __init__(self, path_checkpoints:Path, patience:int=10, threshold:float=0.002, verbose:bool=False):
        self.best_loss = float("inf")
        self.patience = patience
        self._patience = patience
        self.verbose = verbose
        self.path_checkpoints = Path(path_checkpoints)
        self.stop = False
        self.treshold = threshold

    def __call__(self, val_loss):
        if np.isnan(val_loss):
            if self.verbose: print("Giorgio desgraciado \nIgnorando epoch")
            return
        
        if val_loss < self.best_loss:
            self.best_loss = val_loss
            self.patience = self._patience
        if val_loss > self.best_loss or abs(val_loss - self.best_loss) < self.treshold:
            self.patience -= 1
        
        if self.patience == 0:
            if self.verbose: print("Early Stopping")
            self.stop = True