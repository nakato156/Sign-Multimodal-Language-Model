from pathlib import Path
from os import pardir, getcwd
import tomllib

class ConfigLoader:
    def __init__(self, config_path: str|Path, base_path: str|Path = None):
        base_path = Path(base_path) if base_path else Path(f"{getcwd()}/{pardir}")
        self.config_path:Path = base_path / Path(config_path)
    
    def load_config(self) -> dict:
        with open(self.config_path, 'rb') as f:
            return tomllib.load(f)

class PathVariables:
    _instancia = None
    def __new__(cls):
        if cls._instancia is None:
            cls._instancia = super(PathVariables, cls).__new__(cls)
        return cls._instancia
        
    def __init__(self, base_path: str|Path = None):
        self.base_path = Path(base_path) if base_path else Path(f"{getcwd()}/{pardir}")
        self.model_path = self.base_path / "outputs" / "chekpoints"
        self.repo_path = self.base_path / "outputs" / "reports"
        self.logs_path = self.repo_path / "outputs" / "logs"
        self.study_path = self.repo_path / "outputs" / "studies"
        self.data_path = self.base_path / "data" / "dataset2"
        self.h5_file = self.data_path / "keypoints.h5"
        self.csv_file = self.data_path / "meta.csv"