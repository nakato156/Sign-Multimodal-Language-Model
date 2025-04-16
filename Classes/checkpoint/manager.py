import os, json
import torch

class CheckpointManager:
    def __init__(self, base_dir, version, checkpoint):
        self.base = base_dir
        self.v = version
        self.ckpt = checkpoint

    def _path(self, extra=""):
        p = os.path.join(self.base, "checkpoints", str(self.v), str(self.ckpt), extra)
        os.makedirs(p, exist_ok=True)
        return p

    def save_model(self, model, epoch):
        path = self._path(str(epoch))
        torch.save(model.state_dict(), os.path.join(path, "model.pt"))

    def load_model(self, model, path):
        state = torch.load(path)
        model.load_state_dict(state)
        return model

    def save_params(self, params):
        p = self._path()
        with open(os.path.join(p, "parameters.json"), "w") as f:
            json.dump(params, f, indent=2)
