from tqdm import tqdm

import torch
from torch import autocast, GradScaler
from torch.optim import AdamW
from torch.utils.tensorboard import SummaryWriter

from mslm.utils.early_stopping import EarlyStopping
from mslm.training import ImitatorLoss
from mslm.checkpoint.manager import CheckpointManager

import nvtx

class Trainer:
    def __init__(self, model, train_loader, val_loader, embedding_layer, **kwargs):
        self.LOG = kwargs.get("log", False)
        
        self.device = kwargs.get("device", "cuda")
        self.epochs = kwargs.get("epochs", 100)
        self.learning_rate = kwargs.get("learning_rate", 1e-4)
        self.log_interval = kwargs.get("log_interval", 5)
        self.checkpoint_interval = kwargs.get("checkpoint_interval", 5)
        self.embed_layer = embedding_layer
        self.model = model.to(self.device)
        self.ckpt_mgr = CheckpointManager(
            kwargs.get("model_dir", "../outputs/checkpoints"),
            kwargs.get("model_version", 1),
            kwargs.get("checkpoint", 0),
        )
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.writer = SummaryWriter("../outputs/reports/")
        self.scaler = GradScaler(device=self.device)
        self.criterion = ImitatorLoss(alpha=1.0, beta=1.0)
        self.early_stopping = EarlyStopping(patience=5)

    @torch.compile
    @nvtx.annotate("Start Training", color="green")
    def train(self):
        """Entrena el modelo Imitator.
        returns:
            train_loss: float, loss de entrenamiento
            val_loss: float, loss de validación
        """
        optimizer = AdamW(self.model.parameters(), lr=self.learning_rate, weight_decay=1e-3)
        scheduler = torch.optim.lr_scheduler.OneCycleLR(
            optimizer,
            max_lr=1e-3,
            anneal_strategy="cos",
            epochs=self.epochs,
            steps_per_epoch=len(self.train_loader),
            pct_start=0.3,
        )

        train_loss = 0
        val_loss = 0

        for epoch in tqdm(range(self.epochs), desc="Entrenando", colour="green"):
            train_loss = self._train_epoch(epoch, optimizer, scheduler)
            val_loss = self._validate(epoch)
            if self.early_stopping.stop:
                break
        return train_loss, val_loss

    def distributed_train(self):
        pass

    def _train_epoch(self, epoch, optimizer, scheduler):
        self.model.train()
        total_loss = 0

        for data, input_ids in self.train_loader:
            optimizer.zero_grad(set_to_none=True)

            with nvtx.annotate("Data to CUDA", color="yellow"):
                data = data.to(self.device, non_blocking=True)
                print(data.shape)
                input_ids = input_ids.to(self.device, non_blocking=True)
                embeddings = self.embed_layer(input_ids)

            with nvtx.annotate("Training", color="blue"):
                #Change to bfloat16 if the GPU used is with Ampere Architecture or Higher
                if torch.cuda.get_device_capability()[0] >= 8:
                    with autocast(device_type=self.device, dtype=torch.bfloat16):
                        output = self.model(data)
                        loss = self.criterion(output, embeddings)
                else:
                    with autocast(device_type=self.device):
                        output = self.model(data)
                        loss = self.criterion(output, embeddings)
            
            with nvtx.annotate("Backward Pass", color="blue"):
                total_loss += loss.detach()
                final_loss = total_loss.item()

            self.writer.add_scalar("Loss/train", loss, epoch)

            with nvtx.annotate("Update", color="blue"):
                self.scaler.scale(loss).backward()
                
                self.scaler.step(optimizer)
                self.scaler.update()
                scheduler.step(epoch)

        torch.cuda.empty_cache()

        if epoch % self.log_interval == 0:
            print("\nEpoch: ", epoch, ".\t Total loss: ", final_loss/len(self.train_loader))

        if epoch == 1:
            self.ckpt_mgr.save_model(self.model, 1)
        elif (epoch % self.checkpoint_interval == 0 and epoch != 0) or (epoch == self.epochs - 1):
            self.ckpt_mgr.save_model(self.model, epoch)
        return final_loss
    
    def _validate(self, epoch):
        with nvtx.annotate("Prueba de Validacion", color="blue"):
            with torch.no_grad():
                self.model.eval()
                val_loss = 0
                
                for data, input_ids in self.val_loader:
                    data = data.to(self.device)
                    input_ids = input_ids.to(self.device)
                    embeddings = self.embed_layer(input_ids)

                    output = self.model(data)
                    loss = self.criterion(output.to(dtype=torch.bfloat16), embeddings)
                    val_loss += loss.detach()

                    # del output, data, embeddings, cos_sim
                torch.cuda.empty_cache()
                    
                final_val_loss = val_loss.item() / len(self.val_loader)
                if epoch % self.log_interval == 0:
                    print(f"Validation Loss: {final_val_loss}" )
                
                self.early_stopping(final_val_loss)
                if self.early_stopping.stop:
                    self.ckpt_mgr.save_model(self.model, epoch)
                return final_val_loss