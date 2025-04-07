import os
import torch

from torch.utils.data import DataLoader, random_split

#Imported Classes
from Classes.SignDataLoader import SignDataLoader
from Classes.Imitator import Imitator
from Classes.KeypointDataset import KeypointDataset
from Classes.Tools import Tools

#Profilers
from torch.profiler import profile, ProfilerActivity

# sudo env "PATH=$PATH" nsys profile --trace cuda,osrt,nvtx --gpu-metrics-device=all --cuda-memory-usage true --force-overwrite true --output profile_run_v1 --gpu-metrics-frequency=500 python train_CUDA.py

PROFILE = False
LOG = False

def trace_handler(p):
    output = p.key_averages().table(sort_by=sort_by_keyword, row_limit=10)
    print(output)
    p.export_chrome_trace("/tmp/trace_" + str(p.step_num) + ".json")

if __name__ == "__main__":
    tools = Tools(LOG)

    embedding_layer, tokenizer = tools.getLLM()
    vocab_size, d_model = embedding_layer.weight.size()

    print(f"Vocab size: {vocab_size}, d_model: {d_model}")

    DataPath = os.path.join(os.getcwd(), os.pardir, "data", "dataset2")
    ModelPath = os.path.join(os.getcwd(), "model")
    h5File = os.path.join(DataPath, "keypoints.h5")
    csvFile = os.path.join(DataPath, "meta.csv")

    # parameters
    modelParameters = {
        "model": {
            "version": 5,
            "checkpoint": 1
        },
        "input_size": 543*2,
        "output_size": 3072,
        "learning_rate": 2e-4,
        "device": "cuda" if torch.cuda.is_available() else "cpu",
        "epochs": 1000,
        "logIntervals": 20,
        "checkpointIntervals": 40,
        "batchSize": 32,
        "frameClips": 15 * 35,
        "train_ratio": 0.8,
        "validation_ratio": 0.2
    }

    keypointReader = KeypointDataset(h5Path=h5File, labelsCSV=csvFile, max_seq_len=modelParameters["frameClips"])
    dataset = SignDataLoader(tokenizer, embedding_layer, keypointReader, modelParameters["device"])

    keypointReaderSize = len(keypointReader)
    train_size = int(keypointReaderSize * modelParameters["train_ratio"])
    validation_size = keypointReaderSize - train_size

    train_dataset, validation_dataset = random_split(dataset, [train_size, validation_size])
    train_dataloader = DataLoader(train_dataset, batch_size=modelParameters["batchSize"], shuffle=True, collate_fn=tools.collate_fn)
    val_dataloader = DataLoader(validation_dataset, batch_size=modelParameters["batchSize"], shuffle=True, collate_fn=tools.collate_fn)

    validation_dataloader = DataLoader(
        validation_dataset,
        batch_size=modelParameters["batchSize"],
        shuffle=True,
        collate_fn=tools.collate_fn,
    )

    # model
    model = Imitator(input_size=modelParameters["input_size"], T_size=modelParameters["frameClips"], output_size=modelParameters["output_size"]).to(modelParameters["device"])

    print(model)
    
    sort_by_keyword = 'cuda_time_total'
 
    if PROFILE:
        with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA], record_shapes=True, with_stack=True, profile_memory=True) as p:
            tools.train(model, train_dataloader, val_dataloader,epochs=modelParameters["epochs"], log_interval=modelParameters["logIntervals"], learning_rate=modelParameters["learning_rate"], modelVersions=modelParameters["model"], modelDir=ModelPath, checkpoint_interval=modelParameters["checkpointIntervals"])
    else:
        tools.train(
            model,
            train_dataloader,
            val_dataloader,
            epochs=modelParameters["epochs"],
            log_interval=modelParameters["logIntervals"],
            learning_rate=modelParameters["learning_rate"],
            modelVersions=modelParameters["model"],
            modelDir=ModelPath,
            checkpoint_interval=modelParameters["checkpointIntervals"],
        )

    #p.export_chrome_trace("profile_trace.json")
    #print(p.key_averages().table(sort_by="cuda_time_total", row_limit=10))    

    