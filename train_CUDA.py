import os
import torch
import torch.multiprocessing as mp

from torch.utils.data import DataLoader, random_split

#Imported Classes
from Classes.train.SignDataLoader import SignDataLoader
from Classes.train.Imitator import Imitator
from Classes.train.KeypointDataset import KeypointDataset
from Classes.train.Tools import Tools

#Profilers
from torch.profiler import profile, ProfilerActivity

PROFILE = False
LOG = False

def trace_handler(p):
    output = p.key_averages().table(sort_by=sort_by_keyword, row_limit=10)
    print(output)
    p.export_chrome_trace("/tmp/trace_" + str(p.step_num) + ".json")

if __name__ == "__main__":
    #mp.set_start_method("spawn", force=True)
    tools = Tools(LOG)

    embedding_layer, tokenizer = tools.getLLM()
    vocab_size, d_model = embedding_layer.weight.size()

    print(f"Vocab size: {vocab_size}, d_model: {d_model}")

    DataPath = os.path.join(os.getcwd(), os.pardir, "data", "dataset2")
    ModelPath = os.path.join(os.getcwd(), "model")
    h5File = os.path.join(DataPath, "keypoints.h5")
    csvFile = os.path.join(DataPath, "meta.csv")

    # Parameters and Saving Parameteres
    modelParameters = {
        "model": {
            "version": 11,
            "checkpoint": 5,
            "from_checkpoint": False
        },
        "input_size": 543*2,
        "output_size": 3072,
        "learning_rate": 5e-4,
        "device": "cuda" if torch.cuda.is_available() else "cpu",
        "epochs": 10,
        "logIntervals": 20,
        "checkpointIntervals": 5,
        "batchSize": 32,
        "frameClips": 15 * 35,
        "train_ratio": 0.8,
        "validation_ratio": 0.2
    }

    tools.saveParameters(ModelPath, modelParameters)

    keypointReader = KeypointDataset(h5Path=h5File, labelsCSV=csvFile, max_seq_len=modelParameters["frameClips"])
    dataset = SignDataLoader(tokenizer, embedding_layer, keypointReader, modelParameters["device"])

    keypointReaderSize = len(keypointReader)
    train_size = int(keypointReaderSize * modelParameters["train_ratio"])
    validation_size = keypointReaderSize - train_size

    train_dataset, validation_dataset = random_split(dataset, [train_size, validation_size])
    
    train_dataloader = DataLoader(train_dataset, batch_size=modelParameters["batchSize"], shuffle=True, collate_fn=tools.collate_fn)
    val_dataloader = DataLoader(validation_dataset, batch_size=modelParameters["batchSize"], shuffle=True, collate_fn=tools.collate_fn)

    # model
    model = Imitator(input_size=modelParameters["input_size"], T_size=modelParameters["frameClips"], output_size=modelParameters["output_size"]).to(modelParameters["device"])
    model_compiled = torch.compile(model, backend="inductor", mode="default")

    print(model)
    
    sort_by_keyword = 'cuda_time_total'
 
    if PROFILE:
        with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA], record_shapes=True, with_stack=True, profile_memory=True) as p:
            tools.train(
                model_compiled,
                train_dataloader,
                val_dataloader,
                epochs=modelParameters["epochs"],
                log_interval=modelParameters["logIntervals"],
                learning_rate=modelParameters["learning_rate"],
                modelVersions=modelParameters["model"],
                modelDir=ModelPath,
                checkpoint_interval=modelParameters["checkpointIntervals"],
            )
    else:
        tools.train(
            model_compiled,
            train_dataloader,
            val_dataloader,
            epochs=modelParameters["epochs"],
            log_interval=modelParameters["logIntervals"],
            learning_rate=modelParameters["learning_rate"],
            modelVersions=modelParameters["model"],
            modelDir=ModelPath,
            checkpoint_interval=modelParameters["checkpointIntervals"],
        )