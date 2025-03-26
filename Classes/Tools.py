import torch
from unsloth import FastLanguageModel
import gc

class Tools:
    def __init__(self):
        super().__init__()
        self.executeLLM()

    def getLLM(self):
        return self.embedding_layer, self.tokenizer

    def executeLLM(self):
        max_seq_length = 2048 * 2

        dtype = None
        load_in_4bit = True

        model, self.tokenizer = FastLanguageModel.from_pretrained(
            model_name="unsloth/Llama-3.2-3B-Instruct",
            max_seq_length=max_seq_length,
            dtype=dtype,
            load_in_4bit=load_in_4bit,
        )

        # Obtener capa de embeddings
        self.embedding_layer = model.get_input_embeddings()

        del model

        while True:
            torch.cuda.empty_cache()
            if gc.collect() == 0:
                break