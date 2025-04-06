import torch
import torch.nn.functional as F

class MultimodalSignLM:
    def __init__(self, model, tokenizer, device):
        """
        Initialize the MultimodalSignLM class.
        Params
        :model: LLama 3 model.
        :tokenizer: The tokenizer for the model.
        :device: The device to run the model on (e.g., 'cuda' or 'cpu').
        """

        self.model = model.to(device)
        self.tokenizer = tokenizer
        self.device = device

        # Get the embeddings of all tokens in the vocabulary
        self.all_embeddings = model.get_input_embeddings().weight.data.to(self.device)

    def process_inputs(self, keypoints_embeddings, text_input:str):
        # Preprocess the inputs
        keypoints_embeddings = keypoints_embeddings.to(self.device)
        inputs = self.tokenizer(text_input, return_tensors="pt").to(self.device)

        sign_embed_tokens = torch.tensor(
            [self._find_closest_token(emb, self.all_embeddings) for emb in keypoints_embeddings[0]]  # embeddings[0] porque es un batch de tamaño 1
        ).unsqueeze(0).to(self.device)

        # Concatenate the token embeddings with the text input
        inputs['input_ids'] = torch.cat([inputs['input_ids'], sign_embed_tokens], dim=1)
        inputs['attention_mask'] = torch.cat([inputs['attention_mask'], torch.ones_like(sign_embed_tokens)], dim=1)
        
        return inputs
    
    def generate(self, keypoints_embeddings, text_input:str):
        self.model.eval()

        # Process the inputs
        inputs = self.process_inputs(keypoints_embeddings, text_input)

        # Generate the output
        with torch.no_grad():
            output = self.model.generate(
                input_ids=inputs['input_ids'],
                attention_mask=inputs['attention_mask'],
            )

        # Decode the output
        decoded_output = self.tokenizer.decode(output[0], skip_special_tokens=True)
        return decoded_output
    
    def _find_closest_token(self, embedding, all_embeddings):
        embedding = embedding.to(self.device)
        if embedding.dim() > 1:
            embedding = embedding.squeeze()

        # Calcular similitud del coseno
        similarities = F.cosine_similarity(embedding.unsqueeze(0), all_embeddings, dim=1)

        # Encontrar el índice del token más similar
        closest_token_id = torch.argmax(similarities).item()
        return closest_token_id