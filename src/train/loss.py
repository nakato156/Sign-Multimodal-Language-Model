import torch.nn as nn
import torch.nn.functional as F

class ImitatorLoss(nn.Module):

    def __init__(self, alpha=1.0, beta=1.0):
        """
        Combina L2 (MSE) y CosineEmbeddingLoss:
        total_loss = alpha * L2 + beta * (1 - cos_similarity)
        """
        super().__init__()

        self.alpha = alpha
        self.beta = beta
        self.cos_sim = nn.CosineSimilarity(dim=-1)

    def forward(self, pred, target):
        # l2 = F.mse_loss(pred, target)

        cosine_sim = self.cos_sim(F.normalize(pred, p=2, dim=-1), F.normalize(target, p=2, dim=-1))
        cosine_loss = 1 - cosine_sim.mean()

        # return (self.alpha * l2) + (self.beta * cosine_loss)
        return cosine_loss