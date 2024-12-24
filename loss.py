import torch
import torch.nn.functional as F
from util import get_text_embedding

def adversarial_loss(real, fake):
    return torch.mean((real - fake) ** 2)

def cycle_consistency_loss(real, cycle, embedding_layer):
    real_emb = get_text_embedding(real, embedding_layer)  # [batch_size, seq_len, d_model]
    cycle_emb = get_text_embedding(cycle, embedding_layer)  # [batch_size, seq_len, d_model]
    cos_sim = F.cosine_similarity(real_emb, cycle_emb)
    loss = 1 - cos_sim.mean()
    return loss
