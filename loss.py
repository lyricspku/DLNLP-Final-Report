import torch

def adversarial_loss(real, fake):
    return torch.mean((real - fake) ** 2)

def cycle_consistency_loss(real, cycle):
    real = real.float()
    cycle = cycle.float()
    return torch.mean((real - cycle) ** 2)
