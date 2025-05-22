import torch
import random
import numpy as np

def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(seed)

def save_checkpoint(state, filename='checkpoint.pth'):
    torch.save(state, filename)

def load_checkpoint(model, optimizer, filename, device):
    ckpt = torch.load(filename, map_location=device)
    model.load_state_dict(ckpt['state_dict'])
    optimizer.load_state_dict(ckpt['optimizer'])

def accuracy(preds, labels):
    return (preds == labels).float().mean().item()

class AverageMeter:
    def __init__(self): self.sum = 0; self.count = 0
    def update(self, val, n=1): self.sum += val * n; self.count += n
    @property
    def avg(self): return self.sum / self.count if self.count else 0