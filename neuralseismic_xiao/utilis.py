import os
import glob
import numpy as np
import sys, shutil, random, bisect
import torch


def find_files(folder_path, file_pattern_key='*.npy'):
    file_pattern = os.path.join(folder_path, file_pattern_key)
    npy_files = glob.glob(file_pattern)

    return npy_files

def seed_torch(seed=0):
    random.seed(seed)
    np.random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    # torch.backends.cudnn.benchmark = False
    
def calculate_relative_loss(err, target=None, reduction="sum"):
    batch_size = err.shape[0]
    if isinstance(err, torch.Tensor):
        err_norm = torch.norm(err.reshape(batch_size, -1), p=2, dim=1)
        if target is None:
            target_norm = 1.0
        else:
            target_norm = torch.norm(target.reshape(batch_size, -1), p=2, dim=1)
        if reduction is None:
            return err_norm / target_norm
        elif reduction == "sum":
            return torch.sum(err_norm / target_norm)
        else:
            return torch.mean(err_norm / target_norm)
    else:
        err_norm = np.linalg.norm(err.reshape(batch_size, -1), ord=2, axis=1)
        if target is None:
            target_norm = 1.0
        else:
            target_norm = np.linalg.norm(target.reshape(batch_size, -1), ord=2, axis=1)
        if reduction is None:
            return err_norm / target_norm
        elif reduction == "sum":
            return np.sum(err_norm / target_norm)
        else:
            return np.mean(err_norm / target_norm)
        
        
def count_params(model):
    """Returns the total number of parameters of a PyTorch model
    
    Notes
    -----
    One complex number is counted as two parameters (we count real and imaginary parts)'
    """
    return sum(
        [p.numel() * 2 if p.is_complex() else p.numel() for p in model.parameters()]
    )