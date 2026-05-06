# pyright: reportMissingImports=false
import torch

# Cuda Utils
def cuda_synchronize():
    if torch.cuda.is_available():
        torch.cuda.synchronize()


def cuda_empty_cache():
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    
