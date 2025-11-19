import os
import random
import torch
import numpy as np

# Setting the seed for reproducibility across random, numpy, and torch
def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


# Selecting the appropriate device (GPU if available, otherwise CPU)
def get_device():
    return "cuda" if torch.cuda.is_available() else "cpu"


# Ensuring a directory exists before saving any files
def ensure_dir(path):
    if not os.path.exists(path):
        os.makedirs(path)


# Saving the model checkpoint
def save_checkpoint(model, path):
    ensure_dir(os.path.dirname(path))
    torch.save(model.state_dict(), path)

# Loading a model checkpoint
def load_checkpoint(model, path, device):
    state = torch.load(path, map_location=device)
    model.load_state_dict(state)
    return model
