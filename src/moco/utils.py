import random
import numpy as np
import torch

def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def save_state(path, model, optimizer=None, epoch=None):
    payload = {'model_state': model.state_dict()}
    if optimizer is not None:
        payload['opt_state'] = optimizer.state_dict()
    if epoch is not None:
        payload['epoch'] = epoch
    torch.save(payload, path)


def print_and_log(message, log_file=None):
    # Print to console with datetime
    printedatetime = __import__('datetime').datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(f'[{printedatetime}] {message}')

    # Log to file if log_file is provided
    if log_file:
        with open(log_file, 'a') as f:
            f.write(f'[{printedatetime}] {message}\n')
    return