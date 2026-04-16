import torch

from torch import nn
from datetime import datetime
from os.path import join, exists
from os import makedirs, listdir

def yymmdd_hhmm(date: datetime) -> str:
    return date.strftime("%y%m%d_%H%M%S")

MODEL_WEIGHT_DIRECTORY = './models'

def save_weights(model: nn.Module, path: str=None):
    if path is None:
        fname = model._get_name() + '_' + yymmdd_hhmm(datetime.now())
        makedirs(MODEL_WEIGHT_DIRECTORY, exist_ok=True) # os create dir if not exist
        path = join(MODEL_WEIGHT_DIRECTORY, fname)
    torch.save(model.state_dict(), path)
    print("Saved weights to " + path)
    return True

def get_latest(m:nn.Module) -> str | None:
    mname = m._get_name()
    all_weights = listdir(MODEL_WEIGHT_DIRECTORY)
    compatible_weights = [fname for fname in all_weights if fname.startswith(mname)]
    if len(compatible_weights) == 0:

        return None
    latest = sorted(compatible_weights, reverse=True)[0]
    # return latest
    return join(MODEL_WEIGHT_DIRECTORY, latest)

def load_weights(model: nn.Module, path: str=None):
    if path and not exists(path):
        print("No such file: " + path)
        return False

    if path is None:
        print("Loading latest...")
        path = get_latest(model)
    if path is None:
        print("No fitting models found. Did not load.")
        return False
    _ = model.load_state_dict(torch.load(path, weights_only=True))
    return True

