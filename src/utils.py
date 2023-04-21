import random
import os
import torch
from fastai.vision.all import OptimWrapper
import numpy as np

def seed_everything(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True
    
def WrapperAdamW(param_groups,**kwargs):
    return OptimWrapper(param_groups,torch.optim.AdamW)