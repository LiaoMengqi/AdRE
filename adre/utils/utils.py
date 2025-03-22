import os
import json
import torch
import subprocess
import re
import random
import numpy as np
from transformers import set_seed as transformers_seed
from typing import Optional


def set_seed(seed: Optional[int]):
    if seed is None:
        return
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    transformers_seed(seed)

    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
