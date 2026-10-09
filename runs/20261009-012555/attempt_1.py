import os
import time
import numpy as np
import scipy.io
import torch
import torch.nn as nn
import torch.optim as optim

mat_path = os.environ.get("SILVERBOX_MAT", "")
print("MAT path:", mat_path)
mat = scipy.io.loadmat(mat_path)
print("Keys in MAT:", [k for k in mat.keys() if not k.startswith('__')])
for k in mat.keys():
    if not k.startswith('__'):
        val = mat[k]
        if hasattr(val, 'shape'):
            print(k, type(val), val.shape)
        elif hasattr(val, 'dtype'):
            print(k, type(val), val.dtype)
