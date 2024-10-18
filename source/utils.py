import pickle
import random
import numpy as np
import torch

# Data save
def save_data(data, path):
    with open(path, "wb") as fp:   #Pickling
        pickle.dump(data, fp)

# Data load
def load_data(path):
    with open(path, "rb") as fp:   # Unpickling
        b = pickle.load(fp)
        return b
    
    
def remove_surplus(sentences, pad_idx=2):
    end_indices = []
    for sentence in sentences:
        indices = (sentence == pad_idx).nonzero(as_tuple=True)[0]
        if len(indices) > 0:
            end_indices.append(indices[0].item())
        else:
            end_indices.append(-1)

    max_pos_idx = max(end_indices)

    return sentences[:, :max_pos_idx+1]

def set_seed(seed=7, cudnn=False):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    
    if cudnn:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False