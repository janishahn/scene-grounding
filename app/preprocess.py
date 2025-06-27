# %%
import numpy as np
import torch 

def load_object_dict(path):
    return np.load(path, allow_pickle=True)
# %%

object_dict = load_object_dict("/home/vlm_search/scene-grounding/maskclustering/data/scannetpp/data/95d525fbfd/output/object/scannetpp/object_dict.npy")
print(object_dict)

    
# %%
