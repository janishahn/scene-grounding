import numpy as np

file_path = '/home/vlm_search/scene-grounding/maskclustering/data/prediction/scannetpp_class_agnostic/95d525fbfd.npz'
data = np.load(file_path)

print(list(data.keys()))