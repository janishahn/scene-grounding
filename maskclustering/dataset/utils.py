import os
import cv2
import numpy as np


def get_best_view(object_dict_dir, obj_id, dataset_root):
    """Get best view image and metadata for an object.
    
    Args:
        object_dict_dir: Directory containing object dictionary
        obj_id: Object ID in the dataset
        dataset_root: Root directory of the dataset

    Returns:
        rgb_image: The best view RGB image
        metadata: Dict with frame_id, mask_id, coverage, and bbox
    """
    obj_dict = np.load(os.path.join(object_dict_dir, 'object_dict.npy'), allow_pickle=True).item()
    if obj_id not in obj_dict or 'best_view' not in obj_dict[obj_id]:
        return None, None
        
    view_info = obj_dict[obj_id]['best_view']
    rgb_image = cv2.imread(os.path.join(os.path.dirname(dataset_root), view_info['image_path']))
    
    return rgb_image, view_info