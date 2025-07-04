import os
import cv2
import numpy as np


def get_best_views(object_dict_dir: str, obj_id: int, dataset_root: str):
    """Return a *list* of (rgb_image, metadata) tuples for the representative
    views of ``obj_id``. Falls back to empty list if unavailable."""

    obj_dict_path = os.path.join(object_dict_dir, "object_dict.npy")
    if not os.path.exists(obj_dict_path):
        return []

    obj_dict = np.load(obj_dict_path, allow_pickle=True).item()
    entry = obj_dict.get(obj_id, {})

    views = entry.get("views") or ([] if "best_view" not in entry else [entry["best_view"]])

    out = []
    for view in views:
        rel_img_path = view.get("original_path")
        if not rel_img_path:
            continue
        abs_path = os.path.join(os.path.dirname(dataset_root), rel_img_path)
        if not os.path.exists(abs_path):
            continue
        rgb = cv2.imread(abs_path)
        out.append((rgb, view))

    return out


def get_best_view(object_dict_dir, obj_id, dataset_root):
    """Backward-compat wrapper, returns the *first* best view only."""

    views = get_best_views(object_dict_dir, obj_id, dataset_root)
    if not views:
        return None, None
    return views[0]