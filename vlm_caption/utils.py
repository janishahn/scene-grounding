import os
import json
import torch
import tempfile
from typing import Dict, Optional, Tuple
from PIL import Image
import logging

def setup_logging(level: int = logging.INFO) -> None:
    """
    Set up logging configuration.
    
    Parameters
    ----------
    level : int, optional
        Logging level. Default is logging.INFO.
    """
    logging.basicConfig(
        level=level,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

def ensure_dir_exists(path: str) -> None:
    """
    Create directory if it doesn't exist.
    
    Parameters
    ----------
    path : str
        Directory path to create.
    """
    os.makedirs(path, exist_ok=True)

def get_image_path(dataset_root: str, dataset_type: str, seq_name: str, obj_id: int) -> str:
    """
    Get the path to the best view image for an object.
    
    Parameters
    ----------
    dataset_root : str
        Root directory of the dataset.
    dataset_type : str
        Type of dataset (e.g., 'scannet', 'matterport3d', 'scannetpp').
    seq_name : str
        Scene/sequence name.
    obj_id : int
        Object ID.
    
    Returns
    -------
    str
        Path to the best view image.
    """
    if dataset_type == "scannet":
        return os.path.join(dataset_root, "scannet", "processed", seq_name, "output", "object", f"best_view_{obj_id}.jpg")
    elif dataset_type == "matterport3d":
        return os.path.join(dataset_root, "matterport3d", "scans", seq_name, seq_name, "output", "object", f"best_view_{obj_id}.jpg")
    elif dataset_type == "scannetpp":
        return os.path.join(dataset_root, "scannetpp", "data", seq_name, "output", "object", f"best_view_{obj_id}.jpg")
    else:
        return os.path.join(dataset_root, "demo", seq_name, "output", "object", f"best_view_{obj_id}.jpg")

def resolve_best_view_path(dataset_root: str, dataset_type: str, seq_name: str, image_path: str) -> str:
    """
    Resolve absolute path for best view image from relative path.
    
    Parameters
    ----------
    dataset_root : str
        Root directory of the dataset.
    dataset_type : str
        Type of dataset.
    seq_name : str 
        Scene/sequence name.
    image_path : str
        Relative path from object dict.
    
    Returns
    -------
    str
        Absolute path to the image.
    """
    if dataset_type == "scannet":
        scene_root = os.path.join(dataset_root, "scannet", "processed", seq_name)
    elif dataset_type == "matterport3d":
        scene_root = os.path.join(dataset_root, "matterport3d", "scans", seq_name, seq_name)
    elif dataset_type == "scannetpp":
        scene_root = os.path.join(dataset_root, "scannetpp", "data", seq_name)
    else:
        scene_root = os.path.join(dataset_root, "demo", seq_name)
    
    return os.path.join(scene_root, image_path)

def load_object_dict(dataset_root: str, dataset_type: str, seq_name: str) -> Dict:
    """
    Load the object dictionary.
    
    Parameters
    ----------
    dataset_root : str
        Root directory of the dataset.
    dataset_type : str
        Type of dataset (e.g., 'scannet', 'matterport3d', 'scannetpp').
    seq_name : str
        Scene/sequence name.
    
    Returns
    -------
    Dict
        Object dictionary containing information about all objects.
    """
    if dataset_type == "scannet":
        object_dict_path = os.path.join(dataset_root, "scannet", "processed", seq_name, "output", "best_views", "best_view_object_dict.pth")
    elif dataset_type == "matterport3d":
        object_dict_path = os.path.join(dataset_root, "matterport3d", "scans", seq_name, seq_name, "output", "best_views", "best_view_object_dict.pth")
    elif dataset_type == "scannetpp":
        object_dict_path = os.path.join(dataset_root, "scannetpp", "data", seq_name, "output", "best_views", "best_view_object_dict.pth")
    else:
        object_dict_path = os.path.join(dataset_root, "demo", seq_name, "output", "best_views", "best_view_object_dict.pth")
    
    # Explicitly set weights_only=False since we're loading a dictionary
    try:
        return torch.load(object_dict_path, weights_only=False)
    except Exception as e:
        logging.warning(f"Failed to load with default settings, trying alternative approach: {e}")
        # Fall back to map_location if needed
        return torch.load(object_dict_path, map_location="cpu", weights_only=False)

def load_object_dict_safe(path: str) -> Dict:
    """
    Safely load object dictionary from .pth file.
    
    Parameters
    ----------
    path : str
        Path to object_dict.pth file.
    
    Returns
    -------
    Dict
        Loaded object dictionary.
    """
    try:
        # Explicitly set weights_only=False since we're loading a dictionary
        return torch.load(path, weights_only=False)
    except Exception as e:
        logging.error(f"Failed to load object dictionary from {path}: {str(e)}")
        return {}

def save_captions_to_json(captions: Dict[int, str], output_path: str) -> None:
    """
    Save captions to a JSON file.
    
    Parameters
    ----------
    captions : Dict[int, str]
        Dictionary mapping object IDs to captions.
    output_path : str
        Path to save the JSON file.
    """
    with open(output_path, 'w') as f:
        json.dump(captions, f, indent=2)

def save_object_dict_safe(obj_dict: Dict, save_path: str) -> bool:
    """
    Safely save object dictionary using atomic file operation.
    
    Parameters
    ----------
    obj_dict : Dict
        Object dictionary to save.
    save_path : str
        Target path for saving.
        
    Returns
    -------
    bool
        True if save successful, False otherwise.
    """
    tmp = None
    try:
        # Create temp file in same directory for atomic move
        dirname = os.path.dirname(save_path)
        ensure_dir_exists(dirname)
        
        # Use tempfile in same directory for atomic move
        tmp = tempfile.NamedTemporaryFile(dir=dirname, delete=False)
        torch.save(obj_dict, tmp.name)
        tmp.close()
        
        # Atomic replace
        os.replace(tmp.name, save_path)
        return True
        
    except Exception as e:
        logging.error(f"Failed to save object dictionary to {save_path}: {str(e)}")
        if tmp and os.path.exists(tmp.name):
            os.unlink(tmp.name)
        return False

def export_summary(object_dict: Dict, output_dir: str, format: str = "json") -> None:
    """
    Export object dictionary summary to JSON or CSV.
    
    Parameters
    ----------
    object_dict : Dict
        Object dictionary to export.
    output_dir : str
        Directory to save summary file.
    format : str, optional
        Output format - "json" or "csv". Default is "json".
    """
    ensure_dir_exists(output_dir)
    
    summary = {
        obj_id: {
            "best_view": obj_data.get("best_view", {}),
            "num_points": len(obj_data.get("point_ids", [])),
            "num_masks": len(obj_data.get("repre_mask_list", []))
        }
        for obj_id, obj_data in object_dict.items()
    }
    
    if format == "json":
        output_path = os.path.join(output_dir, "object_summary.json")
        with open(output_path, 'w') as f:
            json.dump(summary, f, indent=2)
    else:
        import csv
        output_path = os.path.join(output_dir, "object_summary.csv")
        with open(output_path, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(["object_id", "num_points", "num_masks", "best_view_frame", "best_view_mask"])
            for obj_id, data in summary.items():
                best_view = data["best_view"]
                writer.writerow([
                    obj_id,
                    data["num_points"],
                    data["num_masks"],
                    best_view.get("frame_id", ""),
                    best_view.get("mask_id", "")
                ])
    
    logging.info(f"Exported summary to {output_path}")

def load_image(image_path: str) -> Optional[Image.Image]:
    """
    Load an image from the given path.
    
    Parameters
    ----------
    image_path : str
        Path to the image file.
    
    Returns
    -------
    Optional[Image.Image]
        The loaded image, or None if the file doesn't exist.
    """
    if not os.path.exists(image_path):
        logging.warning(f"Image not found: {image_path}")
        return None
    
    try:
        return Image.open(image_path).convert("RGB")
    except Exception as e:
        logging.error(f"Error loading image {image_path}: {str(e)}")
        return None
