import os
import yaml
import logging
import tempfile
import json
import torch

from typing import List, Dict
from PIL import Image
from tqdm import tqdm
from model_handler import ModelHandler

def load_config(path: str) -> dict:
    with open(path, 'r') as f:
        return yaml.safe_load(f)

def setup_logging(debug: bool):
    level = logging.DEBUG if debug else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(asctime)s %(levelname)s %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S"
    )

def read_splits(splits_file: str) -> List[str]:
    with open(splits_file) as f:
        return [line.strip() for line in f if line.strip()]

def load_object_dict(path: str) -> dict:
    """
    Loads the obj_dict created by maskclustering and returns a python dict.

    The object dictionary (obj_dict) is structured as follows:
    obj_dict: Dict[int, Dict], maps object ID (int) to object data (Dict).

    Each object data dictionary contains:
      'point_ids': np.ndarray, point cloud IDs for the object.
      'mask_list': List[Tuple[int, int, float]], a list of tuples,
                   where each tuple is (frame_id, mask_id, coverage_score)
                   representing a 2D mask of the object.
      'repre_mask_list': List[Tuple[int, int, float]], the top 5 representative
                         masks from 'mask_list', sorted by coverage_score.
      'best_view': Dict, details of the best 2D view of the object, containing:
        'frame_id': int, frame ID of the best view.
        'mask_id': int, mask ID within the frame for the best view.
        'coverage': float, coverage score of this best view mask.
        'original_path': str, relative path to the original best view image.
        'highlighted_path': str, relative path to the highlighted original best view image.
        'cropped_path': str, relative path to the cropped best view image.
        'cropped_highlighted_path': str, relative path to the highlighted cropped best view image.
        'bbox': List, bounding box coordinates.
        'cropped_caption': str, (added by VLM captioning script) caption for the cropped image.
        'original_caption': str, (added by VLM captioning script) caption for the original image.
        Other fields related to the best view selection from the maskclustering process
        might also be present.

    Args:
        path: The file path to the .pth file containing the object dictionary.

    Returns:
        A dictionary containing the loaded object data.
    """
    return torch.load(path, weights_only=False)
 
def save_object_dict(obj_dict: dict, path: str) -> bool:
    tmp = tempfile.NamedTemporaryFile(dir=os.path.dirname(path), delete=False)
    try:
        torch.save(obj_dict, tmp.name)
        tmp.close()
        os.replace(tmp.name, path)
        return True
    finally:
        if os.path.exists(tmp.name):
            os.unlink(tmp.name)

def save_captions(captions: Dict[int, str], out_dir: str, seq: str):
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, f"{seq}.captions.json")
    with open(out_path, 'w') as f:
        json.dump(captions, f, indent=2)

def get_image_paths_for_captioning(obj_dict: Dict, root: str, seq: str) -> List[tuple[int, Dict[str, str]]]:
    """
    Collects absolute image paths for objects that have both cropped and original best views.

    Args:
        obj_dict: The object dictionary loaded from the .pth file.
        root: The root directory of the dataset.
        seq: The sequence name (scene identifier).

    Returns:
        A list of tuples, where each tuple contains an object_id and a dictionary
        mapping image_type ('cropped', 'original') to its absolute path.
        Returns an empty list if no objects qualify.
    """
    to_caption = []
    # Iterate over all objects and extract the paths of the best view images
    for object_id, data in obj_dict.items():
        best_view = data.get("best_view", {})
        if not best_view:
            continue
            
        paths = {}
        for img_type in ["cropped", "original"]:
            rel_path = best_view.get(f"{img_type}_path")            
            abs_path = os.path.join(root, "scannetpp", "data", seq, rel_path)
            if os.path.exists(abs_path):
                paths[img_type] = abs_path
            else:
                logging.warning(f"Image not found for object {object_id} ({img_type}): {abs_path}")
                break # Break from inner loop, this object won't have both paths
        
        if len(paths) == 2: # Ensure both 'cropped' and 'original' paths were found and exist
            to_caption.append((object_id, paths))
    return to_caption

def generate_captions_for_object(
    handler: ModelHandler, 
    object_id: int, 
    img_paths: Dict[str, str], 
    obj_dict: Dict,
    progress_bar: tqdm
) -> Dict[str, str]:
    """
    Generates captions for the cropped and original images of a single object.

    Returns:
        A dictionary containing the generated captions for the object,
        mapping image type to caption text.
    """
    obj_captions: Dict[str, str] = {"cropped": "", "original": ""}
    for img_type, img_path in img_paths.items():
        try:
            # Create caption for image anbd save it
            img = Image.open(img_path).convert("RGB")
            text = handler.caption_image(img)
            obj_dict[object_id]["best_view"][f"{img_type}_caption"] = text
            obj_captions[img_type] = text
        except Exception as e:
            logging.warning(f"Failed to process {img_type} image {img_path} for object {object_id}: {e}")
        finally:
            progress_bar.update(1)
    return obj_captions

def create_vlm_captions(handler: ModelHandler, root: str, seq: str, out_dir: str) -> bool:
    logging.info(f"====> Processing scene {seq}")

    # Load object dict
    dict_path = os.path.join(root, "scannetpp/data", seq, "output/best_views/best_view_object_dict.pth")
    obj_dict = load_object_dict(dict_path)
    logging.info(f"Loaded {len(obj_dict)} objects in dict")

    # Get image paths
    to_caption = get_image_paths_for_captioning(obj_dict, root, seq)
    if len(to_caption) == 0:
        logging.info("Nothing to caption")
        return True

    
    captions: Dict[int, Dict[str, str]] = {}
    bar = tqdm(total=len(to_caption) * 2, desc=f"Captioning {seq}", unit="img")
    # Create captions for all relevant objects
    for object_id, img_paths in to_caption:
        generated_obj_captions = generate_captions_for_object(
            handler, object_id, img_paths, obj_dict, bar
        )
        captions[object_id] = generated_obj_captions

    bar.close()

    if not save_object_dict(obj_dict, dict_path):
        logging.error("Failed to save updated object dict")
        return False

    save_captions(captions, out_dir, seq)
    logging.info(f"Saved {len(captions)} object captions => {out_dir}/{seq}.captions.json")
    return True

def run_vlm_captioning(config_file: str = "vlm_caption/caption.yaml"):
    """
    Run Vision Language Model (VLM) captioning on a set of scenes.
    This function loads configuration, sets up the model, and processes scenes
    to generate captions using a VLM.
    Args:
        config_file (str, optional): Path to the YAML configuration file.
            Defaults to "vlm_caption/caption.yaml".
    Returns:
        None
    """

    # Load config file and configs
    cfg = load_config(config_file)
    dataset_cfg = cfg["dataset"]
    model_cfg = cfg["model"]
    inference_cfg = cfg["inference"]

    setup_logging(inference_cfg.get("debug", False))

    if inference_cfg.get("seq_names"):
        scenes = inference_cfg["seq_names"]
    else:
        scenes = read_splits(dataset_cfg["splits_file"])

    handler = ModelHandler(
        model_name=model_cfg["name"],
        backend=model_cfg.get("backend", "transformers"),
        quantize=model_cfg.get("quantize", False)
    )

    success = 0 
    for seq in scenes:
        ok = create_vlm_captions(
            handler,
            root=dataset_cfg["root"],
            seq=seq,
            out_dir=inference_cfg.get("output_dir", "outputs")
        )
        if ok:
            success += 1

    logging.info(f"Done: {success} succeeded, {len(scenes) - success} failed")

if __name__ == "__main__":
    run_vlm_captioning()
