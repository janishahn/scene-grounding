import os
import yaml
import logging
import tempfile
import json
import time
import torch
import numpy as np
import cv2

from typing import List, Dict
from PIL import Image
from tqdm import tqdm
from vlm_caption.vlm_handler import VLMHandler
from vlm_caption.embedding import build_faiss_index

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

def wait_for_gpu_memory(min_free_mb: int = 1024, timeout: int = 300, interval: int = 5) -> None:
    """Block until at least *min_free_mb* of GPU memory is free.

    Parameters
    ----------
    min_free_mb: int, optional
        Minimum free memory (in MiB) that must be available before returning.
    timeout: int, optional
        Maximum time (in seconds) to wait before giving up.
    interval: int, optional
        How often (in seconds) to poll GPU memory.
    """
    if not torch.cuda.is_available():
        return  # nothing to do on CPU-only systems

    device = torch.cuda.current_device()
    start_time = time.time()

    # Try to clear any cached allocations first
    torch.cuda.empty_cache()

    while True:
        free_bytes, total_bytes = torch.cuda.mem_get_info(device)
        free_mb = free_bytes // (1024 * 1024)

        if free_mb >= min_free_mb:
            logging.info(f"GPU memory check passed: {free_mb} MiB free (≥ {min_free_mb} MiB)")
            break

        elapsed = time.time() - start_time
        if elapsed > timeout:
            logging.warning(
                f"Timeout ({timeout}s) waiting for GPU memory; only {free_mb} MiB free. Proceeding anyway."
            )
            break

        logging.info(
            f"Waiting for GPU memory: {free_mb} MiB free (< {min_free_mb} MiB). Sleeping {interval}s ..."
        )
        time.sleep(interval)

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
        'cropped_highlighted_path': str, relative path to the highlighted cropped best view image.
        'bbox': List, bounding box coordinates.
        'highlighted_caption': str, (added by VLM captioning script) caption for the highlighted image.
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
    Collects absolute image paths for objects that have both (cropped) highlighted and original best views.

    Preference order:
        highlighted -> highlighted_path (full-size image with green contour)
        original    -> cropped_path > original_path

    We pass the full-size highlighted image to DAM so that the pixel mask aligns with the image dimensions.
    """
    to_caption = []
    for object_id, data in obj_dict.items():
        best_view = data.get("best_view", {})
        if not best_view:
            continue

        # Use the full-size image with green contour for DAM. Avoid cropped versions.
        preferred: Dict[str, List[str]] = {
            "highlighted": ["highlighted_path"],
            "original": ["cropped_path", "original_path"],
        }

        paths: Dict[str, str] = {}
        missing = False
        for img_type, candidates in preferred.items():
            found = False
            for key in candidates:
                rel_path = best_view.get(key)
                if rel_path:
                    abs_path = os.path.join(root, "scannetpp", "data", seq, rel_path)
                    if os.path.exists(abs_path):
                        paths[img_type] = abs_path
                        found = True
                        break
            if not found:
                logging.warning(
                    f"No valid image found for object {object_id} (type={img_type}). Tried: {candidates}"
                )
                missing = True
                break  # skip this object entirely

        if not missing and len(paths) == 2:
            to_caption.append((object_id, paths))

    return to_caption

def merge_captions(
    highlighted: str,
    original: str,
    model_name: str = "",
) -> str:
    """Return a single caption that emphasises *highlighted* details while using *original* only as context.

    Falls back to simple concatenation if the LLM call fails or the model name is empty.
    """

    if not highlighted and not original:
        return ""

    if not model_name:
        return (highlighted + " " + original).strip()

    try:
        from ollama import chat

        system_prompt = (
            "You are a helpful assistant. Merge two descriptions that refer to the *same* object into a single "
            "caption. Give higher weight to the first description, which focuses only on the object, and use "
            "information from the second description only to fill in useful context. Keep as much detail from both descriptions as possible,"
            "while avoiding repetition and emphasizing the object as it is described by the first description. "
            "**DO NOT OUTPUT ANYTHING OTHER THAN THE MERGED CAPTION.**"
        )

        user_prompt = (
            "Object-only description (primary):\n" + highlighted.strip() +
            "\n\nContext description (secondary):\n" + original.strip()
        )

        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ]

        resp = chat(model=model_name, messages=messages)
        candidate = resp.get("message", {}).get("content", "").strip()
        return candidate if candidate else (highlighted + " " + original).strip()

    except Exception as e:
        logging.warning(f"Failed to merge captions via LLM ({e}), falling back to concatenation.")
        return (highlighted + " " + original).strip()

def generate_captions_for_object(
    handler_highlighted: VLMHandler,
    handler_original: VLMHandler,
    object_id: int,
    img_paths: Dict[str, str],
    obj_dict: Dict,
    progress_bar: tqdm,
    root_dir: str,
    seq_name: str,
) -> Dict[str, str]:
    """
    Generates captions for the highlighted and original images of a single object.

    Returns:
        A dictionary containing the generated captions for the object,
        mapping image type to caption text.
    """
    obj_captions: Dict[str, str] = {"highlighted": "", "original": ""}
    for img_type, img_path in img_paths.items():
        try:
            # Create caption for image and save it
            img = Image.open(img_path).convert("RGB")

            prompt_file = "vlm_caption/object_captioning_prompt.md" if img_type == "highlighted" else "vlm_caption/general_captioning_prompt.md"
            with open(prompt_file, "r") as f:
                prompt = f.read().strip()

            # Select correct handler based on image type
            handler = handler_highlighted if img_type == "highlighted" else handler_original

            # Build mask for DAM if required and highlighted image
            mask_img = None
            if img_type == "highlighted" and handler.backend == "dam":
                # Retrieve frame_id and mask_id from object dict
                bv = obj_dict[object_id]["best_view"]
                frame_id = bv["frame_id"]
                mask_id = bv["mask_id"]

                seg_path = os.path.join(root_dir, "scannetpp/data", seq_name, "output/mask", f"frame_{frame_id:06d}.png")
                if not os.path.exists(seg_path):
                    raise FileNotFoundError(f"Segmentation file not found: {seg_path}")
                segmentation = cv2.imread(seg_path, cv2.IMREAD_UNCHANGED)

                binary_mask = (segmentation == mask_id).astype("uint8") * 255

                # Ensure the mask dimensions match the image that will be sent to DAM
                img_w, img_h = img.size  # PIL gives (W, H)
                mask_h, mask_w = binary_mask.shape  # NumPy gives (H, W)

                # 1) Crop using stored bounding box if we are captioning a cropped image
                if (mask_h, mask_w) != (img_h, img_w):
                    bv_meta = obj_dict[object_id]["best_view"]
                    if "bbox" in bv_meta:
                        left, top, right, bottom = bv_meta["bbox"]
                        binary_mask = binary_mask[top:bottom + 1, left:right + 1]
                        mask_h, mask_w = binary_mask.shape

                # 2) If dimensions still mismatch, fall back to a resize (keeps nearest-neighbour sampling)
                if (mask_h, mask_w) != (img_h, img_w):
                    binary_mask = cv2.resize(binary_mask, (img_w, img_h), interpolation=cv2.INTER_NEAREST)

                from PIL import Image as PILImage
                mask_img = PILImage.fromarray(binary_mask)

            text = handler.caption_image(img, prompt=prompt, mask=mask_img)
            obj_dict[object_id]["best_view"][f"{img_type}_caption"] = text
            obj_captions[img_type] = {"text": text, "img_path": img_path}
        except Exception as e:
            logging.warning(f"Failed to process {img_type} image {img_path} for object {object_id}: {e}")
        finally:
            progress_bar.update(1)

    # Merge the two captions into one cohesive description using an LLM.
    merged = merge_captions(
        obj_captions["highlighted"]["text"],
        obj_captions["original"]["text"],
        model_name=handler_original.model_name,
    )

    obj_captions["combined"] = {"text": merged}
    obj_dict[object_id]["best_view"]["combined_caption"] = merged

    # Account for the additional LLM call in the progress bar
    progress_bar.update(1)

    return obj_captions

def create_vlm_captions_sequential(root: str, seq: str, out_dir: str, 
                                  original_model_cfg: dict, highlighted_model_cfg: dict,
                                  merging_model_cfg: dict | None = None) -> bool:
    """
    Create VLM captions in a sequential manner, first processing all original images,
    then all highlighted images to avoid GPU memory conflicts.
    """
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
    
    # PHASE 1: Process all original images first
    logging.info("====> PHASE 1: Processing original images with Ollama model")
    handler_original = VLMHandler(
        model_name=original_model_cfg["name"],
        backend=original_model_cfg.get("backend", "transformers"),
        quantize=original_model_cfg.get("quantize", False),
    )
    logging.info(f"Original backend: {handler_original.backend}, model: {handler_original.model_name}")
    
    # Initialize progress bar for original images
    bar_original = tqdm(total=len(to_caption), desc=f"Captioning original images for {seq}", unit="img")
    
    # Process all original images
    for object_id, img_paths in to_caption:
        if "original" not in img_paths:
            bar_original.update(1)
            continue
            
        try:
            # Create caption for original image
            img = Image.open(img_paths["original"]).convert("RGB")
            
            with open("vlm_caption/general_captioning_prompt.md", "r") as f:
                prompt = f.read().strip()
                
            text = handler_original.caption_image(img, prompt=prompt)
            obj_dict[object_id]["best_view"]["original_caption"] = text
            
            if object_id not in captions:
                captions[object_id] = {"captions": {"original": {"text": text, "img_path": img_paths["original"]}}}
            else:
                captions[object_id]["captions"]["original"] = {"text": text, "img_path": img_paths["original"]}
                
        except Exception as e:
            logging.warning(f"Failed to process original image for object {object_id}: {e}")
        finally:
            bar_original.update(1)
    
    bar_original.close()
    
    # Save intermediate results
    if not save_object_dict(obj_dict, dict_path):
        logging.error("Failed to save updated object dict after original image processing")
        return False
        
    # Explicitly unload the original model to free GPU memory
    if handler_original.unload():
        logging.info(f"Successfully unloaded {handler_original.model_name} from memory")
    else:
        logging.warning(f"Failed to unload {handler_original.model_name}, may still be using GPU memory")
    
    # PHASE 2: Process all highlighted images
    logging.info("====> PHASE 2: Processing highlighted images with DAM model")

    # Ensure VRAM is really free before loading the DAM model
    wait_for_gpu_memory(min_free_mb=8192, timeout=300, interval=5)

    handler_highlighted = VLMHandler(
        model_name=highlighted_model_cfg["name"],
        backend=highlighted_model_cfg.get("backend", "transformers"),
        quantize=highlighted_model_cfg.get("quantize", False),
    )
    logging.info(f"Highlighted backend: {handler_highlighted.backend}, model: {handler_highlighted.model_name}")
    
    # Initialize progress bar for highlighted images
    bar_highlighted = tqdm(total=len(to_caption), desc=f"Captioning highlighted images for {seq}", unit="img")
    
    # Process all highlighted images
    for object_id, img_paths in to_caption:
        if "highlighted" not in img_paths:
            bar_highlighted.update(1)
            continue
            
        try:
            # Create caption for highlighted image
            img = Image.open(img_paths["highlighted"]).convert("RGB")
            
            with open("vlm_caption/object_captioning_prompt.md", "r") as f:
                prompt = f.read().strip()
            
            # Build mask for DAM if required
            mask_img = None
            if handler_highlighted.backend == "dam":
                # Retrieve frame_id and mask_id from object dict
                bv = obj_dict[object_id]["best_view"]
                frame_id = bv["frame_id"]
                mask_id = bv["mask_id"]

                seg_path = os.path.join(root, "scannetpp/data", seq, "output/mask", f"frame_{frame_id:06d}.png")
                if not os.path.exists(seg_path):
                    raise FileNotFoundError(f"Segmentation file not found: {seg_path}")
                segmentation = cv2.imread(seg_path, cv2.IMREAD_UNCHANGED)

                binary_mask = (segmentation == mask_id).astype("uint8") * 255

                # Ensure the mask dimensions match the image that will be sent to DAM
                img_w, img_h = img.size  # PIL gives (W, H)
                mask_h, mask_w = binary_mask.shape  # NumPy gives (H, W)

                # 1) Crop using stored bounding box if we are captioning a cropped image
                if (mask_h, mask_w) != (img_h, img_w):
                    bv_meta = obj_dict[object_id]["best_view"]
                    if "bbox" in bv_meta:
                        left, top, right, bottom = bv_meta["bbox"]
                        binary_mask = binary_mask[top:bottom + 1, left:right + 1]
                        mask_h, mask_w = binary_mask.shape

                # 2) If dimensions still mismatch, fall back to a resize (keeps nearest-neighbour sampling)
                if (mask_h, mask_w) != (img_h, img_w):
                    binary_mask = cv2.resize(binary_mask, (img_w, img_h), interpolation=cv2.INTER_NEAREST)

                from PIL import Image as PILImage
                mask_img = PILImage.fromarray(binary_mask)
                
            text = handler_highlighted.caption_image(img, prompt=prompt, mask=mask_img)
            obj_dict[object_id]["best_view"]["highlighted_caption"] = text
            
            if object_id not in captions:
                captions[object_id] = {"captions": {"highlighted": {"text": text, "img_path": img_paths["highlighted"]}}}
            else:
                captions[object_id]["captions"]["highlighted"] = {"text": text, "img_path": img_paths["highlighted"]}
                
        except Exception as e:
            logging.warning(f"Failed to process highlighted image for object {object_id}: {e}")
        finally:
            bar_highlighted.update(1)
    
    bar_highlighted.close()

    # Unload the highlighted model before merging captions to free up GPU memory
    if handler_highlighted.unload():
        logging.info(f"Successfully unloaded {handler_highlighted.model_name} from memory")
    else:
        logging.warning(f"Failed to unload {handler_highlighted.model_name}, may still be using GPU memory")

    # PHASE 3: Merge captions
    logging.info("====> PHASE 3: Merging captions")
    bar_merge = tqdm(total=len(to_caption), desc=f"Merging captions for {seq}", unit="obj")
    
    # Use a dedicated merging model if provided, otherwise fall back to the original model
    merge_model_name = (
        merging_model_cfg["name"] if merging_model_cfg and "name" in merging_model_cfg else original_model_cfg["name"]
    )
    
    for object_id in captions:
        try:
            obj_captions = captions[object_id]["captions"]
            highlighted_text = obj_captions.get("highlighted", {}).get("text", "")
            original_text = obj_captions.get("original", {}).get("text", "")
            
            # Merge the two captions into one cohesive description using an LLM
            merged = merge_captions(
                highlighted_text,
                original_text,
                model_name=merge_model_name,
            )
            
            obj_captions["combined"] = {"text": merged}
            obj_dict[object_id]["best_view"]["combined_caption"] = merged
        except Exception as e:
            logging.warning(f"Failed to merge captions for object {object_id}: {e}")
        finally:
            bar_merge.update(1)
    
    bar_merge.close()
    
    # Save final results
    if not save_object_dict(obj_dict, dict_path):
        logging.error("Failed to save updated object dict")
        return False

    # Save captions JSON
    save_captions(captions, out_dir, seq)
    captions_path = os.path.join(out_dir, f"{seq}.captions.json")
    logging.info(f"Saved {len(captions)} object captions => {captions_path}")

    # Build Faiss index for fast retrieval
    try:
        build_faiss_index(
            captions_path=captions_path,
            out_dir=out_dir,
        )
        logging.info("FAISS index built successfully")
    except Exception as e:
        logging.warning(f"Failed to build FAISS index: {e}")

    return True

def run_vlm_captioning(config_file: str = "vlm_caption/configs/caption.yaml"):
    """
    Run Vision Language Model (VLM) captioning on a set of scenes.
    This function loads configuration, sets up the model, and processes scenes
    to generate captions using a VLM.
    
    The captioning process is sequential:
    1. First process all original images with the Ollama model
    2. Unload the Ollama model to free GPU memory
    3. Then process all highlighted images with the DAM model
    4. Finally merge the captions
    
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

    # Get model configurations
    if "highlighted" in model_cfg and "original" in model_cfg:
        h_cfg = model_cfg["highlighted"]
        o_cfg = model_cfg["original"]
    else:
        # Backwards compatibility: use single model for both
        h_cfg = o_cfg = {
            "name": model_cfg["name"],
            "backend": model_cfg.get("backend", "transformers"),
            "quantize": model_cfg.get("quantize", False),
        }

    # Merger model (optional)
    m_cfg = model_cfg.get("merging", o_cfg)

    out_dir = inference_cfg.get("output_dir", "outputs")
    os.makedirs(out_dir, exist_ok=True)

    success = 0 
    for seq in scenes:
        ok = create_vlm_captions_sequential(
            root=dataset_cfg["root"],
            seq=seq,
            out_dir=out_dir,
            original_model_cfg=o_cfg,
            highlighted_model_cfg=h_cfg,
            merging_model_cfg=m_cfg,
        )
        if ok:
            success += 1

    logging.info(f"Done: {success} succeeded, {len(scenes) - success} failed")

    return os.path.join(out_dir, f"{seq}.captions.json")

if __name__ == "__main__":
    run_vlm_captioning()
