import os
import yaml
import logging
import tempfile
import json
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
        from ollama import chat  # local LLM backend; same dependency used in query stage

        system_prompt = (
            "You are a helpful assistant. Merge two descriptions that refer to the *same* object into a single "
            "caption. Give higher weight to the first description, which focuses only on the object, and use "
            "information from the second description only to fill in useful context. Keep as much detail from both descriptions as possible,"
            "while avoiding repetition and emphasizing the object as it is described by the first description."
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

def create_vlm_captions(handler_highlighted: VLMHandler, handler_original: VLMHandler, root: str, seq: str, out_dir: str) -> bool:
    logging.info(f"====> Processing scene {seq}")
    logging.info(f"Highlighted backend: {handler_highlighted.backend}, model: {handler_highlighted.model_name}")
    logging.info(f"Original backend: {handler_original.backend}, model: {handler_original.model_name}")

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
    bar = tqdm(total=len(to_caption) * 3, desc=f"Captioning {seq}", unit="step")
    # Create captions for all relevant objects
    for object_id, img_paths in to_caption:
        generated_obj_captions = generate_captions_for_object(
            handler_highlighted,
            handler_original,
            object_id,
            img_paths,
            obj_dict,
            bar,
            root_dir=root,
            seq_name=seq,
        )
        captions[object_id] = {"captions": generated_obj_captions}

    bar.close()

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

    # Support separate models for highlighted vs original images.
    if "highlighted" in model_cfg and "original" in model_cfg:
        h_cfg = model_cfg["highlighted"]
        o_cfg = model_cfg["original"]

        handler_highlighted = VLMHandler(
            model_name=h_cfg["name"],
            backend=h_cfg.get("backend", "transformers"),
            quantize=h_cfg.get("quantize", False),
        )

        handler_original = VLMHandler(
            model_name=o_cfg["name"],
            backend=o_cfg.get("backend", "transformers"),
            quantize=o_cfg.get("quantize", False),
        )
    else:
        # Backwards compatibility: use single model for both
        handler_highlighted = handler_original = VLMHandler(
            model_name=model_cfg["name"],
            backend=model_cfg.get("backend", "transformers"),
            quantize=model_cfg.get("quantize", False),
        )

    out_dir = inference_cfg.get("output_dir", "outputs")

    success = 0 
    for seq in scenes:
        ok = create_vlm_captions(
            handler_highlighted,
            handler_original,
            root=dataset_cfg["root"],
            seq=seq,
            out_dir=out_dir,
        )
        if ok:
            success += 1

    logging.info(f"Done: {success} succeeded, {len(scenes) - success} failed")

    return os.path.join(out_dir, f"{seq}.captions.json")

if __name__ == "__main__":
    run_vlm_captioning()
