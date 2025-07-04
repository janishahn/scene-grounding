import os
import yaml
import logging
import tempfile
import json
import time
import torch
import cv2

from typing import List, Dict
from PIL import Image
from tqdm import tqdm
from vlm_caption.vlm_handler import VLMHandler

# PyTorch performance settings
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.benchmark = True

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
    Collects absolute image paths for objects that have highlighted and original best views.

    Preference order:
        highlighted -> original_path
        original    -> original_path

    We pass the full-size highlighted image to DAM so that the pixel mask aligns with the image dimensions.
    """
    to_caption = []
    for object_id, data in obj_dict.items():
        best_view = data.get("best_view", {})
        if not best_view:
            continue

        preferred: Dict[str, List[str]] = {
            "highlighted": ["original_path"],
            "original": ["original_path"],
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

def create_object_specific_captions(
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
            caption_img_path = img_path

            if img_type == "highlighted":
                bv_meta = obj_dict[object_id].get("best_view", {})
                # Prefer the full-size unaltered frame.
                for alt_key in ("original_path",):
                    rel_alt = bv_meta.get(alt_key)
                    if rel_alt:
                        alt_path = os.path.join(root_dir, "scannetpp/data", seq_name, rel_alt)
                        if os.path.exists(alt_path):
                            caption_img_path = alt_path
                            break

            # Load image that will actually be sent to the captioning backend
            img = Image.open(caption_img_path).convert("RGB")

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

                mask_img = Image.fromarray(binary_mask)

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

def create_general_captions(
    root: str,
    seq: str,
    out_dir: str,
    original_model_cfg: dict,
    highlighted_model_cfg: dict,
    merging_model_cfg: dict | None = None,
    num_best_views: int = 1,
    save_debug: bool = False,
) -> bool:
    """Generate global and local DAM captions for every object of seq.

    1. *Global caption* - object in full-scene context, using the prompt in
       ``prompts/global_caption_prompt.md``.
    2. *Local caption* - object-intrinsic description, using the prompt in
       ``prompts/local_caption_prompt.md``.

    Captions are written to ``<out_dir>/<seq>.captions.jsonl`` with one JSON
    object per line:
        {"scene_id": <str>, "object_id": <int>, "global": <str>, "local": <str>}.
    """
    logging.info(f"====> Processing scene {seq}")

    # ------------------------------------------------------------------
    # Load object dictionary and sanity-check
    # ------------------------------------------------------------------
    dict_path = os.path.join(root, "scannetpp/data", seq, "output/best_views/best_view_object_dict.pth")
    obj_dict = load_object_dict(dict_path)
    logging.info(f"Loaded {len(obj_dict)} objects in dict")
    if len(obj_dict) == 0:
        logging.warning("Empty object dictionary - nothing to caption.")
        return False

    # Ensure adequate VRAM is available before we start captioning.
    wait_for_gpu_memory(min_free_mb=8192, timeout=300, interval=5)

    dam_cfg = highlighted_model_cfg if highlighted_model_cfg else original_model_cfg
    handler = VLMHandler(
        model_name=dam_cfg["name"],
        backend="dam",
        quantize=dam_cfg.get("quantize", False),
    )
    logging.info(f"Using DAM model: {handler.model_name}")

    # Load prompts
    with open("vlm_caption/prompts/global_caption_prompt.md", "r") as f:
        global_prompt = f.read().strip()
    with open("vlm_caption/prompts/local_caption_prompt.md", "r") as f:
        local_prompt = f.read().strip()

    # Pre-compute actual number of views we will attempt (up to num_best_views per object)
    total_views = 0
    for data in obj_dict.values():
        v = data.get("views")
        if not v:
            bv = data.get("best_view", {})
            v = [bv] if bv else []
        total_views += min(len(v), num_best_views)

    total_steps = total_views * 2  # two caption passes per view
    bar = tqdm(total=total_steps, desc=f"Captioning {seq}", unit="step")

    captions_jsonl: list[str] = []

    for object_id, data in obj_dict.items():
        views: list[dict] | None = data.get("views")
        if not views:
            bv = data.get("best_view", {})
            views = [bv] if bv else []

        if not views:
            # No imagery available – skip object entirely (nothing to update since not counted)
            continue

        views = views[:num_best_views]

        # Create debug directory lazily if debugging is enabled
        if save_debug:
            dbg_dir_root = os.path.join(out_dir, "debug_masks", seq)
            os.makedirs(dbg_dir_root, exist_ok=True)

        global_caps: list[str] = []
        local_caps: list[str] = []

        for view_idx, view_meta in enumerate(views):
            # Resolve original image path for this view
            rel_img = view_meta.get("original_path")
            if not rel_img:
                bar.update(2)
                continue
            img_path = os.path.join(root, "scannetpp/data", seq, rel_img)
            if not os.path.exists(img_path):
                logging.warning(f"Object {object_id}: image not found – {img_path}")
                bar.update(2)
                continue

            try:
                img = Image.open(img_path).convert("RGB")
            except Exception as e:
                logging.warning(f"Object {object_id}: failed to load image - {e}")
                bar.update(2)
                continue

            # Build binary mask
            try:
                frame_id = view_meta["frame_id"]
                mask_id = view_meta["mask_id"]
                seg_path = os.path.join(root, "scannetpp/data", seq, "output/mask", f"frame_{frame_id:06d}.png")
                if not os.path.exists(seg_path):
                    raise FileNotFoundError(seg_path)
                segmentation = cv2.imread(seg_path, cv2.IMREAD_UNCHANGED)
                binary_mask = (segmentation == mask_id).astype("uint8") * 255

                img_w, img_h = img.size
                mask_h, mask_w = binary_mask.shape
                if (mask_w, mask_h) != (img_w, img_h):
                    binary_mask = cv2.resize(binary_mask, (img_w, img_h), interpolation=cv2.INTER_NEAREST)

                mask_img = Image.fromarray(binary_mask)

                # ----------------------------------------------------------------------
                # Debug: persist inputs (original image + binary mask) for inspection
                # ----------------------------------------------------------------------
                if save_debug:
                    try:
                        # File stem: obj<id>_view<idx>
                        stem = f"obj{object_id}_view{view_idx}"
                        img_save = os.path.join(dbg_dir_root, f"{stem}_img.png")
                        mask_save = os.path.join(dbg_dir_root, f"{stem}_mask.png")

                        # Only save if not already present to avoid unnecessary disk IO
                        if not os.path.exists(img_save):
                            img.save(img_save)
                        if not os.path.exists(mask_save):
                            mask_img.save(mask_save)
                    except Exception as dbg_err:
                        logging.debug(f"Failed to save debug inputs for object {object_id}, view {view_idx}: {dbg_err}")
            except Exception as e:
                logging.warning(f"Object {object_id}: failed to build mask - {e}")
                bar.update(2)
                continue

            # ---- Caption passes ----
            gcap = handler.caption_image(img, prompt=global_prompt, mask=mask_img)
            bar.update(1)
            lcap = handler.caption_image(img, prompt=local_prompt, mask=mask_img)
            bar.update(1)

            # Save into dict + lists
            view_meta["global_caption"] = gcap
            view_meta["local_caption"] = lcap
            global_caps.append(gcap)
            local_caps.append(lcap)

        if global_caps or local_caps:
            captions_jsonl.append(json.dumps({
                "scene_id": seq,
                "object_id": object_id,
                "global": global_caps,
                "local": local_caps,
            }, ensure_ascii=False))

    bar.close()

    # Save results
    if not save_object_dict(obj_dict, dict_path):
        logging.error("Failed to save updated object dict")
        return False

    os.makedirs(out_dir, exist_ok=True)
    captions_path = os.path.join(out_dir, f"{seq}.captions.jsonl")
    with open(captions_path, "w", encoding="utf-8") as f:
        f.write("\n".join(captions_jsonl))

    logging.info(f"Saved captions ⇒ {captions_path}")

    # Unload DAM to free GPU memory
    handler.unload()

    # Convert captions to structured XML via Gemma LLM
    try:
        from vlm_caption.xml_structuring import jsonl_to_xml

        xml_path = os.path.join(out_dir, f"{seq}.xml")
        jsonl_to_xml(captions_path, xml_path, scene_id=seq, model=(merging_model_cfg or {}).get("xml_model", "gemma3:4b-it-qat"))

        # Build unified FAISS index for retrieve & rerank pipeline (Phase 1)
        try:
            from llm_query.build_unified_index import build_unified_index
            build_unified_index(xml_path, out_dir)
        except Exception as ind_err:
            logging.warning(f"Failed to build unified index: {ind_err}")
    except Exception as e:
        logging.warning(f"Failed to convert captions to XML: {e}")

    return True

def run_vlm_captioning(config_file: str = "vlm_caption/configs/caption.yaml"):
    """
    Run the two-pass NVIDIA DAM captioning pipeline (global & local) over a
    collection of scenes.  Captions are saved as ``*.captions.jsonl`` files in
    ``output_dir``.  Each line contains the scene id, object id, and the two
    caption types.

    Args:
        config_file (str, optional): Path to the YAML configuration file.
            Defaults to "vlm_caption/configs/caption.yaml".
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
        ok = create_general_captions(
            root=dataset_cfg["root"],
            seq=seq,
            out_dir=out_dir,
            original_model_cfg=o_cfg,
            highlighted_model_cfg=h_cfg,
            merging_model_cfg=m_cfg,
            num_best_views=inference_cfg.get("num_best_views", 1),
            save_debug=inference_cfg.get("debug", False),
        )
        if ok:
            success += 1

    logging.info(f"Done: {success} succeeded, {len(scenes) - success} failed")

    return os.path.join(out_dir, f"{seq}.captions.jsonl")

if __name__ == "__main__":
    run_vlm_captioning()
