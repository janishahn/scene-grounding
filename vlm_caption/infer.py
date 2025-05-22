import os
import yaml
import logging
import argparse
import tempfile
import json
from typing import List, Dict
from PIL import Image
import torch
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
    if not os.path.exists(splits_file):
        raise FileNotFoundError(f"Splits file not found: {splits_file}")
    with open(splits_file) as f:
        return [line.strip() for line in f if line.strip()]

def load_object_dict(path: str) -> dict:
    try:
        return torch.load(path, weights_only=False)
    except Exception:
        return torch.load(path, map_location="cpu", weights_only=False)

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

def process_scene(handler: ModelHandler, root: str, seq: str, out_dir: str) -> bool:
    logging.info(f"====> Processing scene {seq}")
    dict_path = os.path.join(root, "scannetpp/data", seq, "output/best_views/best_view_object_dict.pth")
    if not os.path.exists(dict_path):
        logging.error(f"Object dict not found: {dict_path}")
        return False

    obj_dict = load_object_dict(dict_path)
    logging.info(f"Loaded {len(obj_dict)} objects in dict")

    to_caption = []
    for oid, data in obj_dict.items():
        bv = data.get("best_view", {})
        if not bv:
            continue
            
        paths = {}
        for img_type in ["cropped", "original"]:
            rel_path = bv.get(f"{img_type}_path")
            if not rel_path:
                continue
            
            abs_path = os.path.join(root, "scannetpp", "data", seq, rel_path)
            if os.path.exists(abs_path):
                paths[img_type] = abs_path
            else:
                logging.warning(f"Image not found for object {oid}: {abs_path}")
                
        if len(paths) == 2:
            to_caption.append((oid, paths))
            
    if not to_caption:
        logging.info("Nothing to caption")
        return True

    captions: Dict[int, Dict[str, str]] = {}
    bar = tqdm(total=len(to_caption) * 2, desc=f"Captioning {seq}", unit="img")
    
    for oid, img_paths in to_caption:
        obj_captions = {"cropped": "", "original": ""}
        
        for img_type, img_path in img_paths.items():
            try:
                img = Image.open(img_path).convert("RGB")
                text = handler.caption_image(img)
                obj_dict[oid]["best_view"][f"{img_type}_caption"] = text
                obj_captions[img_type] = text
            except Exception as e:
                logging.warning(f"Failed to process {img_type} image {img_path}: {e}")
            finally:
                bar.update(1)
        
        captions[oid] = obj_captions

    bar.close()

    if not save_object_dict(obj_dict, dict_path):
        logging.error("Failed to save updated object dict")
        return False

    save_captions(captions, out_dir, seq)
    logging.info(f"Saved {len(captions)} object captions => {out_dir}/{seq}.captions.json")
    return True

def main():
    p = argparse.ArgumentParser()
    p.add_argument("-c", "--config-file", default="vlm_caption/caption.yaml",
                   help="Path to YAML config")
    args = p.parse_args()

    cfg = load_config(args.config_file)
    ds = cfg["dataset"]
    mi = cfg["model"]
    inf = cfg["inference"]

    setup_logging(inf.get("debug", False))

    if inf.get("seq_names"):
        scenes = inf["seq_names"]
    else:
        scenes = read_splits(ds["splits_file"])

    handler = ModelHandler(
        model_name=mi["name"],
        backend=mi.get("backend", "transformers"),
        quantize=mi.get("quantize", False)
    )

    success = fail = 0
    for seq in scenes:
        ok = process_scene(
            handler,
            root=ds["root"],
            seq=seq,
            out_dir=inf.get("output_dir", "outputs")
        )

        if ok:
            success += 1
        else:
            fail += 1

    logging.info(f"Done: {success} succeeded, {fail} failed")

if __name__ == "__main__":
    main()
