import argparse
import os
import logging
from typing import List, Dict, Optional
from PIL import Image
import torch
from tqdm import tqdm

from vlm_caption.utils import (
    setup_logging,
    ensure_dir_exists,
    load_object_dict,
    save_object_dict_safe,
    save_captions_to_json
)

def parse_args():
    """Parse command line arguments for VLM captioning inference."""
    parser = argparse.ArgumentParser(description="Run VLM captioning on object views")
    parser.add_argument("--config", type=str, required=True, help="Config name (e.g., 'scannet') - also determines dataset type")
    parser.add_argument("--dataset-root", type=str, default="maskclustering/data/", help="Root directory of the dataset")
    parser.add_argument("--seq-name", type=str, help="Scene/sequence name to process (if not specified, will process all scenes from splits file)")
    parser.add_argument("--model-name", type=str, default="Salesforce/blip2-opt-2.7b", help="HuggingFace model name")
    parser.add_argument("--batch-size", type=int, default=4, help="Batch size for inference")
    parser.add_argument("--quantize", action="store_true", help="Use 8-bit quantization")
    parser.add_argument("--output-dir", type=str, default=os.path.join(os.path.dirname(__file__), "outputs"), help="Directory to save caption outputs")
    parser.add_argument("--debug", action="store_true", help="Enable detailed debug logging")
    return parser.parse_args()

def get_dataset(args):
    """
    Get dataset information based on command line arguments.
    
    Parameters
    ----------
    args : argparse.Namespace
        Command line arguments
        
    Returns
    -------
    object
        Dataset object with relevant properties
    """
    class Dataset:
        def __init__(self, name: str, root: str):
            self.name = name
            self.root = root
            
    return Dataset(args.config, args.dataset_root)

def resolve_image_path(rel_path: str, dataset_root: str, dataset_type: str, seq_name: str, debug: bool = False) -> Optional[str]:
    """
    Attempt to resolve an image path from a relative path in the object dictionary.
    
    Parameters
    ----------
    rel_path : str
        Relative path from object dictionary
    dataset_root : str
        Root directory of the dataset
    dataset_type : str
        Type of dataset (e.g., 'scannet', 'scannetpp')
    seq_name : str
        Scene/sequence name
    debug : bool, optional
        Whether to log debug information
        
    Returns
    -------
    Optional[str]
        Resolved absolute path if found, None otherwise
    """
    # Try various ways to resolve the path
    potential_paths = [
        rel_path,  # Maybe it's already absolute
        os.path.join(dataset_root, rel_path),  # Relative to dataset root
        
        # Try based on dataset structure
        os.path.join(dataset_root, "output", "best_views", os.path.basename(rel_path)),
        
        # Dataset specific paths
        os.path.join(dataset_root, dataset_type, "data", seq_name, "output", "best_views", os.path.basename(rel_path)),
        
        # Common output structures
        os.path.join(os.path.dirname(dataset_root), "output", "best_views", os.path.basename(rel_path)),
        os.path.join("output", "best_views", os.path.basename(rel_path)),
        
        # For paths that might be within the sequence directory
        os.path.join(dataset_root, dataset_type, "data", seq_name, rel_path),
        os.path.join(dataset_root, seq_name, rel_path),
    ]
    
    # Find the first path that exists
    for path in potential_paths:
        if os.path.exists(path):
            if debug:
                logging.debug(f"Resolved image path: {path}")
            return path
    
    if debug:
        paths_str = "\n  - ".join(potential_paths)
        logging.debug(f"Tried the following paths but couldn't find image:\n  - {paths_str}")
    
    return None

def get_object_dict_path(dataset_root: str, dataset_type: str, seq_name: str) -> str:
    """
    Get the path to the object dictionary file.
    
    Parameters
    ----------
    dataset_root : str
        Root directory of the dataset
    dataset_type : str
        Type of dataset (e.g., 'scannet')
    seq_name : str
        Scene/sequence name
        
    Returns
    -------
    str
        Path to the object dictionary file
    """
    if dataset_type == "scannet":
        return os.path.join(dataset_root, "scannet", "processed", seq_name, "output", "best_views", "best_view_object_dict.pth")
    elif dataset_type == "matterport3d":
        return os.path.join(dataset_root, "matterport3d", "scans", seq_name, seq_name, "output", "best_views", "best_view_object_dict.pth")
    elif dataset_type == "scannetpp":
        return os.path.join(dataset_root, "scannetpp", "data", seq_name, "output", "best_views", "best_view_object_dict.pth")
    else:
        return os.path.join(dataset_root, "demo", seq_name, "output", "best_views", "best_view_object_dict.pth")

def load_vlm_model(model_name: str, quantize: bool = False):
    """
    Load the VLM model for image captioning.
    
    Parameters
    ----------
    model_name : str
        Model name/path for HuggingFace model
    quantize : bool, optional
        Whether to use 8-bit quantization
        
    Returns
    -------
    callable
        Captioning pipeline function
    """
    logging.info(f"Loading model: {model_name}")
    
    if "InternVL3" in model_name:
        from transformers import AutoProcessor, AutoModelForCausalLM, BitsAndBytesConfig
        
        processor = AutoProcessor.from_pretrained(model_name)
        if quantize:
            quantization_config = BitsAndBytesConfig(load_in_8bit=True)
            model = AutoModelForCausalLM.from_pretrained(
                model_name, device_map="auto", quantization_config=quantization_config
            )
        else:
            model = AutoModelForCausalLM.from_pretrained(
                model_name, device_map="auto"
            )
            
        def pipeline(images: List[Image.Image]) -> List[Dict[str, str]]:
            outputs = []
            detailed_prompt = "Describe this image in comprehensive detail. Include information about all objects, their relationships, colors, textures, materials, spatial arrangements, and any other relevant visual details."
            for image in images:
                inputs = processor(text=detailed_prompt, images=image, return_tensors="pt").to(model.device)
                with torch.no_grad():
                    generated_ids = model.generate(
                        **inputs,
                        max_length=500,
                        do_sample=True,
                        temperature=0.7,
                        top_p=0.9
                    )
                generated_text = processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
                outputs.append({"generated_text": generated_text})
            return outputs
            
    elif "blip2" in model_name.lower():
        from transformers import Blip2Processor, Blip2ForConditionalGeneration, BitsAndBytesConfig
        
        processor = Blip2Processor.from_pretrained(model_name, use_fast=True)
        if quantize:
            quantization_config = BitsAndBytesConfig(load_in_8bit=True)
            model = Blip2ForConditionalGeneration.from_pretrained(
                model_name, device_map="auto", quantization_config=quantization_config
            )
        else:
            model = Blip2ForConditionalGeneration.from_pretrained(
                model_name, device_map="auto"
            )
            
        def pipeline(images: List[Image.Image]) -> List[Dict[str, str]]:
            outputs = []
            for image in images:
                inputs = processor(images=image, return_tensors="pt").to(model.device)
                generated_ids = model.generate(
                    **inputs, 
                    max_length=500,
                )
                generated_text = processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
                outputs.append({"generated_text": generated_text})
            return outputs
            
    else:
        from transformers import BlipProcessor, BlipForConditionalGeneration, BitsAndBytesConfig
        
        processor = BlipProcessor.from_pretrained(model_name, use_fast=True)
        if quantize:
            quantization_config = BitsAndBytesConfig(load_in_8bit=True)
            model = BlipForConditionalGeneration.from_pretrained(
                model_name, device_map="auto", quantization_config=quantization_config
            )
        else:
            model = BlipForConditionalGeneration.from_pretrained(
                model_name, device_map="auto"
            )
            
        def pipeline(images: List[Image.Image]) -> List[Dict[str, str]]:
            outputs = []
            for image in images:
                text = "Describe this image"
                    
                inputs = processor(image, text, return_tensors="pt").to(model.device)
                generated_ids = model.generate(
                    **inputs, 
                    max_length=250, 
                    do_sample=True,
                    num_beams=4,
                    length_penalty=1.5
                )
                generated_text = processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
                outputs.append({"generated_text": generated_text})
            return outputs
    
    return pipeline

def load_scene_names(dataset_type: str) -> List[str]:
    """
    Load scene names from splits file.
    
    Parameters
    ----------
    dataset_type : str
        Type of dataset (e.g., 'scannet', 'matterport3d', 'scannetpp')
        
    Returns
    -------
    List[str]
        List of scene names to process
    """
    splits_path = f"maskclustering/splits/{dataset_type}.txt"
    if not os.path.exists(splits_path):
        raise FileNotFoundError(f"Splits file not found: {splits_path}")
    
    with open(splits_path, 'r') as f:
        return [line.strip() for line in f if line.strip()]

def process_scene(pipe, dataset, seq_name: str, batch_size: int, output_dir: str, debug: bool = False) -> bool:
    """
    Process a single scene with the VLM pipeline.
    
    Parameters
    ----------
    pipe : callable
        VLM pipeline function
    dataset : object
        Dataset object with name and root attributes
    seq_name : str
        Scene/sequence name to process
    batch_size : int
        Batch size for processing images
    output_dir : str
        Directory to save caption outputs
    debug : bool, optional
        Whether to enable detailed debug logging
        
    Returns
    -------
    bool
        True if processing was successful, False otherwise
    """
    logging.info(f"Processing sequence: {seq_name}")
    
    # Get object dictionary path
    dict_path = get_object_dict_path(dataset.root, dataset.name, seq_name)
    
    # Check if path exists
    if not os.path.exists(dict_path):
        logging.error(f"Object dictionary not found at: {dict_path}")
        logging.info(f"Checking alternative paths for {seq_name}...")
        
        # Try alternate paths for scannetpp specifically
        if dataset.name == "scannetpp":
            alt_paths = [
                os.path.join(dataset.root, "scannetpp", "data", seq_name, "output", "best_views", "best_view_object_dict.pth"),
                os.path.join(dataset.root, seq_name, "output", "best_views", "best_view_object_dict.pth")
            ]
            
            for alt_path in alt_paths:
                if os.path.exists(alt_path):
                    logging.info(f"Found alternative path at: {alt_path}")
                    dict_path = alt_path
                    break
            else:
                logging.error(f"Could not find object dictionary for {seq_name} after checking alternatives")
                return False
        else:
            return False
    
    # Load object dictionary
    try:
        obj_dict = load_object_dict(dataset.root, dataset.name, seq_name)
        logging.info(f"Successfully loaded object dictionary with {len(obj_dict)} objects")
        
        if debug and obj_dict:
            # Log the first object to understand structure
            first_obj_id = next(iter(obj_dict))
            logging.debug(f"First object structure: {obj_dict[first_obj_id]}")
    except Exception as e:
        logging.error(f"Failed to load object dictionary: {e}")
        return False
    
    # Build to_caption list - include all objects with images regardless of existing captions
    to_caption = []
    for obj_id, obj_data in obj_dict.items():
        if "best_view" in obj_data:
            if "image_path" in obj_data["best_view"]:
                rel_path = obj_data["best_view"]["image_path"]
                abs_path = resolve_image_path(rel_path, dataset.root, dataset.name, seq_name, debug)
                
                if abs_path:
                    to_caption.append((obj_id, abs_path))
                else:
                    logging.warning(f"Image not found for object {obj_id}: {rel_path}")
    
    if not to_caption:
        logging.info(f"No objects with images found in {seq_name}")
        return True
        
    logging.info(f"Found {len(to_caption)} objects to caption")
    
    # Batch through to_caption
    captions = {}
    progress_bar = tqdm(total=len(to_caption), desc=f"Captioning {seq_name}", unit="obj")
    
    for i in range(0, len(to_caption), batch_size):
        batch = to_caption[i:i+batch_size]
        
        # Load images
        images = []
        valid_indices = []
        for idx, (obj_id, img_path) in enumerate(batch):
            try:
                img = Image.open(img_path).convert("RGB")
                images.append(img)
                valid_indices.append(idx)
            except Exception as e:
                logging.warning(f"Failed to load image for object {obj_id} from {img_path}: {e}")
                progress_bar.update(1)  # Count failed images in progress
        
        if not images:
            continue
            
        # Generate captions
        results = pipe(images)
        
        # Update object dictionary
        for res_idx, batch_idx in enumerate(valid_indices):
            obj_id, _ = batch[batch_idx]
            caption = results[res_idx]["generated_text"].strip()
            obj_dict[obj_id]["best_view"]["caption"] = caption.strip()
            captions[obj_id] = caption.strip()
            progress_bar.update(1)
    
    progress_bar.close()
    
    # Save updated object dictionary
    if save_object_dict_safe(obj_dict, dict_path):
        logging.info(f"Saved updated object dictionary to {dict_path}")
    else:
        logging.error(f"Failed to save object dictionary to {dict_path}")
        return False
    
    # Write out captions JSON
    if captions:
        ensure_dir_exists(output_dir)
        output_file = os.path.join(output_dir, f"{seq_name}.captions.json")
        save_captions_to_json(captions, output_file)
        logging.info(f"Saved {len(captions)} captions to {output_file}")
        return True
    
    return True

def main():
    """Main entry point for the VLM captioning inference."""
    args = parse_args()
    setup_logging()
    
    # Configure debug logging if requested
    if args.debug:
        logging.getLogger().setLevel(logging.DEBUG)
        logging.debug("Debug logging enabled")
    
    # Load model once for all scenes
    pipe = load_vlm_model(args.model_name, args.quantize)
    dataset = get_dataset(args)
    
    if args.seq_name:
        scene_names = [args.seq_name]
    else:
        scene_names = load_scene_names(args.config)
        logging.info(f"Loaded {len(scene_names)} scenes from splits file")
    
    success_count = 0
    failure_count = 0
    
    for seq_name in scene_names:
        try:
            if process_scene(pipe, dataset, seq_name, args.batch_size, args.output_dir, args.debug):
                success_count += 1
            else:
                failure_count += 1
        except Exception as e:
            logging.error(f"Failed to process scene {seq_name}: {e}")
            failure_count += 1
            continue
    
    if failure_count == 0:
        logging.info(f"VLM captioning completed successfully for all {success_count} scenes")
    else:
        logging.warning(f"VLM captioning completed with {success_count} successes and {failure_count} failures")

if __name__ == "__main__":
    main()
