import argparse
import os
import sys
import subprocess
from typing import Optional, List

def parse_args():
    """
    Parse command line arguments.
    
    Returns
    -------
    argparse.Namespace
        Parsed command line arguments.
    """
    parser = argparse.ArgumentParser(description="Run scene-grounding pipeline")
    
    parser.add_argument(
        "--mask_cluster", 
        action="store_true",
        help="Run mask clustering"
    )
    parser.add_argument(
        "--caption", 
        action="store_true",
        help="Run VLM captioning"
    )
    parser.add_argument(
        "--generate_captions", 
        action="store_true",
        help="Generate captions using VLM after mask clustering"
    )
    parser.add_argument(
        "--dataset_root", 
        type=str, 
        default="./data",
        help="Root directory of the dataset"
    )
    parser.add_argument(
        "--dataset", 
        type=str, 
        required=True,
        choices=["scannet", "scannetpp", "matterport3d", "demo"],
        help="Dataset type"
    )
    parser.add_argument(
        "--seq_name", 
        type=str, 
        required=True,
        help="Scene/sequence name"
    )
    parser.add_argument(
        "--config", 
        type=str, 
        default=None,
        help="Configuration name for mask clustering (defaults to dataset name)"
    )
    parser.add_argument(
        "--quantize", 
        action="store_true",
        help="Use 8-bit quantization for VLM captioning"
    )
    parser.add_argument(
        "--quantize_captions", 
        action="store_true",
        help="Use 8-bit quantization specifically for VLM captioning"
    )
    
    return parser.parse_args()

def run_mask_clustering(
    dataset_type: str,
    seq_name: str,
    config: Optional[str] = None,
) -> None:
    """
    Run the mask clustering pipeline.
    
    Parameters
    ----------
    dataset_type : str
        Type of dataset (e.g., 'scannet', 'matterport3d', 'scannetpp').
    seq_name : str
        Scene/sequence name.
    config : Optional[str], optional
        Configuration name. Default is None.
    """
    from maskclustering.main import main
    from maskclustering.utils.config import get_args
    
    # Use dataset name as default config if not specified
    if config is None:
        config = dataset_type
    
    # Set up arguments
    sys_args = [
        "--config", config,
        "--seq_name", seq_name,
    ]
    args = get_args(sys_args)
    
    # Run mask clustering
    main(args)

def run_vlm_captioning(
    dataset_root: str,
    dataset_type: str,
    seq_name: str,
    quantize: bool = False,
) -> None:
    """
    Run the VLM captioning pipeline.
    
    Parameters
    ----------
    dataset_root : str
        Root directory of the dataset.
    dataset_type : str
        Type of dataset (e.g., 'scannet', 'matterport3d', 'scannetpp').
    seq_name : str
        Scene/sequence name.
    quantize : bool, optional
        Whether to use 8-bit quantization. Default is False.
    """
    from vlm_caption.infer import process_scene
    from vlm_caption.model_loader import get_image_caption_pipeline
    from vlm_caption.utils import setup_logging
    import logging
    
    # Set up logging
    setup_logging()
    
    # Load the VLM model
    logging.info("Loading VLM model...")
    pipeline = get_image_caption_pipeline("Salesforce/blip2-opt-2.7b", quantize)
    logging.info("VLM model loaded.")
    
    # Run captioning
    process_scene(
        pipeline,
        dataset_root,
        dataset_type,
        seq_name,
    )

def main():
    """Main entry point for the scene-grounding pipeline."""
    args = parse_args()
    
    # Run mask clustering if requested
    if args.mask_cluster:
        print("Running mask clustering...")
        run_mask_clustering(args.dataset, args.seq_name, args.config)
        print("Mask clustering completed.")
        
        # Generate captions after clustering if requested
        if args.generate_captions:
            print("Generating captions after mask clustering...")
            subprocess.run([
                "python", "-m", "vlm_caption.infer",
                "--config", args.config or args.dataset,
                "--dataset-root", args.dataset_root,
                "--seq-name", args.seq_name,
                "--model-name", "Salesforce/blip2-opt-2.7b",
                *(["--quantize"] if args.quantize_captions else [])
            ], check=True)
            print("Caption generation completed.")
    
    # Run VLM captioning if requested (separate from mask clustering)
    if args.caption:
        print("Running VLM captioning...")
        run_vlm_captioning(args.dataset_root, args.dataset, args.seq_name, args.quantize)
        print("VLM captioning completed.")
    
    # If neither is specified, show help
    if not (args.mask_cluster or args.caption):
        print("No action specified. Please use --mask_cluster or --caption.")
        return 1
    
    return 0

if __name__ == "__main__":
    sys.exit(main())
