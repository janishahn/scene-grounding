import sys
import logging 
import yaml 
import os
import subprocess

# Updated VLM imports for config-driven approach
from vlm_caption.infer import run_vlm_captioning
from llm_query.query import query_scene

def setup_main_logging(level: int = logging.INFO) -> None:
    """
    Set up logging configuration for the main orchestrator.
    """
    logging.basicConfig(
        level=level,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

def load_pipeline_config(config_path: str = "pipeline_config.yaml") -> dict:
    """
    Loads the main pipeline YAML configuration file.
    """
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config

def run_mask_clustering_from_config(cfg: dict) -> None:
    """
    Run the mask clustering pipeline using settings from the loaded config via a subprocess.
    
    Parameters
    ----------
    cfg : dict
        The full pipeline configuration dictionary.
    """
    logging.info("Starting mask clustering pipeline via subprocess...")
    
    mc_cfg = cfg["mask_clustering"]
    # Construct command line arguments for maskclustering/run.py
    cmd_args = [
        "--config", mc_cfg["config_name"],
        # Ensure steps_to_run are passed as multiple arguments if needed by the script
        # The original maskclustering/run.py takes them as separate strings after --steps_to_run
    ]
    # Add steps_to_run arguments correctly
    cmd_args.append("--steps_to_run")
    cmd_args.extend(mc_cfg["steps_to_run"])

    if mc_cfg.get("debug", False):
        cmd_args.append("--debug")

    command = [sys.executable, "run.py"] + cmd_args
    logging.info(f"Mask clustering command: {' '.join(command)}")

    # Prepare environment for the subprocess
    subprocess_env = os.environ.copy()

    try:
        logging.info("Executing mask clustering pipeline...")
        # Execute maskclustering/run.py with the specificied arguments
        completed_process = subprocess.run(command, env=subprocess_env, check=True, capture_output=True, text=True, cwd='maskclustering')
        logging.info("Mask clustering pipeline completed successfully.")
        logging.info(f"Subprocess STDOUT: {completed_process.stdout} ")
    except subprocess.CalledProcessError as e:
        logging.error(f"Mask clustering subprocess failed with exit code {e.returncode}")
        logging.error("Subprocess STDERR:")
        logging.error(e.stderr)
        logging.error("Subprocess STDOUT:")
        logging.error(e.stdout)
        raise # Re-raise the exception to indicate failure
    except FileNotFoundError:
        logging.error(f"Error: The script 'maskclustering/run.py' was not found. Make sure the path is correct.")
        raise

def main():
    """Main entry point for the scene-grounding pipeline"""
    pipeline_cfg = load_pipeline_config(config_path="pipeline_config.yaml")

    main_log_level = logging.DEBUG if pipeline_cfg.get("general", {}).get("debug_logging", False) else logging.INFO
    setup_main_logging(level=main_log_level)

    if pipeline_cfg.get("run_mask_clustering_pipeline", False):
        try:
            run_mask_clustering_from_config(pipeline_cfg)
        except Exception as e:
            logging.error(f"Mask clustering pipeline failed: {e}")
            sys.exit(1) # Exit if mask clustering fails

    if pipeline_cfg.get("run_vlm_captioning_pipeline", False):
        try:
            run_vlm_captioning() # defaults to vlm_caption/configs/caption.yaml
        except Exception as e:
            logging.error(f"VLM captioning pipeline failed: {e}")
            sys.exit(1) # Exit if VLM captioning fails

    if pipeline_cfg.get("run_llm_query_pipeline", False):
        try:
            with open('vlm_caption/configs/caption.yaml', 'r') as f:
                vlm_config = yaml.safe_load(f)
            query_scene(captions_path="vlm_caption/outputs/88cf747085.captions.json")
        except Exception as e:
            logging.error("LLM querying has thrown the following error: {e} ")
            sys.exit(1)

    logging.info("Scene grounding pipeline orchestration finished successfully.")
    return 0

if __name__ == "__main__":
    sys.exit(main())
