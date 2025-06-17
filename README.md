# Scene-Grounding: Open-Vocabulary 3D Scene Understanding with Vision-Language Models

## Overview
This project extends the MaskClustering pipeline for open-vocabulary 3D instance segmentation by integrating vision-language (VLM) and large language model (LLM) modules. The system enables detailed, object-centric scene understanding and natural language querying of 3D environments. Our key contributions are:

- **Vision-Language Captioning**: Automatic generation of rich, multi-level natural language descriptions for 3D scene objects using state-of-the-art VLMs.
- **LLM-Based Scene Querying**: Natural language interface for identifying and retrieving objects in 3D scenes based on user queries, leveraging LLM reasoning over VLM-generated captions.
- **Pipeline Orchestration**: Modular orchestration of MaskClustering, VLM captioning, and LLM querying for end-to-end open-vocabulary 3D scene understanding.

The core 3D instance segmentation and mask clustering is provided by the [MaskClustering](https://arxiv.org/abs/2401.07745) pipeline (CVPR 2024), which is used as a foundation with minor modifications.

## Pipeline Structure
1. **3D Instance Segmentation**: MaskClustering merges 2D instance masks from RGB-D scans into 3D object instances using multi-view verification.
2. **Best-View Selection**: For each 3D object, the most informative 2D view is selected and highlighted.
3. **Vision-Language Captioning**: VLMs generate detailed captions for both the highlighted object view and the original scene context.
4. **LLM Querying**: Users can query the scene in natural language; an LLM matches the query to the most relevant object using the generated captions.

## Installation
1. Install [PyTorch](https://pytorch.org/) and [Pytorch3D](https://github.com/facebookresearch/pytorch3d) (see `maskclustering/README.md` for details).
2. Install project dependencies:
```bash
pip install -r requirements.txt
```
3. (Optional) For VLM and LLM backends, install [transformers](https://huggingface.co/docs/transformers/index), [ollama](https://ollama.com/), or other supported libraries as needed.

## Usage
### 1. Data Preparation & MaskClustering
Follow the instructions in `maskclustering/README.md` to prepare data and run the 3D instance segmentation pipeline. Example:
```bash
python run.py --config scannetpp
```

### 2. Vision-Language Captioning
Generate captions for all detected objects:
```bash
python run.py  # Ensure `run_vlm_captioning_pipeline: true` in `pipeline_config.yaml`
```
Captions are saved as JSON in `vlm_caption/outputs/`.

### 3. LLM-Based Scene Querying
Query the scene using natural language:
```bash
python run.py  # Ensure `run_llm_query_pipeline: true` in `pipeline_config.yaml`
```
You will be prompted for a query. The system returns the image path of the best-matching object.

## Example Output
A sample object caption (highlighted view):
```
The object is a large, circular ventilation grate, likely part of an industrial or mechanical system. Its primary function is to provide airflow and exhaust fumes or heated air... [truncated]
```
A sample LLM query and response:
```
User: "Find the object used for washing hands."
LLM: {
  "object_id": "5",
  "reasoning": "This object is a bathroom sink with faucet, which matches the query for a place to wash hands."
}
```

## Configuration
- `pipeline_config.yaml`: Controls which pipeline stages to run.
- `vlm_caption/configs/caption.yaml`: VLM model and dataset settings.
- `llm_query/query.yaml`: LLM model and object dictionary path.

## Acknowledgements
- [MaskClustering](https://github.com/pku-epic/MaskClustering) for the 3D instance segmentation pipeline.
- [CropFormer](https://github.com/qqlu/Entity) and [OpenCLIP](https://github.com/mlfoundations/open_clip) for 2D mask prediction and feature extraction. 