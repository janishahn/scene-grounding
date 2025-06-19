import json
import logging
import yaml
import re
from pathlib import Path

import numpy as np
import torch

from torch import load

from sentence_transformers import SentenceTransformer
import faiss

def query_scene(captions_path: str):
    """
    Query a scene using captions and natural language input to identify a specific object.
    This function processes scene captions, takes a user query, and uses an LLM to identify 
    the most relevant object in the scene. It then returns the path to an image highlighting
    that object.

    Args:
        captions_path (str): Path to the JSON file containing object captions.
    
    Returns:
        str or None: Path to the highlighted image of the identified object, or None if
                     no object could be identified from the LLM response.
    """
    
    logging.info(f"Querying scene with captions from {captions_path}...")
    # Get the configs from query.yaml
    with open("llm_query/query.yaml", 'r') as f:
        query_config = yaml.safe_load(f)

    model_name = query_config.get("model", "")

    # Load captions and get the query from the user
    with open(captions_path, 'r') as f:
        captions = json.load(f)
    # Reduce captions to only the cropped version
    lean_captions = {}
    for id, val in captions.items():
        lean_captions[id] = val['captions']['highlighted']['text']

    # Optional vector retrieval to shrink candidate set
    seq_name = Path(captions_path).name.split(".")[0]
    retrieval_cfg = query_config.get("retrieval", {})
    index_dir = retrieval_cfg.get("faiss_index_dir", Path(captions_path).parent)
    index_path = Path(index_dir) / f"{seq_name}.faiss"
    ids_path = Path(index_dir) / f"{seq_name}.obj_ids.npy"
    top_k = int(retrieval_cfg.get("top_k", 10))
    embedder_name = retrieval_cfg.get("embedder", "BAAI/bge-base-en-v1.5")

    if index_path.exists() and ids_path.exists():
        try:
            index = faiss.read_index(str(index_path))
            obj_ids = np.load(ids_path)
            embedder = SentenceTransformer(embedder_name, device="cuda" if torch.cuda.is_available() else "cpu")
            # Get user query first
            query = input("Please enter your query here: ")
            q_emb = embedder.encode([query], normalize_embeddings=True).astype("float32")
            _, I = index.search(q_emb, top_k)
            selected_ids = {str(obj_ids[i]) for i in I[0]}
            lean_captions = {k: v for k, v in lean_captions.items() if k in selected_ids}
        except Exception as e:
            logging.warning(f"Retrieval failed, falling back to full captions: {e}")
            query = input("Please enter your query here: ")
    else:
        logging.info("FAISS index not found, using all captions")
        query = input("Please enter your query here: ")

    # Print size of captions dictionary in bytes (after any filtering)
    print(f"Captions dictionary size: {len(json.dumps(lean_captions))}")

    # Build prompt
    system_prompt = build_system_prompt(lean_captions)
    user_content = (
        "Available objects:\n" + json.dumps(lean_captions, indent=2) +
        f"\n\nUser query: {query}"
    )
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_content},
    ]
    
    logging.debug(
        f"Prompt sizes (chars) – system: {len(system_prompt)}, user payload: {len(user_content)}"
    )

    # Call LLM with chat endpoint
    logging.info("Querying LLM (chat) to find best object description for query...")
    from ollama import chat
    response = chat(
        model=model_name,
        messages=messages,
        options={"num_ctx": 8192},
    )

    in_tok = response.get("prompt_eval_count")
    out_tok = response.get("eval_count")
    if in_tok is not None and out_tok is not None:
        logging.info(
            f"LLM token usage - input: {in_tok} tokens, output: {out_tok} tokens, total: {in_tok + out_tok}"
        )
    else:
        logging.debug("Token usage metadata not found in LLM response")

    resp_text = response["message"]["content"].strip()
    logging.info("The LLM has returned the following response:")
    logging.info(resp_text)

    # Extract <object_id>...</object_id> from XML-like response
    id_match = re.search(r"<object_id>(\d+)</object_id>", resp_text)
    if not id_match:
        logging.error("No <object_id> tag found in the LLM response.")
        return None
    object_id = id_match.group(1)

    # Resolve image path from object dictionary
    obj_dict_path = query_config.get('obj_dict_path', "")
    obj_dict = load(obj_dict_path, weights_only=False)
    try:
        img_path = obj_dict[int(object_id)]['best_view']['highlighted_path']
    except Exception as e:
        logging.error(f"Could not retrieve image path for object {object_id}: {e}")
        return None
    
    logging.info("Successfully identified object, returning image path.")
    return img_path

def build_system_prompt(captions: dict) -> str:
    with open("llm_query/query_prompt.md", "r") as f:
        base_prompt = f.read().strip()
    return base_prompt