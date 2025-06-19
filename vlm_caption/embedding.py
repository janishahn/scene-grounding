from __future__ import annotations

import os
import json
from typing import Dict, List

import numpy as np
import faiss
from sentence_transformers import SentenceTransformer
import torch


def _load_captions(captions_path: str) -> List[tuple[int, str]]:
    """Return a list of (object_id, combined_caption) tuples sorted by object_id."""
    with open(captions_path, "r") as f:
        raw: Dict[str, Dict] = json.load(f)

    pairs: List[tuple[int, str]] = []
    for k, v in raw.items():
        try:
            obj_id = int(k)
        except ValueError:
            continue
        highlighted = v["captions"].get("highlighted", {}).get("text", "")
        original = v["captions"].get("original", {}).get("text", "")
        combined = (highlighted + " " + original).strip()
        if combined:
            pairs.append((obj_id, combined))
    pairs.sort(key=lambda x: x[0])
    return pairs


def build_faiss_index(
    captions_path: str,
    out_dir: str,
    model_name: str = "BAAI/bge-base-en-v1.5",
    batch_size: int = 64,
) -> tuple[str, str]:
    """Create a FAISS index + id mapping for the given captions file.

    Parameters
    ----------
    captions_path
        Path to the ``*.captions.json`` produced by the captioning pipeline.
    out_dir
        Directory where ``.faiss`` and ``.obj_ids.npy`` will be written.
    model_name
        HuggingFace identifier for the embedder.
    batch_size
        Encoding batch size.

    Returns
    -------
    tuple[str, str]
        Paths to the saved FAISS index and NumPy id mapping.
    """
    os.makedirs(out_dir, exist_ok=True)

    pairs = _load_captions(captions_path)
    if len(pairs) == 0:
        raise ValueError("No captions found for index building.")

    obj_ids, sentences = zip(*pairs)

    model = SentenceTransformer(model_name, device="cuda" if torch.cuda.is_available() else "cpu")
    embeddings = model.encode(
        list(sentences),
        batch_size=batch_size,
        normalize_embeddings=True,
        show_progress_bar=False,
    ).astype("float32")

    d = embeddings.shape[1]
    index = faiss.IndexFlatIP(d)
    index.add(embeddings)

    seq_name = os.path.basename(captions_path).split(".")[0]
    index_path = os.path.join(out_dir, f"{seq_name}.faiss")
    ids_path = os.path.join(out_dir, f"{seq_name}.obj_ids.npy")

    faiss.write_index(index, index_path)
    np.save(ids_path, np.array(obj_ids, dtype=np.int32))

    return index_path, ids_path 