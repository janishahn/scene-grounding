import logging
from pathlib import Path
from typing import List, Tuple, Dict

import numpy as np
import torch

from sentence_transformers import SentenceTransformer
import faiss
from xml.etree import ElementTree as ET

from vlm_caption.xml_structuring import FIELDS

__all__ = ["query_scene"]

DEFAULT_EMBEDDER = "BAAI/bge-base-en-v1.5"

def query_scene(
    scene_name: str,
    query: str,
    data_dir: str = "vlm_caption/outputs",
    k: int = 5,
    per_field_k: int = 10,
    score_threshold: float = 0.0,
    embedder_name: str = DEFAULT_EMBEDDER,
) -> Dict:
    """Return top-k matching objects for *query* in *scene_name*.

    Parameters
    ----------
    scene_name
        Scene identifier (prefix used when saving XML & indices).
    query
        Natural-language query.
    data_dir
        Directory containing scene XML and FAISS index files.
    k
        Number of matches to return.
    per_field_k
        Number of candidates to retrieve from every field before aggregation.
    score_threshold
        Similarity threshold; candidates below this value are discarded.
    embedder_name
        Sentence-Transformer model.

    Returns
    -------
    dict
        {"field": <str>, "objects": List[Tuple[object_id, score, info_dict]]}
        {'objects': List[Tuple[object_id, score, field, info_dict]], 'field': <str> (deprecated)}
    """
    data_path = Path(data_dir)
    xml_path = data_path / f"{scene_name}.xml"
    if not xml_path.exists():
        raise FileNotFoundError(xml_path)

    # Load indices
    indices = _load_indices(scene_name, data_path)
    if len(indices) == 0:
        raise FileNotFoundError("No FAISS indices found for the scene.")

    # Embed query
    embedder = SentenceTransformer(embedder_name, device="cuda" if torch.cuda.is_available() else "cpu")
    q_emb = embedder.encode([query], normalize_embeddings=True).astype("float32")

    # Aggregate across all fields
    ids, scores, best_fields = _aggregate_fields(q_emb, indices, k, per_field_k, score_threshold)

    if len(ids) == 0:
        logging.info("All similarities < threshold; returning empty result list.")
        return {"objects": []}

    # Collect info from XML
    root = _load_xml(xml_path)
    data = _gather_info(root, list(ids))

    results = []
    for obj_id, score, fld in zip(ids, scores, best_fields):
        results.append((int(obj_id), float(score), fld, data.get(int(obj_id), {})))

    # For backward compatibility keep the old key but mark as deprecated.
    if results:
        logging.warning("DEPRECATED: 'field' key in return dict will be removed in a future release. "
                        "Consume the 'objects' list instead.")
        deprecated_field = results[0][2]
    else:
        deprecated_field = None

    return {"objects": results, "field": deprecated_field}

def _load_indices(scene_prefix: str, index_dir: Path) -> Dict[str, Tuple[faiss.Index, np.ndarray]]:
    """Load FAISS indices and id maps for all fields present."""
    indices = {}
    for field in FIELDS:
        idx_path = index_dir / f"{scene_prefix}_{field}.faiss"
        ids_path = index_dir / f"{scene_prefix}_{field}.obj_ids.npy"
        if idx_path.exists() and ids_path.exists():
            try:
                idx = faiss.read_index(str(idx_path))
                ids = np.load(ids_path)
                indices[field] = (idx, ids)
            except Exception as e:
                logging.warning(f"Failed to load index for field '{field}': {e}")
    return indices

def _aggregate_fields(
    query_emb: np.ndarray,
    indices: Dict[str, Tuple[faiss.Index, np.ndarray]],
    top_k: int,
    per_field_k: int,
    score_threshold: float,
) -> Tuple[np.ndarray, np.ndarray, List[str]]:
    """Search all fields and aggregate best-scoring unique objects.

    Parameters
    ----------
    query_emb
        Normalised query embedding, shape (1, dim).
    indices
        Mapping ``field -> (faiss_index, id_map)``.
    top_k
        Number of objects to return after aggregation.
    per_field_k
        Candidates to fetch from each field-specific index.
    score_threshold
        Discard hits whose similarity is below this value.

    Returns
    -------
    ids, scores, best_fields
        Arrays/lists aligned such that *ids[i]* has similarity *scores[i]* and came
        from *best_fields[i]*.
    """
    best_per_obj: Dict[int, Tuple[float, str]] = {}

    for field, (index, id_map) in indices.items():
        try:
            scores, I = index.search(query_emb, per_field_k)
            for s, idx in zip(scores[0], I[0]):
                if s < score_threshold:
                    continue
                oid = int(id_map[idx])
                if oid not in best_per_obj or s > best_per_obj[oid][0]:
                    best_per_obj[oid] = (float(s), field)
        except Exception as e:
            logging.warning(f"Search failed for field '{field}': {e}")

    if not best_per_obj:
        # No candidates survived filtering.
        return np.array([]), np.array([]), []

    # Sort globally by score descending and slice top_k
    sorted_items = sorted(best_per_obj.items(), key=lambda it: it[1][0], reverse=True)[:top_k]
    ids = np.array([item[0] for item in sorted_items])
    scores = np.array([item[1][0] for item in sorted_items], dtype="float32")
    fields = [item[1][1] for item in sorted_items]
    return ids, scores, fields

def _load_xml(scene_xml: Path) -> ET.Element:
    return ET.parse(scene_xml).getroot()

def _gather_info(root: ET.Element, object_ids: List[int]) -> Dict[int, Dict[str, str]]:
    info = {oid: {} for oid in object_ids}
    for obj in root.findall("object"):
        obj_id_attr = obj.attrib.get("id", "")
        try:
            oid = int(obj_id_attr.lstrip("obj_"))
        except ValueError:
            continue
        if oid in info:
            for tag in FIELDS:
                elem = obj.find(tag)
                if elem is not None and elem.text:
                    info[oid][tag] = elem.text.strip()
    return info
