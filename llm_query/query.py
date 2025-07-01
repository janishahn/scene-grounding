import logging
from pathlib import Path
from typing import List, Tuple, Dict

import numpy as np
import torch

from sentence_transformers import SentenceTransformer, CrossEncoder
import faiss
from xml.etree import ElementTree as ET

from vlm_caption.xml_structuring import FIELDS

# Public symbols
__all__ = ["query_scene"]

# -----------------------------------------------------------------------------
# Two-stage retrieve & rerank parameters
# -----------------------------------------------------------------------------
DEFAULT_EMBEDDER = "BAAI/bge-base-en-v1.5"
DEFAULT_CROSS_ENCODER = "cross-encoder/ms-marco-MiniLM-L-6-v2"

# Template MUST match the one used during indexing
_TEMPLATE = (
    "Name: {name}. "
    "Role: {role}. "
    "Purpose: {purpose}. "
    "Details: {details}. "
    "Shape: {shape}. "
    "Materials: {materials}. "
    "Color: {color}. "
    "Location: {spatial}."
)


def _build_documents(root: ET.Element) -> Dict[int, str]:
    """Return mapping oid -> concatenated document (same logic as indexing)."""
    docs: Dict[int, str] = {}
    for obj in root.findall("object"):
        id_attr = obj.attrib.get("id", "")
        try:
            oid = int(id_attr.lstrip("obj_"))
        except ValueError:
            continue
        values = {tag: "" for tag in FIELDS}
        for tag in FIELDS:
            elem = obj.find(tag)
            if elem is not None and elem.text:
                values[tag] = elem.text.strip()
        docs[oid] = _TEMPLATE.format(**values)
    return docs


# -----------------------------------------------------------------------------
# Main public API
# -----------------------------------------------------------------------------

def query_scene(
    scene_name: str,
    query: str,
    data_dir: str = "vlm_caption/outputs",
    k: int = 5,
    per_field_k: int = 10,
    k_retrieval: int = 50,
    score_threshold: float = 0.0,
    embedder_name: str = DEFAULT_EMBEDDER,
    cross_encoder_name: str = DEFAULT_CROSS_ENCODER,
    ce_only: bool = False,
) -> Dict:
    """Return top-k objects ranked by cross-encoder similarity.

    The function operates in two stages:
    1. Fast retrieval from the unified scene index (recall-oriented).
    2. Precise reranking with a cross-encoder (precision-oriented).
    """

    data_path = Path(data_dir)
    xml_path = data_path / f"{scene_name}.xml"
    if not xml_path.exists():
        raise FileNotFoundError(xml_path)

    # Parse XML and build per-object documents (needed for any mode)
    root = ET.parse(xml_path).getroot()
    docs_map = _build_documents(root)

    device = "cuda" if torch.cuda.is_available() else "cpu"

    # If we operate in CE-only mode, collect *all* objects now
    if ce_only:
        kept_oids = list(docs_map.keys())
        pair_texts: List[Tuple[str, str]] = [(query, docs_map[oid]) for oid in kept_oids]
    else:
        # ------------------------------------------------------------------
        # Load unified FAISS index
        # ------------------------------------------------------------------
        idx_path = data_path / f"{scene_name}.unified.faiss"
        ids_path = data_path / f"{scene_name}.unified.obj_ids.npy"

        if not idx_path.exists() or not ids_path.exists():
            raise FileNotFoundError("Unified FAISS index not found – please build it first.")

        index = faiss.read_index(str(idx_path))
        obj_ids_map = np.load(ids_path)

        # ------------------------------------------------------------------
        # Embed query & retrieve candidates
        # ------------------------------------------------------------------
        embedder = SentenceTransformer(embedder_name, device=device)
        q_emb = embedder.encode([query], normalize_embeddings=True).astype("float32")

        # Retrieve top k_retrieval candidate vectors
        scores, faiss_ids = index.search(q_emb, k_retrieval)
        scores = scores[0]
        faiss_ids = faiss_ids[0]

        # Filter by threshold and map to object ids
        candidates: List[Tuple[int, float]] = []
        for s, idx in zip(scores, faiss_ids):
            if idx < 0:
                continue
            if s < score_threshold:
                continue
            oid = int(obj_ids_map[idx])
            candidates.append((oid, float(s)))

        if not candidates:
            logging.info("No candidates above similarity threshold; returning empty list.")
            return {"objects": []}

        # ------------------------------------------------------------------
        # Prepare documents for reranking
        # ------------------------------------------------------------------
        pair_texts: List[Tuple[str, str]] = []
        kept_oids: List[int] = []
        for oid, _ in candidates:
            doc = docs_map.get(oid)
            if doc:
                pair_texts.append((query, doc))
                kept_oids.append(oid)

        if not pair_texts:
            logging.warning("No documents found for retrieved candidate objects.")
            return {"objects": []}

    # ------------------------------------------------------------------
    # Rerank with cross-encoder
    # ------------------------------------------------------------------
    logging.info(f"Loading CrossEncoder model: {cross_encoder_name}")
    cross_encoder = CrossEncoder(cross_encoder_name, device=device)
    ce_scores = cross_encoder.predict(pair_texts)
    logging.info("Cross-encoder reranking completed (%d candidates).", len(pair_texts))

    # Sort by cross-encoder score desc & slice top-k
    sorted_idx = np.argsort(-ce_scores)[:k]
    final_oids = [kept_oids[i] for i in sorted_idx]
    final_scores = ce_scores[sorted_idx]

    # Gather full XML info for UI
    info_map = _gather_info(root, final_oids)

    # Attach the concatenated document so the GUI can display a reason/snippet
    for oid in final_oids:
        doc_txt = docs_map.get(oid, "")
        if oid in info_map:
            info_map[oid]["unified"] = doc_txt
        else:
            info_map[oid] = {"unified": doc_txt}

    results = [
        (int(oid), float(score), "unified", info_map.get(int(oid), {}))
        for oid, score in zip(final_oids, final_scores)
    ]

    return {"objects": results}


# -----------------------------------------------------------------------------
# Helpers (unchanged from previous implementation)
# -----------------------------------------------------------------------------

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
