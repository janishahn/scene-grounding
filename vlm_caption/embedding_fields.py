import os
import numpy as np
from typing import Dict, List
import faiss
from sentence_transformers import SentenceTransformer
from xml.etree import ElementTree as ET
import torch

from vlm_caption.xml_structuring import FIELDS

__all__ = ["build_field_indices"]


def _gather_sentences(xml_path: str) -> Dict[str, List[tuple[int, str]]]:
    """Parse XML and collect texts per field.

    Returns
    -------
    dict
        Mapping field name -> list of (object_id, text).
    """
    out: Dict[str, List[tuple[int, str]]] = {f: [] for f in FIELDS}
    root = ET.parse(xml_path).getroot()
    for obj in root.findall("object"):
        try:
            oid = int(obj.attrib.get("id", "-1").lstrip("obj_"))
        except ValueError:
            continue
        for tag in FIELDS:
            elem = obj.find(tag)
            if elem is not None and elem.text:
                out[tag].append((oid, elem.text.strip()))
    return out


def _make_ivfpq_index(emb: np.ndarray, nlist: int = 256, m: int = 8, nbits: int = 8):
    d = emb.shape[1]
    quantizer = faiss.IndexFlatIP(d)
    index = faiss.IndexIVFPQ(quantizer, d, nlist, m, nbits)
    index.train(emb)
    index.add(emb)
    return index


def build_field_indices(xml_path: str, out_dir: str, model_name: str = "BAAI/bge-base-en-v1.5", batch_size: int = 64,
                         nlist: int = 256, m: int = 8, nbits: int = 8) -> Dict[str, str]:
    """Create one FAISS IVFPQ index per XML field.

    Parameters
    ----------
    xml_path
        Path to the structured XML file.
    out_dir
        Directory to write ``*.faiss`` and ``*.obj_ids.npy``.
    model_name
        Sentence-Transformer model.
    batch_size
        Encoding batch size.
    nlist, m, nbits
        IVFPQ hyper-parameters.

    Returns
    -------
    dict
        Mapping field -> index path.
    """
    os.makedirs(out_dir, exist_ok=True)
    field_pairs = _gather_sentences(xml_path)
    model = SentenceTransformer(model_name, device="cuda" if torch.cuda.is_available() else "cpu")

    paths: Dict[str, str] = {}
    for field, pairs in field_pairs.items():
        if len(pairs) == 0:
            continue
        obj_ids, sentences = zip(*pairs)
        emb = model.encode(sentences, batch_size=batch_size, normalize_embeddings=True, show_progress_bar=False).astype("float32")
        if emb.shape[0] < nlist:
            # too few vectors - fall back to flat index
            index = faiss.IndexFlatIP(emb.shape[1])
            index.add(emb)
        else:
            index = _make_ivfpq_index(emb, nlist, m, nbits)
        seq = os.path.splitext(os.path.basename(xml_path))[0]
        idx_path = os.path.join(out_dir, f"{seq}_{field}.faiss")
        ids_path = os.path.join(out_dir, f"{seq}_{field}.obj_ids.npy")
        faiss.write_index(index, idx_path)
        np.save(ids_path, np.array(obj_ids, dtype=np.int32))
        paths[field] = idx_path
    return paths 