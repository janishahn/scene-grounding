import logging
import argparse
from pathlib import Path
from typing import List, Tuple

import faiss
import numpy as np
import torch
from sentence_transformers import SentenceTransformer
from xml.etree import ElementTree as ET

from vlm_caption.xml_structuring import FIELDS

__all__ = ["build_unified_index"]

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


def _gather_documents(root: ET.Element) -> Tuple[List[int], List[str]]:
    """Return object ids and concatenated documents from parsed XML root."""
    obj_ids: List[int] = []
    docs: List[str] = []
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
        # Build document string using template
        doc = _TEMPLATE.format(**values)
        obj_ids.append(oid)
        docs.append(doc)
    return obj_ids, docs


def build_unified_index(
    xml_path: str | Path,
    out_dir: str | Path | None = None,
    embedder_name: str = "BAAI/bge-base-en-v1.5",
    batch_size: int = 64,
) -> Tuple[Path, Path]:
    """Build and save unified FAISS index for the provided scene XML.

    Parameters
    ----------
    xml_path
        Path to ``<scene>.xml`` produced by the captioning pipeline.
    out_dir
        Directory where ``<scene>.unified.faiss`` and ``<scene>.unified.obj_ids.npy`` are saved.
        Defaults to the directory containing *xml_path*.
    embedder_name
        Name of the SentenceTransformer bi-encoder.
    batch_size
        Encoding batch size.

    Returns
    -------
    (faiss_path, ids_path)
        Paths to the saved files.
    """
    xml_path = Path(xml_path)
    out_dir = Path(out_dir or xml_path.parent)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Parse XML and gather documents
    try:
        root = ET.parse(xml_path).getroot()
    except Exception as e:
        raise RuntimeError(f"Failed to parse XML {xml_path}: {e}") from e

    obj_ids, docs = _gather_documents(root)
    if len(obj_ids) == 0:
        raise ValueError("No objects with valid captions found in XML.")

    # Encode documents
    model = SentenceTransformer(embedder_name, device="cuda" if torch.cuda.is_available() else "cpu")
    embs = model.encode(docs, batch_size=batch_size, normalize_embeddings=True, show_progress_bar=False).astype("float32")

    # Build FAISS index
    dim = embs.shape[1]
    index = faiss.IndexFlatIP(dim)
    index.add(embs)

    scene_name = xml_path.stem
    faiss_path = out_dir / f"{scene_name}.unified.faiss"
    ids_path = out_dir / f"{scene_name}.unified.obj_ids.npy"

    faiss.write_index(index, str(faiss_path))
    np.save(ids_path, np.array(obj_ids, dtype=np.int32))

    logging.info(f"Saved unified index ⇒ {faiss_path} (objects: {len(obj_ids)})")
    return faiss_path, ids_path


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Build unified FAISS index for scene." )
    parser.add_argument("xml", help="Path to <scene>.xml file")
    parser.add_argument("--embedder", default="BAAI/bge-base-en-v1.5", help="SentenceTransformer model name")
    parser.add_argument("--out-dir", default=None, help="Output directory for index files")
    parser.add_argument("--batch-size", type=int, default=64, help="Encoding batch size")
    args = parser.parse_args()

    build_unified_index(args.xml, args.out_dir, args.embedder, args.batch_size) 