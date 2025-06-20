import argparse
import json
from pathlib import Path

import faiss
import numpy as np
import torch
from sentence_transformers import SentenceTransformer


def build_index(
    captions_path: str | Path,
    embedder_name: str = "BAAI/bge-base-en-v1.5",
    caption_key: str = "combined",
    out_dir: str | Path | None = None,
) -> None:
    """Create a FAISS index (inner-product) over caption embeddings.

    Parameters
    ----------
    captions_path : str | Path
        JSON file produced by the pipeline containing captions per object.
    embedder_name : str, optional
        SentenceTransformer model to embed text, by default ``BAAI/bge-base-en-v1.5``.
    caption_key : str, optional
        Which caption variant to embed.  Typical values: ``combined`` or ``highlighted``.
    out_dir : str | Path | None, optional
        Where to save ``<seq>.faiss`` and ``<seq>.obj_ids.npy``.  Defaults to the
        directory of *captions_path*.
    """

    captions_path = Path(captions_path)
    out_dir = Path(out_dir or captions_path.parent)
    out_dir.mkdir(parents=True, exist_ok=True)

    with captions_path.open() as f:
        captions = json.load(f)

    texts: list[str] = []
    obj_ids: list[int] = []
    for obj_id, blob in captions.items():
        block = blob.get("captions", {})
        if caption_key == "highlighted":
            txt = block.get("highlighted", {}).get("text", "")
        elif caption_key == "combined":
            txt = block.get("combined", {}).get("text", "")
        else:
            txt = block.get(caption_key, "")
        if txt:
            texts.append(txt)
            obj_ids.append(int(obj_id))

    embedder = SentenceTransformer(embedder_name, device="cuda" if torch.cuda.is_available() else "cpu")
    embs = embedder.encode(texts, normalize_embeddings=True, show_progress_bar=True).astype("float32")

    index = faiss.IndexFlatIP(embs.shape[1])
    index.add(embs)

    seq_name = captions_path.stem
    faiss_path = out_dir / f"{seq_name}.faiss"
    ids_path = out_dir / f"{seq_name}.obj_ids.npy"

    faiss.write_index(index, str(faiss_path))
    np.save(ids_path, np.array(obj_ids))

    print(f"Saved index to {faiss_path} and ids to {ids_path} (objects: {len(obj_ids)})")


if __name__ == "__main__":
    p = argparse.ArgumentParser(description="Build FAISS index for scene captions")
    p.add_argument("captions", help="Path to captions JSON file")
    p.add_argument("--embedder", default="BAAI/bge-base-en-v1.5")
    p.add_argument("--caption-key", default="combined", choices=["combined", "highlighted"])
    p.add_argument("--out-dir", default=None, help="Directory to store index files")
    args = p.parse_args()
    build_index(args.captions, args.embedder, args.caption_key, args.out_dir) 