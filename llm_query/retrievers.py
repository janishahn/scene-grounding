from __future__ import annotations

import logging
import re
import time
from abc import ABC, abstractmethod
from enum import Enum
from pathlib import Path
from typing import Dict, List, Tuple, Optional

import ollama
import tiktoken
import os
import requests
from xml.etree import ElementTree as ET

from vlm_caption.xml_structuring import FIELDS

# Heavyweight deps only needed for embedding strategy
import numpy as np
import torch
from sentence_transformers import SentenceTransformer, CrossEncoder
import faiss

__all__ = [
    "RetrievalStrategy",
    "Retriever",
    "OllamaLLMRetriever",
    "OpenRouterLLMRetriever",
    "EmbeddingRetriever",
]


class RetrievalStrategy(str, Enum):
    """Enumeration of supported retrieval/ranking back-ends."""

    EMBEDDING = "embedding"
    CE_ONLY = "ce_only"
    LLM_OLLAMA = "llm_ollama"
    LLM_OPENROUTER = "llm_openrouter"


class Retriever(ABC):
    """Abstract base class for any retrieval / ranking model."""

    @abstractmethod
    def rank(self, scene_name: str, query: str, k: int = 5, **kwargs) -> Dict:
        """Return a JSON-serialisable dictionary with ranked objects."""
        raise NotImplementedError


class OllamaLLMRetriever(Retriever):
    """Rank objects with a local LLM served by Ollama.

    The model receives *all* object XML snippets at once and is asked to output
    only the object identifier that best matches the user query.
    """

    _HARD_MAX_CTX = 50000

    def __init__(
        self,
        *,
        model_name: str = "gemma3:4b-it-qat",
        ollama_host: Optional[str] = None,
        temperature: float = 0.7,
        xml_token_buffer: int = 5000,
    ) -> None:
        self.model_name = model_name
        self.ollama_host = ollama_host  # None == default http://localhost:11434
        self.temperature = temperature
        self.xml_token_buffer = xml_token_buffer

        # Pre-load tiktoken encoder for Gemma once.
        try:
            self._enc = tiktoken.encoding_for_model("gemma-3b-it-qat")
        except KeyError:
            # Fallback to generic encoder if specific one unavailable.
            self._enc = tiktoken.get_encoding("cl100k_base")

    # ---------------------------------------------------------------------
    # Public API
    # ---------------------------------------------------------------------

    def rank(self, scene_name: str, query: str, k: int = 5, **kwargs) -> Dict:
        """Return up to *k* (max 5) best-matching objects with confidences."""
        start_t = time.time()

        scene_xml = self._load_scene_xml(scene_name, kwargs.get("data_dir", "vlm_caption/outputs"))
        objects_str, obj_id_ints = self._prepare_objects_section(scene_xml)

        # ------------------------------------------------------------------
        # Build prompt & token budget
        # ------------------------------------------------------------------
        xml_tok_len = len(self._enc.encode(objects_str))
        max_ctx = min(xml_tok_len + self.xml_token_buffer, self._HARD_MAX_CTX)
        logging.info(f"XML tokens: {xml_tok_len}, max context: {max_ctx}")

        prompt = self._build_prompt(objects_str, query)

        # ------------------------------------------------------------------
        # Ollama request with retries
        # ------------------------------------------------------------------
        attempt, resp_content = 0, None
        while attempt < 3:
            attempt += 1
            try:
                if self.ollama_host:
                    _client = ollama.Client(host=self.ollama_host)
                    resp = _client.chat(
                        model=self.model_name,
                        messages=[{"role": "user", "content": prompt}],
                        options={"num_ctx": max_ctx, "temperature": self.temperature},
                    )
                else:
                    resp = ollama.chat(
                        model=self.model_name,
                        messages=[{"role": "user", "content": prompt}],
                        options={"num_ctx": max_ctx, "temperature": self.temperature},
                    )
                resp_content = self._extract_content(resp)
                if resp_content:
                    break
            except Exception as e:
                logging.warning(
                    "Ollama request failed on attempt %d/3: %s", attempt, e
                )
                time.sleep(2 ** attempt)

        if resp_content is None:
            logging.error("All Ollama requests failed - returning empty result.")
            return {"objects": []}

        # ------------------------------------------------------------------
        # Parse reasoning & list of returned objects with confidences
        # ------------------------------------------------------------------
        reasoning_text: str = ""

        # 1) Extract reasoning
        m_reason = re.search(r"<reasoning>(.*?)</reasoning>", resp_content, re.IGNORECASE | re.DOTALL)
        if m_reason:
            reasoning_text = m_reason.group(1).strip()

        # 2) Extract object entries of form <object id="object_XX" confidence="YY" />
        obj_matches: List[Tuple[str, float]] = []
        for m in re.finditer(r"<object[^>]*?>", resp_content, re.IGNORECASE):
            tag = m.group(0)
            id_m = re.search(r"(?:object_?id|id)=\"([^\"]+)\"", tag, re.IGNORECASE)
            conf_m = re.search(r"confidence=\"([^\"]+)\"", tag, re.IGNORECASE)
            if id_m and conf_m:
                id_str = id_m.group(1).strip().strip('"').strip("'")
                try:
                    conf_val = float(conf_m.group(1))
                except ValueError:
                    continue
                obj_matches.append((id_str, conf_val))



        # ------------------------------------------------------------------
        # Validate & convert IDs
        # ------------------------------------------------------------------
        valid_objects: List[Tuple[int, float]] = []
        for id_str, conf in obj_matches:
            try:
                if id_str.startswith("object_"):
                    oid = int(id_str.lstrip("object_"))
                else:
                    oid = int(id_str.lstrip("obj_"))
            except ValueError:
                logging.warning("Skipping unparsable object id '%s'", id_str)
                continue
            if oid not in obj_id_ints:
                logging.warning("Skipping object id '%s' not present in scene", id_str)
                continue
            # Clip confidence to [0,100]
            conf_clamped = max(0.0, min(conf, 100.0))
            valid_objects.append((oid, conf_clamped))

        if not valid_objects:
            logging.error("LLM returned no valid object ids - falling back to CE-only.")
            return self._fallback_ce(scene_name, query, k, kwargs.get("data_dir", "vlm_caption/outputs"))

        # Sort by confidence descending and keep top-k requested
        valid_objects.sort(key=lambda x: -x[1])
        final_objects = valid_objects[:k]

        oids = [oid for oid, _ in final_objects]
        info_map = self._gather_info(scene_xml, oids)
        for oid in oids:
            # Attach common metadata for frontend display.
            info_map[int(oid)]["model_name"] = self.model_name
            if reasoning_text:
                info_map[int(oid)]["reasoning"] = reasoning_text

        elapsed = time.time() - start_t
        logging.info(
            "Ollama ranking finished - XML tokens=%d, round-trip=%.2fs", xml_tok_len, elapsed
        )

        return {
            "objects": [
                (
                    int(oid),
                    conf,
                    "llm_ollama",
                    info_map.get(int(oid), {}),
                )
                for oid, conf in final_objects
            ]
        }

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    @staticmethod
    def _extract_content(resp) -> str | None:
        """Robustly obtain the textual content from diverse Ollama response variants."""
        if resp is None:
            return None
        if isinstance(resp, str):
            return resp
        # Mapping/dict response
        if isinstance(resp, dict):
            # API has changed a few times; try common keys.
            for path in [
                ("message", "content"),
                ("response",),
                ("content",),
            ]:
                node = resp
                for key in path:
                    node = node.get(key) if isinstance(node, dict) else None
                if isinstance(node, str):
                    return node
            return None
        # Fallback for dataclass / object responses coming from the official
        # Ollama client (attributes instead of dict keys).
        if hasattr(resp, "message") and resp.message is not None:
            # `resp.message` may itself be a dict or an object.
            msg = resp.message
            if isinstance(msg, dict):
                return msg.get("content")
            if hasattr(msg, "content"):
                return msg.content

        for attr in ("response", "content"):
            if hasattr(resp, attr):
                val = getattr(resp, attr)
                if isinstance(val, str):
                    return val
        # Ultimate fallback: string representation (may include noise).
        return str(resp)

    @staticmethod
    def _load_scene_xml(scene_name: str, data_dir: str) -> ET.Element:
        data_path = Path(data_dir)
        xml_path = data_path / f"{scene_name}.xml"
        if not xml_path.exists():
            raise FileNotFoundError(xml_path)
        return ET.parse(xml_path).getroot()

    @staticmethod
    def _prepare_objects_section(root: ET.Element) -> Tuple[str, List[int]]:
        sections: List[str] = []
        obj_int_ids: List[int] = []
        for obj in root.findall("object"):
            obj_id = obj.attrib.get("id", "").strip()
            if not obj_id:
                continue
            obj_int_ids.append(int(obj_id.lstrip("obj_")))
            xml_str = ET.tostring(obj, encoding="unicode", method="xml").strip()
            sections.append(f"{obj_id}\n{xml_str}")
        return "\n\n".join(sections), obj_int_ids

    @staticmethod
    def _build_prompt(objects_section: str, user_query: str) -> str:
        """Return an instruction prompt that asks the LLM to reason and then output *up to five* matching
        objects together with a confidence score (0-100).

        Required response format (XML-like, **do not** add markdown fencing):

        <reasoning>YOUR DETAILED CHAIN-OF-THOUGHT</reasoning>
        <objects>
          <object id="object_12" confidence="92" />
          <object id="object_05" confidence="67" />
          ... up to 5 entries ...
        </objects>

        • The <objects> block may be empty if none of the scene objects match the USER QUERY well enough.
        • Confidence must be an integer between 0 and 100 (higher = better match).
        """

        return (
            "SYSTEM: You are an expert assistant whose task is to choose up to five objects from a 3-D scene that best match the USER QUERY.\n"
            "OBJECTS:\n"
            f"{objects_section}\n"
            f"USER QUERY: {user_query}\n"
            "INSTRUCTIONS:\n"
            "1. Think step-by-step which objects satisfy the USER QUERY.\n"
            "2. Assign each promising object a confidence score (0-100).\n"
            "3. Return the *top five* objects sorted by confidence (highest first). If no object is suitable, return an empty <objects> block.\n\n"
            "Respond **exactly** in the following XML-style without extra commentary or markdown fencing:\n"
            "<reasoning>YOUR DETAILED CHAIN-OF-THOUGHT</reasoning>\n"
            "<objects>\n"
            "  <object id=\"object_XX\" confidence=\"YY\" />\n"
            "  ... up to five such lines ...\n"
            "</objects>"
        )

    @staticmethod
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

    # ------------------------------------------------------------------
    # Fallback - CE-only rerank if LLM fails.
    # ------------------------------------------------------------------

    @staticmethod
    def _fallback_ce(scene_name: str, query: str, k: int, data_dir: str) -> Dict:
        """Fallback to cross-encoder-only ranking in case the LLM result is unusable."""
        logging.info("Falling back to cross-encoder-only reranking.")
        from llm_query.query import query_scene as _query_scene  # Local import to avoid cycle

        return _query_scene(
            scene_name,
            query,
            data_dir=data_dir,
            k=k,
            ce_only=True,
            retrieval_strategy="embedding",
        )


# =============================================================================
# Embedding-based two-stage retriever (bi-encoder + cross-encoder)
# =============================================================================


class EmbeddingRetriever(Retriever):
    """Bi-encoder FAISS retrieval followed by cross-encoder reranking."""

    def __init__(
        self,
        *,
        embedder_name: str = "BAAI/bge-base-en-v1.5",
        cross_encoder_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2",
        per_field_k: int = 10,  # legacy, unused in unified index mode
        k_retrieval: int = 50,
        score_threshold: float = 0.0,
        ce_only: bool = False,
    ) -> None:
        self.embedder_name = embedder_name
        self.cross_encoder_name = cross_encoder_name
        self.per_field_k = per_field_k
        self.k_retrieval = k_retrieval
        self.score_threshold = score_threshold
        self.ce_only = ce_only

        # Device setup once.
        self._device = "cuda" if torch.cuda.is_available() else "cpu"

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def rank(self, scene_name: str, query: str, k: int = 5, **kwargs) -> Dict:
        data_dir: str = kwargs.get("data_dir", "vlm_caption/outputs")

        data_path = Path(data_dir)
        xml_path = data_path / f"{scene_name}.xml"
        if not xml_path.exists():
            raise FileNotFoundError(xml_path)

        # Parse XML & build documents once
        root = ET.parse(xml_path).getroot()
        docs_map = self._build_documents(root)

        # --------------------------------------------------------------
        # Prepare candidate list (FAISS or all objects)
        # --------------------------------------------------------------
        if self.ce_only:
            kept_oids = list(docs_map.keys())
            pair_texts: List[Tuple[str, str]] = [(query, docs_map[oid]) for oid in kept_oids]
        else:
            idx_path = data_path / f"{scene_name}.unified.faiss"
            ids_path = data_path / f"{scene_name}.unified.obj_ids.npy"

            if not idx_path.exists() or not ids_path.exists():
                raise FileNotFoundError("Unified FAISS index not found – please build it first.")

            index = faiss.read_index(str(idx_path))
            obj_ids_map = np.load(ids_path)

            # Embed query
            embedder = SentenceTransformer(self.embedder_name, device=self._device)
            q_emb = embedder.encode([query], normalize_embeddings=True).astype("float32")

            scores, faiss_ids = index.search(q_emb, self.k_retrieval)
            scores = scores[0]
            faiss_ids = faiss_ids[0]

            candidates: List[Tuple[int, float]] = []
            for s, idx in zip(scores, faiss_ids):
                if idx < 0:
                    continue
                if s < self.score_threshold:
                    continue
                oid = int(obj_ids_map[idx])
                candidates.append((oid, float(s)))

            if not candidates:
                logging.info("No candidates above similarity threshold; returning empty list.")
                return {"objects": []}

            pair_texts = []
            kept_oids = []
            for oid, _ in candidates:
                doc = docs_map.get(oid)
                if doc:
                    pair_texts.append((query, doc))
                    kept_oids.append(oid)

            if not pair_texts:
                logging.warning("No documents found for retrieved candidate objects.")
                return {"objects": []}

        # --------------------------------------------------------------
        # Cross-encoder reranking
        # --------------------------------------------------------------
        cross_encoder = CrossEncoder(self.cross_encoder_name, device=self._device)
        ce_scores = cross_encoder.predict(pair_texts)

        sorted_idx = np.argsort(-ce_scores)[:k]
        final_oids = [kept_oids[i] for i in sorted_idx]
        final_scores = ce_scores[sorted_idx]

        info_map = self._gather_info(root, final_oids)

        results = [
            (int(oid), float(score), "unified", info_map.get(int(oid), {}))
            for oid, score in zip(final_oids, final_scores)
        ]

        return {"objects": results}

    # ------------------------------------------------------------------
    # Helpers (adapted from previous query.py)
    # ------------------------------------------------------------------

    @staticmethod
    def _build_documents(root: ET.Element) -> Dict[int, str]:
        template = (
            "Name: {name}. "
            "Role: {role}. "
            "Purpose: {purpose}. "
            "Details: {details}. "
            "Shape: {shape}. "
            "Materials: {materials}. "
            "Color: {color}. "
            "Location: {spatial}."
        )

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
            docs[oid] = template.format(**values)
        return docs

    @staticmethod
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


class OpenRouterLLMRetriever(Retriever):
    """Rank objects using an LLM served by OpenRouter.

    The logic mirrors :class:`OllamaLLMRetriever` but sends the request to
    the OpenRouter Chat Completions API instead of a local Ollama server.
    """

    def __init__(
        self,
        *,
        model_name: str = "mistralai/mistral-small-3.2-24b-instruct:free",
        api_key: Optional[str] = None,
        temperature: float = 0.7,
        xml_token_buffer: int = 5000,
    ) -> None:
        self.model_name = model_name
        self.api_key = api_key or os.getenv("OPENROUTER_API_KEY")
        if not self.api_key:
            raise ValueError("OpenRouter API key must be provided via argument or OPENROUTER_API_KEY env var.")
        self.temperature = temperature
        self.xml_token_buffer = xml_token_buffer

        # Token encoder for rough budgeting (fallback to cl100k if model unknown)
        try:
            self._enc = tiktoken.encoding_for_model("gpt-3.5-turbo")
        except KeyError:
            self._enc = tiktoken.get_encoding("cl100k_base")

    # ---------------------------------------------------------------------
    # Public API
    # ---------------------------------------------------------------------

    def rank(self, scene_name: str, query: str, k: int = 5, **kwargs) -> Dict:
        """Return up to *k* (max 5) best-matching objects with confidences."""
        start_t = time.time()

        scene_xml = OllamaLLMRetriever._load_scene_xml(
            scene_name, kwargs.get("data_dir", "vlm_caption/outputs")
        )
        objects_str, obj_id_ints = OllamaLLMRetriever._prepare_objects_section(scene_xml)

        # --------------------------------------------------------------
        # Build prompt & token budget
        # --------------------------------------------------------------
        xml_tok_len = len(self._enc.encode(objects_str))

        prompt = OllamaLLMRetriever._build_prompt(objects_str, query)

        # --------------------------------------------------------------
        # OpenRouter request with retries
        # --------------------------------------------------------------
        url = "https://openrouter.ai/api/v1/chat/completions"
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }
        payload = {
            "model": self.model_name,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": self.temperature,
        }

        attempt, resp_content = 0, None
        while attempt < 3:
            attempt += 1
            try:
                logging.info(f"OpenRouter request started")
                resp = requests.post(url, headers=headers, json=payload, timeout=60)
                resp.raise_for_status()
                resp_json = resp.json()
                resp_content = self._extract_content(resp_json)
                if resp_content:
                    break
            except Exception as e:
                logging.warning("OpenRouter request failed on attempt %d/3: %s", attempt, e)
                time.sleep(2 ** attempt)

        if resp_content is None:
            logging.error("All OpenRouter requests failed - returning empty result.")
            return {
                "objects": []
            }

        # --------------------------------------------------------------
        # Parse reasoning & list of returned objects with confidences
        # --------------------------------------------------------------
        reasoning_text: str = ""

        # 1) Extract reasoning
        m_reason = re.search(r"<reasoning>(.*?)</reasoning>", resp_content, re.IGNORECASE | re.DOTALL)
        if m_reason:
            reasoning_text = m_reason.group(1).strip()

        # 2) Extract object entries of form <object id="object_XX" confidence="YY" />
        obj_matches: List[Tuple[str, float]] = []
        for m in re.finditer(r"<object[^>]*?>", resp_content, re.IGNORECASE):
            tag = m.group(0)
            id_m = re.search(r"(?:object_?id|id)=\"([^\"]+)\"", tag, re.IGNORECASE)
            conf_m = re.search(r"confidence=\"([^\"]+)\"", tag, re.IGNORECASE)
            if id_m and conf_m:
                id_str = id_m.group(1).strip().strip('"').strip("'")
                try:
                    conf_val = float(conf_m.group(1))
                except ValueError:
                    continue
                obj_matches.append((id_str, conf_val))


        # ------------------------------------------------------------------
        # Validate & convert IDs
        # ------------------------------------------------------------------
        valid_objects: List[Tuple[int, float]] = []
        for id_str, conf in obj_matches:
            try:
                if id_str.startswith("object_"):
                    oid = int(id_str.lstrip("object_"))
                else:
                    oid = int(id_str.lstrip("obj_"))
            except ValueError:
                logging.warning("Skipping unparsable object id '%s'", id_str)
                continue
            if oid not in obj_id_ints:
                logging.warning("Skipping object id '%s' not present in scene", id_str)
                continue
            # Clip confidence to [0,100]
            conf_clamped = max(0.0, min(conf, 100.0))
            valid_objects.append((oid, conf_clamped))

        if not valid_objects:
            logging.error("LLM returned no valid object ids - falling back to CE-only.")
            return OllamaLLMRetriever._fallback_ce(
                scene_name, query, k, kwargs.get("data_dir", "vlm_caption/outputs")
            )

        # Sort by confidence descending and keep top-k requested
        valid_objects.sort(key=lambda x: -x[1])
        final_objects = valid_objects[:k]

        oids = [oid for oid, _ in final_objects]
        info_map = OllamaLLMRetriever._gather_info(scene_xml, oids)
        for oid in oids:
            # Attach common metadata for frontend display.
            info_map[int(oid)]["model_name"] = self.model_name
            if reasoning_text:
                info_map[int(oid)]["reasoning"] = reasoning_text

        elapsed = time.time() - start_t
        logging.info("OpenRouter ranking finished - XML tokens=%d, round-trip=%.2fs", xml_tok_len, elapsed)

        return {
            "objects": [
                (
                    int(oid),
                    conf,
                    "llm_openrouter",
                    info_map.get(int(oid), {}),
                )
                for oid, conf in final_objects
            ]
        }

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    @staticmethod
    def _extract_content(resp) -> str | None:
        """Extract the assistant message content from OpenRouter response."""
        if resp is None:
            return None
        if isinstance(resp, str):
            return resp
        if isinstance(resp, dict):
            # Standard OpenAI-style schema
            if "choices" in resp and resp["choices"]:
                choice = resp["choices"][0]
                if isinstance(choice, dict):
                    # OpenRouter may wrap message inside 'message', but some models may use 'delta'
                    msg = choice.get("message") or choice.get("delta") or {}
                    if isinstance(msg, dict):
                        content = msg.get("content")
                        if isinstance(content, str):
                            return content
            # Fallback to Ollama's extraction paths for safety
            return OllamaLLMRetriever._extract_content(resp)
        # Fallback to str representation if all else fails
        return str(resp)

    @staticmethod
    def _load_scene_xml(scene_name: str, data_dir: str) -> ET.Element:
        data_path = Path(data_dir)
        xml_path = data_path / f"{scene_name}.xml"
        if not xml_path.exists():
            raise FileNotFoundError(xml_path)
        return ET.parse(xml_path).getroot()

    @staticmethod
    def _prepare_objects_section(root: ET.Element) -> Tuple[str, List[int]]:
        sections: List[str] = []
        obj_int_ids: List[int] = []
        for obj in root.findall("object"):
            obj_id = obj.attrib.get("id", "").strip()
            if not obj_id:
                continue
            obj_int_ids.append(int(obj_id.lstrip("obj_")))
            xml_str = ET.tostring(obj, encoding="unicode", method="xml").strip()
            sections.append(f"{obj_id}\n{xml_str}")
        return "\n\n".join(sections), obj_int_ids

    @staticmethod
    def _build_prompt(objects_section: str, user_query: str) -> str:
        """Return an instruction prompt that asks the LLM to reason and then output *up to five* matching
        objects together with a confidence score (0-100).

        Required response format (XML-like, **do not** add markdown fencing):

        <reasoning>YOUR DETAILED CHAIN-OF-THOUGHT</reasoning>
        <objects>
          <object id="object_12" confidence="92" />
          <object id="object_05" confidence="67" />
          ... up to 5 entries ...
        </objects>

        • The <objects> block may be empty if none of the scene objects match the USER QUERY well enough.
        • Confidence must be an integer between 0 and 100 (higher = better match).
        """

        return (
            "SYSTEM: You are an expert assistant whose task is to choose up to five objects from a 3-D scene that best match the USER QUERY.\n"
            "OBJECTS:\n"
            f"{objects_section}\n"
            f"USER QUERY: {user_query}\n"
            "INSTRUCTIONS:\n"
            "1. Think step-by-step which objects satisfy the USER QUERY.\n"
            "2. Assign each promising object a confidence score (0-100).\n"
            "3. Return the *top five* objects sorted by confidence (highest first). If no object is suitable, return an empty <objects> block.\n\n"
            "Respond **exactly** in the following XML-style without extra commentary or markdown fencing:\n"
            "<reasoning>YOUR DETAILED CHAIN-OF-THOUGHT</reasoning>\n"
            "<objects>\n"
            "  <object id=\"object_XX\" confidence=\"YY\" />\n"
            "  ... up to five such lines ...\n"
            "</objects>"
        )

    @staticmethod
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

    @staticmethod
    def _fallback_ce(scene_name: str, query: str, k: int, data_dir: str) -> Dict:
        """Fallback to cross-encoder-only ranking in case the LLM result is unusable."""
        logging.info("Falling back to cross-encoder-only reranking.")
        from llm_query.query import query_scene as _query_scene  # Local import to avoid cycle

        return _query_scene(
            scene_name,
            query,
            data_dir=data_dir,
            k=k,
            ce_only=True,
            retrieval_strategy="embedding",
        ) 