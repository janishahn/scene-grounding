import logging
import json
import re
import os
from typing import List
from xml.etree import ElementTree as ET
from concurrent.futures import ThreadPoolExecutor, as_completed

from tqdm import tqdm
import ollama

__all__ = ["jsonl_to_xml"]

FIELDS = [
    "name",          # name of the object
    "role",          # role & function within the scene
    "spatial",       # spatial relationships / position
    "location",      # coarse/high-level location in the scene environment
    "interaction",   # interaction with nearby items
    "environment",   # environmental cues
    "scene_purpose", # inference about scene purpose
    "purpose",       # typical use of object
    "shape",         # shape & construction
    "materials",     # material / texture
    "color",         # colour & pattern
    "condition",     # state of repair / wear
    "details",       # notable fine details
]

SCHEMA_DESC = (
    "Return an <object> XML snippet with the following tags (all mandatory, single-line each):\n"
    "<object id=\"...\">\n"
    + "\n".join([f"  <{f}>...</{f}>" for f in FIELDS]) + "\n</object>\n"
    "- name - name of the object.\n"
    "- role - summarise the object's role in the scene.\n"
    "- spatial - describe its position and relationships.\n"
    "- location - coarse/high-level cue about where the object is situated (e.g. indoors, outdoors, on the rug).\n"
    "- interaction - how it interacts/supports/blocks etc.\n"
    "- environment - lighting/background cues if relevant.\n"
    "- scene_purpose - what the object's presence implies about the scene.\n"
    "- purpose - typical use of the object.\n"
    "- shape - overall form and parts.\n"
    "- materials - material & texture.\n"
    "- color - dominant colours/patterns.\n"
    "- condition - wear, damage, cleanliness.\n"
    "- details - fine-grained notable details.\n\n"
    "In each individual XML field, make sure to include the object's so that each field is interpretable on its own.\n"
    "Only output the snippet, no markdown fences or commentary."
)

# quick presence check via substrings
def _has_all_tags(text: str) -> bool:
    # Accept either explicit <tag>...</tag> pairs or self-closing <tag/> variants.
    for tag in FIELDS:
        pattern = rf"<{tag}\b[^>]*>(.*?)</{tag}>|<{tag}\b[^>]*/>"
        if not re.search(pattern, text, flags=re.IGNORECASE | re.DOTALL):
            return False
    return True

def _prompt_for_object(obj_id: int, global_cap: str, local_cap: str) -> List[dict]:
    sys_msg = (
        "You are a structured-information extraction assistant. "
        "Convert the provided GLOBAL and LOCAL captions into the XML snippet described below.\n\n"
        + SCHEMA_DESC
    )
    user_msg = (
        f"Object id: obj_{obj_id}\n\nGLOBAL:\n{global_cap}\n\nLOCAL:\n{local_cap}\n\nProduce XML now."
    )
    return [
        {"role": "system", "content": sys_msg},
        {"role": "user", "content": user_msg},
    ]

def _clean_response(text: str) -> str:
    """Return *text* stripped from common LLM markdown decorations.

    This removes leading/trailing back-tick fences (single, triple or ```xml) as
    well as a possible leading "xml" language identifier. Surrounding
    whitespace is trimmed.
    """
    # Trim surrounding whitespace
    text = text.strip()

    # Remove Markdown ``` fences
    if text.startswith("```") and text.endswith("```"):
        # ```xml ... ```  or ``` ... ```
        text = text.split("```", 2)[1] if text.count("```") >= 2 else text
        text = text.strip()

    # Remove any leading/trailing back-ticks that may remain
    text = text.lstrip("`\n ").rstrip("`\n ")

    # Drop a leading language identifier
    if text.lower().startswith("xml"):
        text = text[3:].lstrip()

    return text

def _extract_object_snippets(text: str) -> List[str]:
    """Return a list of <object>...</object> XML snippets found in *text*."""
    pattern = re.compile(r"<object\b[^>]*>.*?</object>", re.IGNORECASE | re.DOTALL)
    return pattern.findall(text)

def _repair_tag_mismatches(text: str) -> str:
    """
    Repair mismatched opening and closing tags by replacing the closing tag with the correct one.
    """
    for tag in FIELDS:
        # Find patterns where the closing tag is NOT the same as opening.
        pattern = rf"(<{tag}\b[^>]*>)(.*?)</([a-zA-Z0-9_]+)>"
        def _repl(match):
            open_tag, content, close_tag = match.groups()
            if close_tag.lower() != tag.lower():
                return f"{open_tag}{content}</{tag}>"
            return match.group(0)
        text = re.sub(pattern, _repl, text, flags=re.IGNORECASE | re.DOTALL)
    return text

def _generate_snippet(model: str, obj_id: int, gcap: str, lcap: str, max_attempts: int = 3) -> str:
    for attempt in range(1, max_attempts + 1):
        try:
            messages = _prompt_for_object(obj_id, gcap, lcap)
            resp = ollama.chat(model=model, messages=messages, options={"temperature": 0.8, "num_predict": 1024, "num_ctx": 8192})
            text = resp.get("message", {}).get("content", "").strip()
            if not text:
                raise ValueError("Empty response")

            # Clean markdown / fencing decorations
            text = _clean_response(text)

            # Attempt to repair common mismatched closing tags
            text = _repair_tag_mismatches(text)

            snippets = _extract_object_snippets(text)
            if snippets:
                # Try to find the exact object id first
                selected = None
                for snip in snippets:
                    m = re.search(r"id=\"?([^\">\s]+)\"?", snip)
                    if m and m.group(1).strip().lower() == f"obj_{obj_id}".lower():
                        selected = snip.strip()
                        break
                # Fallback to first snippet with all required tags
                if selected is None:
                    for snip in snippets:
                        if _has_all_tags(snip):
                            selected = snip.strip()
                            break
                # Fallback to very first snippet if still none
                if selected is None:
                    selected = snippets[0].strip()
                text = selected

            # Final whitespace trim to avoid stray chars before root element
            text = text.strip()

            if not _has_all_tags(text):
                logging.debug("Snippet missing tags:\n%s", text)
                raise ValueError("Missing required tags")
            # XML well-formedness
            try:
                ET.fromstring(text)
            except ET.ParseError as e:
                logging.debug("Malformed XML snippet returned by LLM:\n%s", text)
                raise ValueError(f"XML parse error: {e}")
            return text
        except Exception as e:
            logging.warning(f"Attempt {attempt}/{max_attempts} XML generation for object {obj_id} failed: {e}")
            logging.warning(f"Ollama query parameters: {messages}")
            logging.warning(f"Ollama response: {resp}")
            if attempt == max_attempts:
                logging.error(f"Giving up on object {obj_id}; saving raw captions.")
                # Fill fallback snippet with concatenated captions in <details>
                from xml.sax.saxutils import escape
                safe_global = escape(gcap)
                safe_local = escape(lcap)
                fallback = [
                    f"<{tag}>" + (safe_global if tag in ["role", "spatial", "location", "interaction", "environment", "scene_purpose"] else safe_local) + f"</{tag}>"
                    for tag in FIELDS
                ]
                return f"<object id=\"obj_{obj_id}\">" + "".join(fallback) + "</object>"

def jsonl_to_xml(jsonl_path: str, xml_out_path: str, scene_id: str, model: str = "gemma3:4b-it-qat") -> bool:
    """Convert a captions JSONL file to an XML document using an LLM.

    Parameters
    ----------
    jsonl_path
        Path to the ``*.captions.jsonl`` file produced by the captioning stage.
    xml_out_path
        Output path for the XML file.
    scene_id
        Scene identifier (for the root element).
    model
        Ollama model name.
    """
    # Ensure concurrent generation is enabled
    os.environ.setdefault("OLLAMA_NUM_PARALLEL", "4")

    logging.getLogger("httpx").setLevel(logging.WARNING)

    objects: List[str] = []
    # Collect all jobs so we can run them in parallel
    jobs = []  # (obj_id, gcap, lcap)
    with open(jsonl_path, "r", encoding="utf-8") as f:
        for line in f:
            try:
                j = json.loads(line)
                obj_id = int(j["object_id"])
                gcap_raw = j.get("global", "")
                lcap_raw = j.get("local", "")

                # Allow list-valued entries (multi-view). Convert to verbose text block.
                if isinstance(gcap_raw, list):
                    gcap = "\n".join(
                        f"[GLOBAL view {idx + 1}] {c}" for idx, c in enumerate(gcap_raw)
                    )
                else:
                    gcap = gcap_raw

                if isinstance(lcap_raw, list):
                    lcap = "\n".join(
                        f"[LOCAL view {idx + 1}] {c}" for idx, c in enumerate(lcap_raw)
                    )
                else:
                    lcap = lcap_raw

                jobs.append((obj_id, gcap, lcap))
            except Exception as e:
                logging.warning(f"Skipping malformed line: {e}")
                continue

    # Number of parallel requests
    max_workers = 4

    # Run snippet generation concurrently
    results = {}
    with ThreadPoolExecutor(max_workers=max_workers) as pool:
        future_to_id = {
            pool.submit(_generate_snippet, model, oid, gcap, lcap, 5): oid
            for oid, gcap, lcap in jobs
        }
        for future in tqdm(
            as_completed(future_to_id), total=len(jobs), desc="Structuring objects"
        ):
            oid = future_to_id[future]
            try:
                results[oid] = future.result()
            except Exception as exc:
                logging.error(f"Object {oid} generation failed: {exc}")

    # Reconstruct objects list in the original order
    objects = [results[oid] for oid, _, _ in jobs if oid in results]

    n_ok = len(objects)
    n_err = len(jobs) - n_ok
    summary_msg = f"Converted {n_ok}/{len(jobs)} objects"
    if n_err > 0:
        summary_msg += f" ({n_err} errors)."
    else:
        summary_msg += "."
    logging.info(summary_msg)

    doc = f"<scene id=\"{scene_id}\">\n" + "\n".join(objects) + "\n</scene>\n"
    with open(xml_out_path, "w", encoding="utf-8") as f:
        f.write(doc)
    logging.info(f"Saved structured XML ⇒ {xml_out_path}")
    return True 