import logging
import json
import re
from typing import List
from xml.etree import ElementTree as ET

import ollama

__all__ = ["jsonl_to_xml"]

FIELDS = [
    "name",          # name of the object
    "role",          # role & function within the scene
    "spatial",       # spatial relationships / position
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
    return all(f"<{tag}>" in text and f"</{tag}>" in text for tag in FIELDS)

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

def _generate_snippet(model: str, obj_id: int, gcap: str, lcap: str, max_attempts: int = 3) -> str:
    for attempt in range(1, max_attempts + 1):
        try:
            messages = _prompt_for_object(obj_id, gcap, lcap)
            resp = ollama.chat(model=model, messages=messages, options={"temperature": 0.8, "num_predict": 1024, "num_ctx": 8192})
            text = resp.get("message", {}).get("content", "").strip()
            if not text:
                raise ValueError("Empty response")
            if text.startswith("```"):
                text = text.strip("`\n").strip()
            # Remove a leading markdown language identifier (e.g. "xml") that may remain
            if text.lower().startswith("xml"):
                text = text[3:].lstrip()
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
            if attempt == max_attempts:
                logging.error(f"Giving up on object {obj_id}; saving raw captions.")
                # Fill fallback snippet with concatenated captions in <details>
                from xml.sax.saxutils import escape
                safe_global = escape(gcap)
                safe_local = escape(lcap)
                fallback = [
                    f"<{tag}>" + (safe_global if tag in ["role", "spatial", "interaction", "environment", "scene_purpose"] else safe_local) + f"</{tag}>"
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
    objects: List[str] = []
    with open(jsonl_path, "r", encoding="utf-8") as f:
        for line in f:
            try:
                j = json.loads(line)
                obj_id = int(j["object_id"])
                gcap = j.get("global", "")
                lcap = j.get("local", "")
            except Exception as e:
                logging.warning(f"Skipping malformed line: {e}")
                continue
            snippet = _generate_snippet(model, obj_id, gcap, lcap)
            objects.append(snippet)

    doc = f"<scene id=\"{scene_id}\">\n" + "\n".join(objects) + "\n</scene>\n"
    with open(xml_out_path, "w", encoding="utf-8") as f:
        f.write(doc)
    logging.info(f"Saved structured XML ⇒ {xml_out_path}")
    return True 