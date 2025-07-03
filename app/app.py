import gradio as gr
import logging
import os
import re
import html

from llm_query.query import query_scene
from app.highlighting import create_highlighted_scene
from app.convert import convert_to_glb
from typing import List, Tuple
from numpy import exp


SCENE_ID = "95d525fbfd"

# Construct scene model path from project directory
SCENE_MODEL_PATH = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),  # Current script directory
    "scans",  # Assuming 'data' is the folder where the GLB file is stored
    f"{SCENE_ID}.glb"  # GLB file name based on SCENE_ID
)

# Directory that contains the GLB file – whitelist for Gradio
SCAN_DIR = os.path.dirname(SCENE_MODEL_PATH)

def initialization():
    # init logging 
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler()
        ]
    )
    logging.info("Initializing 3D Scene Object Highlighting Gradio App")
    logging.info(f"Using SCENE_ID: {SCENE_ID}")
    # Convert the original PLY to GLB so that orientation is corrected
    logging.info(f"Ensuring {SCENE_ID}.ply is converted to GLB with correct orientation...")
    convert_to_glb(
        os.path.join(os.path.dirname(os.path.abspath(__file__)), "scans", f"{SCENE_ID}.ply"),
        SCENE_MODEL_PATH,
    )
    logging.info("Initialization complete.")

def _logit_to_prob(score: float):
    return 100 / (1 + exp(-score))

def _score_to_percent(score: float, field: str) -> float:
    if field.startswith("llm_"):
        return score
    return _logit_to_prob(score)

def _find_best_view(scene_id: str, obj_id: int) -> str | None:
    """Return path to the best-view image for *obj_id* if available."""
    best_view_dir = f"maskclustering/data/scannetpp/data/{scene_id}/output/best_views"
    if not os.path.exists(best_view_dir):
        return None
    pattern = rf"obj{obj_id:04d}_.*highlighted\.(?:jpg|jpeg|png|bmp)$"
    for filename in os.listdir(best_view_dir):
        if re.match(pattern, filename, re.IGNORECASE) and 'cropped' not in filename.lower():
            return os.path.join(best_view_dir, filename)
    return None


def build_gallery_and_explanations(scene_id: str, objects: List[Tuple[int, float, str, dict]]):
    """Create data for Gradio Gallery and explanation table."""
    gallery_items: List[Tuple[str, str]] = []
    explanation_rows: List[List] = []

    for obj_id, score, field, info in objects:
        img_path = _find_best_view(scene_id, obj_id)

        # --------------------------------------------------------------
        # Build snippet / explanation
        # --------------------------------------------------------------
        if field.startswith("llm_"):
            # For any LLM-based retriever, display a concise preview (first 120 chars).
            snippet = info.get("reasoning", "") if isinstance(info, dict) else ""
            truncated_snippet = snippet[:120] + ("..." if len(snippet) > 120 else "")
        else:
            # For embedding or CE-based retrievers, attempt to show the
            # most relevant object attributes (name / purpose / role / details).
            important_tags = ["name", "purpose", "role", "details"]
            parts = [info.get(tag, "") for tag in important_tags if info.get(tag)] if isinstance(info, dict) else []
            snippet = " | ".join(parts)
            truncated_snippet = snippet[:80] + ("..." if len(snippet) > 80 else "")

        obj_name = info.get("name", "") if isinstance(info, dict) else ""
        if not obj_name:
            caption = f"obj{obj_id}"
        else:
            caption = f"{obj_id} | {obj_name}"

        if img_path:
            gallery_items.append((img_path, caption))

        explanation_rows.append([obj_id, round(_score_to_percent(score, field), 3)])

    return gallery_items, explanation_rows


def build_details_html(objects: List[Tuple[int, float, str, dict]]) -> str:
    """Return HTML with <details> blocks for each object containing full XML fields."""
    blocks: List[str] = []
    for obj_id, score, field, info in objects:
        rows = "".join(
            f"<li><b>{html.escape(k)}</b>: {html.escape(v)}</li>" for k, v in (info or {}).items()
        ) or "<i>No details available</i>"
        display_field = field
        if field.startswith("llm_") and isinstance(info, dict) and info.get("model_name"):
            display_field = info["model_name"]
        block = (
            f"<details><summary>obj{obj_id} | {_score_to_percent(score, field):.3f} | {display_field}</summary>"
            f"<ul style='margin-left:1em'>{rows}</ul></details>"
        )
        blocks.append(block)
    return "\n".join(blocks)


def find_object(user_query: str, mode: str):
    """Return highlighted scene along with gallery and explanation table."""
    try:
        if mode == "LLM ranking (Ollama)":
            result: dict = query_scene(
                scene_name=SCENE_ID,
                query=user_query,
                data_dir="vlm_caption/outputs",
                k=5,
                retrieval_strategy="llm_ollama",
            )
        elif mode == "LLM ranking (OpenRouter)":
            result: dict = query_scene(
                scene_name=SCENE_ID,
                query=user_query,
                data_dir="vlm_caption/outputs",
                k=5,
                retrieval_strategy="llm_openrouter",
                model_name="mistralai/mistral-small-3.2-24b-instruct:free"
                # model_name="openrouter/cypher-alpha:free"
                # model_name="google/gemma-3-27b-it:free"
            )
        else:
            use_fast = (mode == "Fast retrieval (bi-encoder + CE)")
            result: dict = query_scene(
                scene_name=SCENE_ID,
                query=user_query,
                data_dir="vlm_caption/outputs",
                k=5,
                ce_only=not use_fast,
                retrieval_strategy="embedding" if use_fast else "ce_only",
            )
        objects = result.get('objects', [])

        object_ids = [obj[0] for obj in objects]

        # Highlight in 3D
        path_to_ply_file = create_highlighted_scene(scene_id=SCENE_ID, object_ids_to_highlight=object_ids)
        final_path = convert_to_glb(path_to_ply_file)

        # Build gallery, explanations, and detailed HTML
        gallery_items, explanation_rows = build_gallery_and_explanations(SCENE_ID, objects)
        details_html = build_details_html(objects)

        # Extract full reasoning for LLM modes (only for the top-ranked object).
        full_reasoning = ""
        if objects and objects[0][2].startswith("llm_"):
            info_top = objects[0][3]
            if isinstance(info_top, dict):
                full_reasoning = info_top.get("reasoning", "")

        return final_path, gallery_items, explanation_rows, full_reasoning, details_html
    except Exception as e:
        logging.warning(f"Query failed: {e}")
        # Failure path – keep output arity consistent.
        return SCENE_MODEL_PATH, [], [], "", ""


# --- Gradio Frontend Definition ---
with gr.Blocks(theme=gr.themes.Soft()) as app:
    gr.Markdown(
        """
        # 3D Scene Object Highlighting
        Type in a description of an object in the scene to highlight it.        
        """
    )
    
    # Query input field and retrieval mode selector
    with gr.Row():
        with gr.Column():
            text_input = gr.Textbox(
                label="What object are you looking for?",
                info="e.g., 'A place where I can wash my hands'",
                placeholder="Type here and press Enter..."
            )

        with gr.Column():
            ranking_mode = gr.Dropdown(
                label="Ranking mode",
                choices=[
                    "Fast retrieval (bi-encoder + CE)",
                    "Cross-encoder only",
                    "LLM ranking (Ollama)",
                    "LLM ranking (OpenRouter)",
                ],
                value="LLM ranking (OpenRouter)",
            )

    with gr.Row():
        # The 3D Model component.
        with gr.Column():
            model_3d = gr.Model3D(
                value=SCENE_MODEL_PATH,  # Initial model
            )
        # Gallery for object images
        with gr.Column():
            image_gallery = gr.Gallery(
                format=".jpg",
                label="Highlighted Objects",
                show_label=True,
                elem_id="object_gallery",
                columns=2,
                rows=3,
                height="400px",
                preview=True
            )

    # Explanation table and LLM reasoning side-by-side
    with gr.Row():
        with gr.Column(scale=0.8):
            explanation_table = gr.Dataframe(
                headers=["Object", "Probability (%)"],
                datatype=["number", "number"],
                label="Object Probabilities",
                interactive=False,
                wrap=True,
                visible=True,
            )
        with gr.Column(scale=1.2):
            gr.Markdown("### Reasoning")
            reasoning_markdown = gr.Markdown(visible=True)

    # Collapsible full-object details (HTML <details> blocks)
    object_details = gr.HTML(label="Full Object Attributes")

    # Event listener for the textbox submission.
    text_input.submit(
        fn=find_object,
        inputs=[text_input, ranking_mode],
        outputs=[model_3d, image_gallery, explanation_table, reasoning_markdown, object_details]
    )

# Launch the Gradio app.
if __name__ == "__main__":
    initialization()
    app.launch(allowed_paths=[SCAN_DIR])

