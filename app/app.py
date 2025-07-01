import gradio as gr
import logging
import os
import re
import html

from llm_query.query import query_scene
from app.highlighting import create_highlighted_scene
from app.convert import convert_to_glb
from typing import List, Tuple


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
        snippet = info.get(field, "") if isinstance(info, dict) else ""
        caption = f"obj{obj_id} | {score:.2f}"
        if snippet:
            caption += f"\n{snippet[:100]}"
        if img_path:
            gallery_items.append((img_path, caption))
        truncated_snippet = snippet[:80] + ("..." if len(snippet) > 80 else "")
        explanation_rows.append([obj_id, round(score, 3), truncated_snippet])

    return gallery_items, explanation_rows


def build_details_html(objects: List[Tuple[int, float, str, dict]]) -> str:
    """Return HTML with <details> blocks for each object containing full XML fields."""
    blocks: List[str] = []
    for obj_id, score, field, info in objects:
        rows = "".join(
            f"<li><b>{html.escape(k)}</b>: {html.escape(v)}</li>" for k, v in (info or {}).items()
        ) or "<i>No details available</i>"
        block = (
            f"<details><summary>obj{obj_id} | {score:.3f} | {field}</summary>"
            f"<ul style='margin-left:1em'>{rows}</ul></details>"
        )
        blocks.append(block)
    return "\n".join(blocks)


def find_object(user_query: str, mode: str):
    """Return highlighted scene along with gallery and explanation table."""
    try:
        use_fast = (mode == "Fast retrieval (bi-encoder + CE)")
        result: dict = query_scene(scene_name=SCENE_ID, query=user_query, data_dir="vlm_caption/outputs", k=3, ce_only=not use_fast)
        objects = result.get('objects', [])

        object_ids = [obj[0] for obj in objects]

        # Highlight in 3D
        path_to_ply_file = create_highlighted_scene(scene_id=SCENE_ID, object_ids_to_highlight=object_ids)
        final_path = convert_to_glb(path_to_ply_file)

        # Build gallery, explanations, and detailed HTML
        gallery_items, explanation_rows = build_gallery_and_explanations(SCENE_ID, objects)
        details_html = build_details_html(objects)

        return final_path, gallery_items, explanation_rows, details_html
    except Exception as e:
        logging.warning(f"Query failed: {e}")
        return SCENE_MODEL_PATH, [], [], ""


def test_highlighting(object_ids_str: str):
    try:
        # Parse comma-separated string into list of integers
        if not object_ids_str.strip():
            return SCENE_MODEL_PATH, [], [], ""
            
        object_ids = [int(x.strip()) for x in object_ids_str.split(',') if x.strip()]
        logging.info(f"Parsed object IDs: {object_ids}")
        
        path_to_ply_file = create_highlighted_scene(scene_id=SCENE_ID, object_ids_to_highlight=object_ids)
        final_path = convert_to_glb(path_to_ply_file)

        gallery_items: List[Tuple[str, str]] = []
        for obj_id in object_ids:
            img_path = _find_best_view(SCENE_ID, obj_id)
            if img_path:
                gallery_items.append((img_path, f"obj{obj_id}"))

        explanation_rows = [[obj_id, "-", "-"] for obj_id in object_ids]
        return final_path, gallery_items, explanation_rows, ""
    except ValueError as e:
        logging.warning(f"Invalid object ID format: {e}")
        return SCENE_MODEL_PATH, [], [], ""
    except Exception as e:
        logging.warning(f"Highlighting failed: {e}")
        return SCENE_MODEL_PATH, [], [], ""


# --- Gradio Frontend Definition ---
with gr.Blocks(theme=gr.themes.Soft()) as app:
    gr.Markdown(
        """
        # 3D Scene Object Highlighting
        Type in a description of an object in the scene to highlight it.        
        """
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
                rows=2,
                height="400px",
                preview=True
            )

    # Explanation table below
    explanation_table = gr.Dataframe(
        headers=["Object", "Logit", "Snippet"],
        datatype=["number", "number", "str"],
        label="Explanation of Selection",
        interactive=False,
        wrap=True,
        visible=True,
    )

    # Collapsible full-object details (HTML <details> blocks)
    object_details = gr.HTML(label="Full Object Attributes")

    with gr.Row():
        
        with gr.Column():
            # The Textbox component for user input.
            text_input = gr.Textbox(
                label="What object are you looking for?",
                info="e.g., 'A place where I can wash my hands'",
                placeholder="Type here and press Enter..."
            )

        with gr.Column():
            ranking_mode = gr.Dropdown(
                label="Ranking mode",
                choices=["Fast retrieval (bi-encoder + CE)", "Cross-encoder only"],
                value="Fast retrieval (bi-encoder + CE)",
            )

        with gr.Column():
            # Test textbox for highlighting
            test_input = gr.Textbox(
                label="Test Object IDs",
                info="Enter object IDs to highlight (comma-separated)",
                placeholder="e.g., 0, 1, 2"
            )

    # Event listener for the textbox submission.
    text_input.submit(
        fn=find_object,
        inputs=[text_input, ranking_mode],
        outputs=[model_3d, image_gallery, explanation_table, object_details]
    )

    test_input.submit(
        fn=test_highlighting,
        inputs=test_input,
        outputs=[model_3d, image_gallery, explanation_table, object_details]
    )

# Launch the Gradio app.
if __name__ == "__main__":
    initialization()
    app.launch(allowed_paths=[SCAN_DIR])

