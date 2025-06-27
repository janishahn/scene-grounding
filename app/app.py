import gradio as gr
import logging
import os

from llm_query.query import query_scene
from app.highlighting import create_highlighted_scene
from app.convert import convert_to_glb
from typing import List


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
    # Convert .ply file to .glb if necessary
    if not os.path.exists(SCENE_MODEL_PATH):
        logging.info(f"Converting {SCENE_ID}.ply to GLB format...")
        convert_to_glb(os.path.join(os.path.dirname(os.path.abspath(__file__)), "scans", f"{SCENE_ID}.ply"), SCENE_MODEL_PATH)
    logging.info("Initialization complete.")


def find_object(user_query: str) -> str:
    """
    Find object matching the user_query, using the pre-computed captions. Return a new scene, 
    where the relevant objects are highlighted. 
    """
    try:
        result: dict = query_scene(scene_name=SCENE_ID, query=user_query, data_dir="vlm_caption/outputs", k=3)
        # TODO: Figure out what and how to display this
        queried_field = result['field']
        objects = result['objects']

        object_ids = [obj[0] for obj in objects]
        # Create a scene, where the objects are highlighted
        path_to_ply_file = create_highlighted_scene(scene_id=SCENE_ID, object_ids_to_highlight=object_ids)
        # Convert to .glb
        final_path = convert_to_glb(path_to_ply_file)
    except Exception as e:
        logging.warning(f"Query failed: {e}")
        final_path = SCENE_MODEL_PATH
    
    return final_path

def test_highlighting(object_ids_str: str):
    try:
        # Parse comma-separated string into list of integers
        if not object_ids_str.strip():
            return SCENE_MODEL_PATH
            
        object_ids = [int(x.strip()) for x in object_ids_str.split(',') if x.strip()]
        logging.info(f"Parsed object IDs: {object_ids}")
        
        path_to_ply_file = create_highlighted_scene(scene_id=SCENE_ID, object_ids_to_highlight=object_ids)
        # Convert to .glb
        final_path = convert_to_glb(path_to_ply_file)
    except ValueError as e:
        logging.warning(f"Invalid object ID format: {e}")
        return SCENE_MODEL_PATH
    except Exception as e:
        logging.warning(f"Highlighting failed: {e}")
        return SCENE_MODEL_PATH

    return final_path

# --- Gradio Frontend Definition ---
with gr.Blocks(theme=gr.themes.Soft()) as demo:
    gr.Markdown(
        """
        # 3D Scene Object Highlighting
        Type in a description of an object in the scene to highlight it.        
        """
    )
    
    with gr.Row():
        # The 3D Model component. It now gets updated by receiving a new file path.
        model_3d = gr.Model3D(
            value=SCENE_MODEL_PATH,  # Initial model
        )
    
    with gr.Row():
        
        with gr.Column():
            # The Textbox component for user input.
            text_input = gr.Textbox(
                label="What object are you looking for?",
                info="e.g., 'A place where I can wash my hands'",
                placeholder="Type here and press Enter..."
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
        inputs=text_input, 
        outputs=model_3d
    )

    test_input.submit(
        fn=test_highlighting, 
        inputs=test_input, 
        outputs=model_3d
    )

# Launch the Gradio app.
if __name__ == "__main__":
    initialization()
    demo.launch(allowed_paths=[SCAN_DIR])

