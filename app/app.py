import gradio as gr
import logging
import os

from llm_query.query import query_scene

# Scene identifier must correspond to generated XML / indices
SCENE_ID = "95d525fbfd"

# Build absolute path to the scene GLB (maskclustering sits one level above repo root)
repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
SCENE_MODEL_PATH = os.path.abspath(
    os.path.join(repo_root, os.pardir, "maskclustering", "data", "scannetpp", "data", SCENE_ID, "scans", "mesh_aligned_0.05.glb")
)

# Directory that contains the GLB file – whitelist for Gradio
SCAN_DIR = os.path.dirname(SCENE_MODEL_PATH)


def find_object(user_query: str) -> str:
    """Return path to scene model after logging retrieval info.

    Currently returns the unmodified scene model; retrieval results are logged
    for debugging. In a future iteration this could dynamically colour the
    matched object and return a new GLB path.
    """

    try:
        res = query_scene(scene_name=SCENE_ID, query=user_query, data_dir="vlm_caption/outputs", k=3)
        logging.info(res)
    except Exception as e:
        logging.warning(f"Query failed: {e}")

    # TODO: generate highlighted model, for now return original
    return SCENE_MODEL_PATH

# --- Gradio Frontend Definition ---
with gr.Blocks(theme=gr.themes.Soft()) as demo:
    gr.Markdown(
        """
        # 3D Scene Object Highlighting (Corrected)
        Type the name of an object in the scene to highlight it.
        Try "sink", "chair", or "table".
        
        **Important:** Make sure you have downloaded `kitchen.obj` and `kitchen.mtl` into the same folder as this script.
        """
    )
    
    with gr.Row():
        # The 3D Model component. It now gets updated by receiving a new file path.
        model_3d = gr.Model3D(
            value=SCENE_MODEL_PATH,  # Initial model
        )       
        
        
        with gr.Column():
            # The Textbox component for user input.
            text_input = gr.Textbox(
                label="What object are you looking for?",
                info="e.g., 'A place where I can wash my hands'",
                placeholder="Type here and press Enter..."
            )

    # Event listener for the textbox submission.
    text_input.submit(
        fn=find_object, 
        inputs=text_input, 
        outputs=model_3d
    )

# Launch the Gradio app.
if __name__ == "__main__":
    demo.launch(allowed_paths=[SCAN_DIR])

