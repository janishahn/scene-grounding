import gradio as gr
import os
import shutil
import re

# --- Backend: File-based Highlighting ---

FILE_PATH = "../maskclustering/data/scannetpp/data/95d525fbfd/scans/mesh_aligned_0.05.glb"


def find_object(query: str) -> str: 
    """
    Takes the user query and returns a file path to a new & highlighted 3 model.
    """
    return FILE_PATH

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
            value=FILE_PATH, # Initial model to display
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
    demo.launch()

