import trimesh
import os

def convert_to_glb(input_path, output_path=None):
    """
    Loads a mesh file and converts it to the GLB format.
    """
    try:
        # If no output path is specified, create one automatically
        if output_path is None:
            base_name = os.path.splitext(input_path)[0]
            output_path = base_name + ".glb"

        # Load the input mesh file (e.g., your .ply file)
        print(f"Loading mesh from: {input_path}")
        mesh = trimesh.load(input_path)

        # Export the mesh to a GLB file
        print(f"Exporting to GLB format at: {output_path}")
        mesh.export(file_type='glb', file_obj=output_path)

        print("\nConversion successful!")
        return output_path

    except Exception as e:
        print(f"An error occurred during conversion: {e}")
        return None

# --- USAGE ---
# Replace 'your_problem_file.ply' with the path to the file causing the error.
input_file = '../maskclustering/data/scannetpp/data/95d525fbfd/scans/mesh_aligned_0.05.ply'

# Convert the file
converted_file_path = convert_to_glb(input_file)

# You can now use the 'converted_file_path' in your Gradio app
if converted_file_path:
    print(f"\nUse this file in your Gradio app: {converted_file_path}")