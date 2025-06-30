import trimesh
import numpy as np
from trimesh.transformations import rotation_matrix
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

        # Load the input mesh file
        print(f"Loading mesh from: {input_path}")
        mesh = trimesh.load(input_path)

        # Rotate the mesh only when the source is a PLY file. ScanNet scenes are Z-up
        # whereas glTF / three.js assume Y-up. A -90° rotation around the X-axis
        # converts from Z-up to Y-up, ensuring intuitive camera controls in Gradio.
        if os.path.splitext(input_path)[1].lower() == ".ply":
            rot_mat = rotation_matrix(np.radians(-90), [1, 0, 0])

            # The loaded object can be either a Trimesh or a Scene. Handle both.
            if isinstance(mesh, trimesh.Scene):
                mesh.apply_transform(rot_mat)
            else:
                mesh.apply_transform(rot_mat)

        # Export the (possibly transformed) mesh to a GLB file
        print(f"Exporting to GLB format at: {output_path}")
        mesh.export(file_type='glb', file_obj=output_path)

        print("\nConversion successful!")
        return output_path

    except Exception as e:
        print(f"An error occurred during conversion: {e}")
        return None