import open3d as o3d
import numpy as np
from sklearn.neighbors import NearestNeighbors
import os
import logging

def highlight_objects_in_mesh(ply_path, segmentation_points, masks, object_probs, output_path, search_radius=0.02):
    """
    Create highlighted PLY mesh by mapping segmentation points to mesh vertices using nearest neighbors.
    
    Args:
        ply_path: path to original PLY mesh
        segmentation_points: (N, 3) point cloud used for segmentation
        masks: (N, num_objects) boolean masks from prediction
        object_probs: dictionary of object IDs to probabilities
        output_path: where to save highlighted mesh
        search_radius: radius in meters for finding nearby mesh vertices
    
    Returns:
        output_path if successful, None if failed
    """

    if not object_probs:
        logging.warning("No objects to highlight, skipping mesh processing.")
        return None
    
    logging.info(f"Loading PLY mesh from {ply_path}")
    mesh = o3d.io.read_triangle_mesh(ply_path)
    mesh_vertices = np.asarray(mesh.vertices)
    colors = np.asarray(mesh.vertex_colors)

    logging.info(f"Mesh: {len(mesh_vertices):,} vertices, Segmentation: {len(segmentation_points):,} points")
    
    # Build spatial index for mesh vertices
    nbrs = NearestNeighbors(radius=search_radius, algorithm='ball_tree').fit(mesh_vertices)
    
    logging.info(f"Highlighting {len(object_probs)} objects")
    
    # Step 1: Dim all existing colors to 30%
    colors = colors * 0.3
    
    # Step 2: Pre-compute min/max probabilities for color scaling (red→green)
    probs = list(object_probs.values())
    if not probs:
        logging.warning("No probabilities provided – skipping highlight.")
        return None

    min_prob = min(probs)
    max_prob = max(probs)
    range_prob = max_prob - min_prob if max_prob != min_prob else 1.0

    # Step 3: Highlight each requested object with a gradient colour
    for obj_id, prob in object_probs.items():
        if obj_id >= masks.shape[1]:
            logging.warning(f"Object {obj_id} doesn't exist (max: {masks.shape[1]-1})")
            continue
        
        # Get segmentation points belonging to this object
        object_mask = masks[:, obj_id]
        seg_point_indices = np.where(object_mask)[0]
        object_seg_points = segmentation_points[seg_point_indices]
        
        # Find mesh vertices near these segmentation points
        mesh_vertex_indices = set()
        batch_size = 1000  # Process in batches for memory efficiency
        
        # For each segmentation point of the object, find the closest mesh(es)
        for start_idx in range(0, len(object_seg_points), batch_size):
            end_idx = min(start_idx + batch_size, len(object_seg_points))
            batch_points = object_seg_points[start_idx:end_idx]
            
            # Find neighbors for this batch
            distances, indices = nbrs.radius_neighbors(batch_points)
            
            # Collect all mesh vertex indices
            for vertex_indices in indices:
                mesh_vertex_indices.update(vertex_indices)
        
        mesh_vertex_indices = list(mesh_vertex_indices)
        
        # Apply bright highlight color to matching mesh vertices
        if len(mesh_vertex_indices) > 0:
            # Linear interpolation between red (low) and green (high)
            if max_prob == min_prob:
                t = 1.0  # Only one object – use green
            else:
                t = (prob - min_prob) / range_prob
            color = [1.0 - t, t, 0.0]
            colors[mesh_vertex_indices] = color
    
    # Save the highlighted mesh
    mesh.vertex_colors = o3d.utility.Vector3dVector(colors)
    success = o3d.io.write_triangle_mesh(output_path, mesh)
    
    if success:
        logging.info(f"Successfully saved highlighted mesh to {output_path}")
        return output_path
    else:
        logging.error(f"Failed to save mesh to {output_path}")
        return None

def load_scene_data(scene_id):
    """Load all required data for a scene."""
    
    logging.info(f"Loading data for scene {scene_id}")
    
    # Absolute paths
    ply_path = f"/home/vlm_search/scene-grounding/app/scans/{scene_id}.ply"
    masks_path = f'/home/vlm_search/scene-grounding/maskclustering/data/prediction/scannetpp_class_agnostic/{scene_id}.npz'
    seg_cloud_path = f'/home/vlm_search/scene-grounding/maskclustering/data/scannetpp/pcld_0.25/{scene_id}.pth'    
    
    # Load masks
    masks = np.load(masks_path)['pred_masks']
    logging.info(f"Loaded masks with shape: {masks.shape}")
    
    # Load segmentation point cloud
    segmentation_points = None
    if os.path.exists(seg_cloud_path):
        try:
            import torch
            data = torch.load(seg_cloud_path, weights_only=False)
            segmentation_points = np.asarray(data['sampled_coords'])            
            logging.info(f"Loaded segmentation points with shape: {segmentation_points.shape}")
        except Exception as e:
            logging.error(f"Failed to load from {seg_cloud_path}: {e}")
    
    if segmentation_points is None:
        raise FileNotFoundError("Could not find segmentation point cloud file")
    
    return ply_path, segmentation_points, masks

def create_highlighted_scene(scene_id, object_probs, output_dir="app/scans"):
    """
    Complete pipeline to create a highlighted scene.
    
    Args:
        scene_id: Scene identifier (e.g., "95d525fbfd")
        object_probs: Dictionary of object IDs to probabilities
        output_dir: Directory to save results
    
    Returns:
        Path to highlighted mesh file
    """
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Load scene data
    ply_path, segmentation_points, masks = load_scene_data(scene_id)
    
    # Generate output filename
    obj_str = "_".join(map(str, object_probs.keys()))
    output_filename = f"{scene_id}_highlighted_objects_{obj_str}.ply"
    output_path = os.path.join(output_dir, output_filename)
    
    # Create highlighted mesh
    result = highlight_objects_in_mesh(
        ply_path=ply_path,
        segmentation_points=segmentation_points,
        masks=masks,
        object_probs=object_probs,
        output_path=output_path,
        search_radius=0.02  # 2cm radius based on coordinate analysis
    )
    
    return result