import numpy as np
import pyviz3d.visualizer as viz
from utils.config import get_dataset, get_args
import open3d as o3d

# Since there are hundreds of objects in the scene, assigning visually distinguishable colors to each object is difficult. You can change the random seed to check if the two objects are actually segmented apart.
np.random.seed(4)

def vis_one_object(point_ids, scene_points):
    points = scene_points[point_ids]
    color = (np.random.rand(3) * 0.7 + 0.3) * 255
    colors = np.tile(color, (points.shape[0], 1))
    return point_ids, points, colors, color, np.mean(points, axis=0)

def main(args):
    point_size = 20
    label_colors, labels, centers = [], [], []
    dataset = get_dataset(args)

    # NOTE: Added by us (Markus & Janis)
    # Handle different dataset types
    if hasattr(dataset, 'mesh_path'):
        mesh = o3d.io.read_triangle_mesh(dataset.mesh_path)
        scene_points = np.asarray(mesh.vertices)
        scene_colors = np.asarray(mesh.vertex_colors)
        # Since the color of raw scan may be too dark, we brighten it tone mapping
        scene_colors = np.power(scene_colors, 1/2.2)
        scene_colors = scene_colors * 255
    else: 
        # ScanNet++ case - load from point cloud
        scene_points = dataset.get_scene_points()
        # Generate default colors if not available
        scene_colors = np.ones((scene_points.shape[0], 3)) * 128  # Gray default

    scene_points = scene_points - np.mean(scene_points, axis=0)
    instance_colors = np.zeros_like(scene_colors)

    v = viz.Visualizer()

    pred = np.load(f'data/prediction/{args.config}_class_agnostic/{args.seq_name}.npz')
    # pred = np.load(f'data/prediction/{args.config}/{args.seq_name}.npz')

    masks = pred['pred_masks']
    num_instances = masks.shape[1]
    
    # NEW: Highlight specific objects if specified
    highlight_color_palette = [
        [255, 0, 0],    # Red
        [0, 255, 0],    # Green  
        [0, 0, 255],    # Blue
        [255, 255, 0],  # Yellow
        [255, 0, 255],  # Magenta
        [0, 255, 255],  # Cyan
    ]
    
    # NEW: Check if we're in highlighting mode
    highlighting_mode = args.highlight_objects and len(args.highlight_objects) > 0
    
    for idx in range(num_instances):
        mask = masks[:, idx]
        point_ids = np.where(mask)[0]

        # Check if this object should be highlighted
        if highlighting_mode and idx in args.highlight_objects:
            # Use bright highlighting color (full opacity)
            highlight_idx = args.highlight_objects.index(idx)
            highlight_color = highlight_color_palette[highlight_idx % len(highlight_color_palette)]
            colors = np.tile(highlight_color, (len(point_ids), 1))
            label_color = highlight_color
        else:
            # Use normal random color
            point_ids, points, colors, label_color, center = vis_one_object(point_ids, scene_points)
            
            # NEW: If in highlighting mode, make non-highlighted objects semi-transparent
            if highlighting_mode:
                label_color = label_color * 0.3  # 30% opacity
                colors = colors * 0.3  # 30% opacity
        
        instance_colors[point_ids] = label_color
        label_colors.append(label_color)
        labels.append(str(idx))
        centers.append(np.mean(scene_points[point_ids], axis=0))
        # If you want to visualize each object separately, you can uncomment the following line.
        # v.add_points(f'{idx}', points, colors, visible=True, point_size=point_size)

    v.add_points('RGB', scene_points, scene_colors, visible=False, point_size=point_size)

    labeled_scene_points_mask = np.where(np.sum(instance_colors, axis=1) != 0)
    v.add_points('Instances', scene_points[labeled_scene_points_mask], instance_colors[labeled_scene_points_mask], visible=True, point_size=point_size)

    # If you want to visualize the label id of each object, you can uncomment the following line.
    # v.add_labels('Labels', labels, centers, label_colors)

    # NEW: Add suffix to filename if highlighting
    suffix = "_highlighted" if highlighting_mode else ""
    v.save(f'data/vis/{args.seq_name}{suffix}')

if __name__ == '__main__':
    args = get_args()
    main(args)