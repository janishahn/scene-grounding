import os
import cv2
import numpy as np
import torch


def calculate_padded_bounding_box(mask, padding_ratio=0.1):
    """Compute bounding box of mask with padding percentage."""
    y_indices, x_indices = np.where(mask > 0)
    if len(y_indices) == 0 or len(x_indices) == 0:
        return None
    
    # Get tight bbox
    left = np.min(x_indices)
    right = np.max(x_indices)
    top = np.min(y_indices)
    bottom = np.max(y_indices)
    
    # Add padding
    width = right - left + 1
    height = bottom - top + 1
    padding_x = int(width * padding_ratio)
    padding_y = int(height * padding_ratio)
    
    # Clip to image bounds
    img_h, img_w = mask.shape
    left = max(0, left - padding_x)
    right = min(img_w - 1, right + padding_x)
    top = max(0, top - padding_y)
    bottom = min(img_h - 1, bottom + padding_y)
    
    return [left, top, right, bottom]

def create_best_view_dir(dataset_root: str, debug: bool = False) -> str:

    best_views_dir = os.path.join(dataset_root, 'output', 'best_views')
    try:
        os.makedirs(best_views_dir, exist_ok=True)
        if debug:
            print(f"Created or confirmed best_views_dir at: {best_views_dir}")
    except Exception as e:
        print(f"ERROR: Failed to create best views directory at {best_views_dir}: {str(e)}")
        raise
        
    return best_views_dir

def create_highlighted_version(dataset, frame_id, mask_id, highlighted_rgb, padding_gap: int = 6):
    """
    Creates a highlighted version of an image by overlaying a semi-transparent color
    on a specific segmentation mask and adding a border around the masked object.
    This function helps visualize a particular object in a frame by making it stand
    out with a colored highlight.
    """
    segmentation = dataset.get_segmentation(frame_id)
    mask = (segmentation == mask_id)

    # Apply morphological closing to bridge small gaps in the mask
    kernel_size = 50
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))
    closed_mask = cv2.morphologyEx(mask.astype(np.uint8), cv2.MORPH_CLOSE, kernel)
    
    highlight_color = (0, 255, 0)  # Green highlight
    outline_thickness = 5  # Thickness of the contour line

    dilation_radius = padding_gap + (outline_thickness // 2)

    if dilation_radius > 0:
        dilation_kernel_size = 2 * dilation_radius + 1
        dilation_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (dilation_kernel_size, dilation_kernel_size))
        
        padded_mask_for_contour = cv2.dilate(closed_mask, dilation_kernel, iterations=1)
    else:
        padded_mask_for_contour = closed_mask
    
    contours, _ = cv2.findContours(padded_mask_for_contour.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    cv2.drawContours(highlighted_rgb, contours, -1, highlight_color, outline_thickness)

    return closed_mask, highlighted_rgb


def save_best_views(dataset, object_dict, args):
    """Save best view images and metadata for each object."""
    # Add debug logging
    debug = getattr(args, "debug", False)
    if debug:
        print(f"====> Starting save_best_views for {len(object_dict)} objects")
    
    scene_root = dataset.root
    best_views_dir = create_best_view_dir(scene_root, debug)

    # Track successful saves
    saved_count = 0

    num_views = int(getattr(args, "num_best_views", 1))

    for obj_id, obj_data in object_dict.items():
        # Skip if no representative masks
        if 'repre_mask_list' not in obj_data or len(obj_data['repre_mask_list']) == 0:
            if debug:
                print(f"Object {obj_id} has no representative masks, skipping")
            continue
            
        # NOTE: We currently rely on MaskClustering to provide us with the most representative view of the object
        view_meta_list = []

        # Iterate over top-N representative masks
        for view_idx, (frame_id, mask_id, coverage) in enumerate(obj_data["repre_mask_list"][:num_views]):
            try:
                rgb = dataset.get_rgb(frame_id, change_color=False)
                if rgb is None:
                    if debug:
                        print(
                            f"Could not load RGB image for object {obj_id}, frame {frame_id}")
                    continue

                base_filename = (
                    f"obj{obj_id:04d}_v{view_idx}_f{frame_id:04d}_m{mask_id:02d}")

                # All output paths for this view
                paths = {
                    "original": os.path.join(best_views_dir, f"{base_filename}.jpg"),
                    "highlighted": os.path.join(
                        best_views_dir, f"{base_filename}_highlighted.jpg"),
                }

                # Build metadata (relative paths)
                metadata = {
                    "frame_id": frame_id,
                    "mask_id": mask_id,
                    "coverage": coverage,
                    "view_idx": view_idx,
                }
                for key, path in paths.items():
                    metadata[f"{key}_path"] = os.path.relpath(path, scene_root)

                # Write images to disk
                cv2.imwrite(paths["original"], rgb)

                # Highlighted versions
                try:
                    mask, highlighted_rgb = create_highlighted_version(
                        dataset, frame_id, mask_id, rgb.copy())
                    cv2.imwrite(paths["highlighted"], highlighted_rgb)

                    if getattr(args, "crop_best_views", False) or getattr(
                        args, "best_view_crop", False):
                        bbox = calculate_padded_bounding_box(
                            mask, padding_ratio=getattr(args, "best_view_padding", 0.1))
                        if bbox is not None:
                            left, top, right, bottom = bbox
                            metadata["bbox"] = bbox

                    saved_count += 1
                    if debug and saved_count % 10 == 0:
                        print(f"Saved {saved_count} best view image sets so far")

                except Exception as e:
                    print(
                        f"Warning: Failed to process highlights for object {obj_id} view {view_idx}: {str(e)}")
                    continue

                # Append view metadata to list
                view_meta_list.append(metadata)

            except Exception as e:
                print(f"ERROR processing object {obj_id} view {view_idx}: {str(e)}")

        # Attach aggregated metadata to object_dict
        if view_meta_list:
            object_dict[obj_id]["views"] = view_meta_list
            # Backwards-compatibility shim
            object_dict[obj_id]["best_view"] = view_meta_list[0]

    if debug:
        print(f"====> Completed save_best_views: saved {saved_count} out of {len(object_dict)} objects")
 
    obj_dict_path = os.path.join(best_views_dir, 'best_view_object_dict.pth')
    try:
        torch.save(object_dict, obj_dict_path)
        if debug:
            print(f"Saved object dictionary to: {obj_dict_path}")
    except Exception as e:
        print(f"ERROR: Failed to save object dictionary: {str(e)}")

    # Return enriched object_dict
    return object_dict
