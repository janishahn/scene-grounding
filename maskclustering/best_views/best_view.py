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

def create_highlighted_version(dataset, frame_id, mask_id, highlighted_rgb):
    """
    Creates a highlighted version of an image by overlaying a semi-transparent color
    on a specific segmentation mask and adding a border around the masked object.
    This function helps visualize a particular object in a frame by making it stand
    out with a colored highlight.
    """
    segmentation = dataset.get_segmentation(frame_id)
    mask = (segmentation == mask_id)
    
    # Apply highlight - semi-transparent overlay with border
    highlight_color = (0, 255, 0)  # Green highlight
    alpha = 0.4  # Transparency factor
    
    # Create color overlay
    overlay = highlighted_rgb.copy()
    overlay[mask] = highlight_color
    
    # Blend with original image
    highlighted_rgb = cv2.addWeighted(overlay, alpha, highlighted_rgb, 1 - alpha, 0)
    
    # Add border around the mask
    contours, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(highlighted_rgb, contours, -1, highlight_color, 2)

    return mask, highlighted_rgb


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
    
    # Process each object
    for obj_id, obj_data in object_dict.items():
        # Skip if no representative masks
        if 'repre_mask_list' not in obj_data or len(obj_data['repre_mask_list']) == 0:
            if debug:
                print(f"Object {obj_id} has no representative masks, skipping")
            continue
            
        # NOTE: We currently rely on MaskClustering to provide us with the most representative view of the object
        # Get best view info from first entry in repre_mask_list
        frame_id, mask_id, coverage = obj_data['repre_mask_list'][0]

        try:
            # Get RGB image
            rgb = dataset.get_rgb(frame_id, change_color=False)
            if rgb is None:
                if debug:
                    print(f"Could not load RGB image for object {obj_id}, frame {frame_id}")
                continue
                
            # Generate output filename using standardized format
            out_filename = f'obj{obj_id:04d}_f{frame_id:04d}_m{mask_id:02d}.jpg'
            out_path = os.path.join(best_views_dir, out_filename)
            
            # Generate highlighted version filename
            out_highlighted_filename = f'obj{obj_id:04d}_f{frame_id:04d}_m{mask_id:02d}_highlighted.jpg'
            out_highlighted_path = os.path.join(best_views_dir, out_highlighted_filename)
            
            # Initialize metadata
            metadata = {
                'frame_id': frame_id,
                'mask_id': mask_id, 
                'coverage': coverage,
                'image_path': os.path.relpath(out_path, scene_root),
                'highlighted_image_path': os.path.relpath(out_highlighted_path, scene_root)
            }

            # Create a copy of the original image for highlighting
            highlighted_rgb = rgb.copy()

            # Load segmentation mask to create the highlighted version
            try:
                mask, highlighted_rgb = create_highlighted_version(dataset, frame_id, mask_id, highlighted_rgb)
            except Exception as highlight_err:
                print(f"Warning: Failed to create highlighted image for object {obj_id}: {str(highlight_err)}")

            # Optional crop if enabled 
            if getattr(args, 'crop_best_views', False) or getattr(args, 'best_view_crop', False):
                try:
                    # Compute bbox
                    bbox = calculate_padded_bounding_box(mask, padding_ratio=getattr(args, 'best_view_padding', 0.1))
                    if bbox is not None:
                        left, top, right, bottom = bbox
                        # Crop image to according to bbox
                        rgb = rgb[top:bottom+1, left:right+1]
                        highlighted_rgb = highlighted_rgb[top:bottom+1, left:right+1]

                        metadata['bbox'] = bbox
                except Exception as crop_err:
                    print(f"Warning: Failed to crop object {obj_id}: {str(crop_err)}")
            
            # Save images
            try:
                cv2.imwrite(out_path, rgb)
                cv2.imwrite(out_highlighted_path, highlighted_rgb)
                
                # Debug information 
                saved_count += 1
                if debug and saved_count % 10 == 0:
                    print(f"Saved {saved_count} best view image pairs so far")

            except Exception as save_err:
                print(f"ERROR: Failed to save image for object {obj_id}: {str(save_err)}")
                continue
            
            # Attach metadata to object_dict
            object_dict[obj_id]['best_view'] = metadata
            
        except Exception as e:
            print(f"ERROR processing object {obj_id}: {str(e)}")
    
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
