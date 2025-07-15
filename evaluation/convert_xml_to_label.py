import xml.etree.ElementTree as ET
import numpy as np
import torch
import open_clip
from open_clip import tokenizer
import os
import argparse
import sys
import json

# --- Add this block to fix the import path ---
# Get the absolute path of the project's root directory
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
# Add the project root to the Python path
if project_root not in sys.path:
    sys.path.insert(0, project_root)
# --- End of block ---

# Add the maskclustering path to import constants
# sys.path.append('/home/vlm_search/scene-grounding/maskclustering')
from maskclustering.evaluation.constants import SCANNETPP_LABELS, SCANNETPP_IDS

def rle_encode(mask):
    """Encode RLE (Run-length-encode) from 1D binary mask.

    Args:
        mask (np.ndarray): 1D binary mask
    Returns:
        rle (dict): encoded RLE
    """
    length = mask.shape[0]
    mask = np.concatenate([[0], mask, [0]])
    runs = np.where(mask[1:] != mask[:-1])[0] + 1
    runs[1::2] -= runs[::2]
    counts = ' '.join(str(x) for x in runs)
    rle = dict(length=length, counts=counts)
    return rle


def rle_decode(rle):
    """Decode rle to get binary mask.

    Args:
        rle (dict): rle of encoded mask
    Returns:
        mask (np.ndarray): decoded mask
    """
    length = rle['length']
    counts = rle['counts']
    s = counts.split()
    starts, nums = [np.asarray(x, dtype=np.int32) for x in (s[0:][::2], s[1:][::2])]
    starts -= 1
    ends = starts + nums
    mask = np.zeros(length, dtype=np.uint8)
    for lo, hi in zip(starts, ends):
        mask[lo:hi] = 1
    return mask

def load_clip():
    """Load CLIP model exactly like maskclustering does."""
    print(f'[INFO] loading CLIP model...')
    model, _, _ = open_clip.create_model_and_transforms("ViT-H-14", pretrained="laion2b_s32b_b79k")
    model.cuda()
    model.eval()
    print(f'[INFO] finish loading CLIP model...')
    return model

def create_label_mappings():
    """Create mappings between class names and IDs for ScanNet++."""
    label_to_id = {}
    id_to_label = {}
    
    # Create mapping from SCANNETPP_LABELS to SCANNETPP_IDS
    for i in range(min(len(SCANNETPP_IDS), len(SCANNETPP_LABELS))):
        class_id = SCANNETPP_IDS[i]
        class_label = SCANNETPP_LABELS[i]
        label_to_id[class_label] = class_id
        id_to_label[class_id] = class_label
    
    print(f"📋 Created mappings for {len(label_to_id)} classes")
    print(f"   Class ID range: {min(SCANNETPP_IDS)} to {max(SCANNETPP_IDS)}")
    
    return label_to_id, id_to_label

def extract_object_names_from_xml(xml_path):
    """Extract object names from XML file in the same order as masks."""
    tree = ET.parse(xml_path)
    root = tree.getroot()
    
    object_names = []
    object_ids = []
    
    # Extract objects in order (they should correspond to mask order)
    for obj in root.findall('object'):
        obj_id = obj.get('id')
        name_elem = obj.find('name')
        if name_elem is not None and name_elem.text:
            name = name_elem.text.strip().lower()
            object_names.append(name)
            object_ids.append(obj_id)
        else:
            # Handle objects without names
            object_names.append("unknown object")
            object_ids.append(obj_id)
    
    print(f"📝 Extracted {len(object_names)} object names from XML")
    return object_names, object_ids

def extract_text_features_batched(model, descriptions, batch_size=64):
    """Extract text features in batches to reduce GPU memory usage."""
    features = []
    for i in range(0, len(descriptions), batch_size):
        batch = descriptions[i:i+batch_size]
        text_tokens = tokenizer.tokenize(batch).cuda()
        with torch.no_grad():
            batch_features = model.encode_text(text_tokens).float()
            batch_features /= batch_features.norm(dim=-1, keepdim=True)
        features.append(batch_features.cpu())  # Move to CPU to free GPU memory
    features = torch.cat(features, dim=0)
    return features

def find_best_semantic_matches(object_embeddings, valid_class_embeddings, 
                             object_names, valid_classes, label_to_id):
    """Find best semantic matches between object names and valid classes."""
    
    # Compute cosine similarity matrix (same as maskclustering)
    similarity_matrix = torch.mm(object_embeddings, valid_class_embeddings.T)
    
    results = []
    
    for i, obj_name in enumerate(object_names):
        # Get similarities for this object
        similarities = similarity_matrix[i]
        
        # Find best match
        best_match_idx = similarities.argmax().item()
        best_score = similarities[best_match_idx].item()
        best_class = valid_classes[best_match_idx]
        
        # Convert to label ID
        if best_class in label_to_id:
            label_id = label_to_id[best_class]
        else:
            # Fallback to 'object' class (ID 15 in SCANNETPP_IDS[15])
            label_id = SCANNETPP_IDS[15]  # This should be 'object'
        
        results.append({
            'object_name': obj_name,
            'best_match': best_class,
            'similarity': best_score,
            'label_id': label_id
        })
        
        # Print some examples
        if i < 10:  # Show first 10 matches
            print(f"  {obj_name:20} -> {best_class:20} (score: {best_score:.3f}, ID: {label_id})")
    
    print(f"\n📊 Matching Summary:")
    print(f"   Total: {len(results)}")
    
    return results

def save_predictions_for_evaluation(scene_id, output_dir, pred_masks, label_ids, similarity_scores):
    """
    Saves predictions in the format required by eval_instance.py.

    Args:
        scene_id (str): The ID of the scene.
        output_dir (str): The root directory to save predictions.
        pred_masks (np.ndarray): Boolean masks of shape (num_vertices, num_instances).
        label_ids (list): List of semantic label IDs for each instance.
        similarity_scores (list): List of confidence scores for each instance.
    """
    # Create the main output directory and the subdirectory for masks
    masks_dir = os.path.join(output_dir, "masks", scene_id)
    os.makedirs(masks_dir, exist_ok=True)

    # Path for the main prediction file (e.g., <output_dir>/<scene_id>.txt)
    prediction_file_path = os.path.join(output_dir, f"{scene_id}.txt")
    
    num_instances = pred_masks.shape[1]
    
    with open(prediction_file_path, 'w') as f:
        for i in range(num_instances):
            instance_mask = pred_masks[:, i]
            
            # Ensure mask is boolean for RLE encoding
            if instance_mask.dtype != bool:
                instance_mask = instance_mask.astype(bool)

            # RLE encode the mask
            rle = rle_encode(instance_mask)

            # Define the path for the individual mask file
            mask_filename = f"instance_{i}.json"
            relative_mask_path = os.path.join("masks", scene_id, mask_filename)
            absolute_mask_path = os.path.join(output_dir, relative_mask_path)

            # Save the RLE-encoded mask to its JSON file
            with open(absolute_mask_path, 'w') as mask_file:
                json.dump(rle, mask_file)

            # Write the entry to the main prediction file
            label_id = label_ids[i]
            confidence = similarity_scores[i]
            # Clip confidence score to be within [0, 1] to avoid assertion errors
            confidence = np.clip(confidence, 0, 1)
            f.write(f"{relative_mask_path} {label_id} {confidence}\n")

    print(f"\n💾 Saved predictions for evaluation to {output_dir}")
    print(f"   - Main prediction file: {prediction_file_path}")
    print(f"   - Masks saved in: {masks_dir}")

def convert_xml_to_semantic_predictions(xml_path, class_agnostic_npz_path, output_dir, device='cuda'):
    """
    Convert XML object names to semantic predictions using CLIP similarity.
    Saves results in the format required for the official evaluation script.
    """
    
    print(f"🚀 Converting XML to semantic predictions")
    print(f"   XML: {xml_path}")
    print(f"   Input NPZ: {class_agnostic_npz_path}")
    print(f"   Output Dir: {output_dir}")
    print("=" * 80)
    
    # Create label mappings
    label_to_id, id_to_label = create_label_mappings()
    
    # Load CLIP model exactly like maskclustering
    model = load_clip()
    
    # Extract object names from XML
    print("📄 Extracting object names from XML...")
    object_names, object_ids = extract_object_names_from_xml(xml_path)
    
    # Load existing class-agnostic predictions
    print("📥 Loading class-agnostic predictions...")
    class_agnostic_data = np.load(class_agnostic_npz_path)
    pred_masks = class_agnostic_data['pred_masks']
    
    print(f"   Masks shape: {pred_masks.shape}")
    
    # Verify dimensions match
    num_instances_masks = pred_masks.shape[1]
    num_instances_xml = len(object_names)
    
    if num_instances_masks != num_instances_xml:
        print(f"⚠️  WARNING: Mask count ({num_instances_masks}) != XML object count ({num_instances_xml})")
        raise ValueError(
            f"Mask count ({num_instances_masks}) does not match XML object count ({num_instances_xml})"
        )
    else:
        print(f"✅ Dimensions match: {num_instances_masks} instances")
    
    # Extract text features for object names using maskclustering's method
    print("🧠 Extracting CLIP text features for object names...")
    object_embeddings = extract_text_features_batched(model, object_names, batch_size=32)
    print(f"   Object embeddings shape: {object_embeddings.shape}")
    
    # Extract text features for ScanNet++ class labels using maskclustering's method
    print("🧠 Extracting CLIP text features for ScanNet++ class labels...")
    valid_class_embeddings = extract_text_features_batched(model, SCANNETPP_LABELS, batch_size=32)
    print(f"   Class embeddings shape: {valid_class_embeddings.shape}")
    
    # Find best matches
    print("🎯 Finding best semantic matches...")
    matches = find_best_semantic_matches(
        object_embeddings, 
        valid_class_embeddings, 
        object_names, 
        SCANNETPP_LABELS, 
        label_to_id,
    )
    
    # Extract label IDs and similarity scores
    label_ids = [match['label_id'] for match in matches]
    similarity_scores = [match['similarity'] for match in matches]  # Use CLIP similarities as scores
    
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Save predictions in the evaluation format
    scene_id = os.path.splitext(os.path.basename(xml_path))[0]
    save_predictions_for_evaluation(
        scene_id,
        output_dir, # Pass the explicit output directory
        pred_masks,
        label_ids,
        similarity_scores
    )
    
    # The original np.savez can be removed or commented out
    # np.savez(output_path,
    #          pred_masks=pred_masks,
    #          pred_score=np.array(similarity_scores, dtype=np.float32),
    #          label_id=np.array(label_ids, dtype=np.int32))
    
    print(f"\n💾 Semantic predictions for scene {os.path.splitext(os.path.basename(xml_path))[0]} processed.")
    
    # Print statistics
    unique_labels, counts = np.unique(label_ids, return_counts=True)
    print(f"\n📊 Label distribution:")
    for label, count in zip(unique_labels, counts):
        if label in id_to_label:
            class_name = id_to_label[label]
            print(f"   {class_name} ({label}): {count}")
        else:
            print(f"   UNKNOWN_ID ({label}): {count}")
    
    # Print similarity statistics
    print(f"\n📈 CLIP Similarity Score Statistics (our confidence scores):")
    print(f"   Mean: {np.mean(similarity_scores):.3f}")
    print(f"   Min: {np.min(similarity_scores):.3f}")
    print(f"   Max: {np.max(similarity_scores):.3f}")
    print(f"   Std: {np.std(similarity_scores):.3f}")
    
    # Show distribution of confidence levels
    high_conf = np.sum(np.array(similarity_scores) >= 0.5)
    med_conf = np.sum((np.array(similarity_scores) >= 0.3) & (np.array(similarity_scores) < 0.5))
    low_conf = np.sum(np.array(similarity_scores) < 0.3)
    print(f"\n🎯 Confidence Distribution:")
    print(f"   High confidence (≥0.5): {high_conf} ({100*high_conf/len(similarity_scores):.1f}%)")
    print(f"   Medium confidence (0.3-0.5): {med_conf} ({100*med_conf/len(similarity_scores):.1f}%)")
    print(f"   Low confidence (<0.3): {low_conf} ({100*low_conf/len(similarity_scores):.1f}%)")
    
    return matches

def main():
    parser = argparse.ArgumentParser(description='Convert XML predictions to semantic format using CLIP')
    parser.add_argument('--scene_id', default='95d525fbfd', help='Scene ID to process')
    parser.add_argument('--xml_path', help='Path to XML file (if not provided, uses scene_id)')
    parser.add_argument('--class_agnostic_path', help='Path to class-agnostic .npz file (if not provided, uses scene_id)')
    parser.add_argument('--output_dir', help='Path to save the prediction files for evaluation (if not provided, uses a default)')
    
    args = parser.parse_args()
    
    # Set default paths based on scene_id if not provided
    if not args.xml_path:
        args.xml_path = f"/home/vlm_search/scene-grounding/vlm_caption/outputs/{args.scene_id}.xml"
    
    if not args.class_agnostic_path:
        args.class_agnostic_path = f"/home/vlm_search/scene-grounding/maskclustering/data/prediction/scannetpp_class_agnostic/{args.scene_id}.npz"
    
    if not args.output_dir:
        args.output_dir = f"/home/vlm_search/scene-grounding/maskclustering/data/prediction/scannetpp_vlm_caption/"
    
    # Convert predictions
    print(f"🎬 Starting conversion for scene: {args.scene_id}")
    matches = convert_xml_to_semantic_predictions(
        args.xml_path,
        args.class_agnostic_path,
        args.output_dir
    )
    
    print("\n🎉 Conversion completed successfully!")
    print(f"📁 Output saved to: {args.output_dir}")

if __name__ == "__main__":
    main()