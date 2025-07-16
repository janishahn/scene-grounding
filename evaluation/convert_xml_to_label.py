import xml.etree.ElementTree as ET
import numpy as np
import torch
import open_clip
from open_clip import tokenizer
import os
import argparse
import json


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
            label_id = label_to_id[12]  # This should be 'object'
        
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

def load_semantic_classes(semantic_classes_path):
    """Load instance classes from a text file."""
    with open(semantic_classes_path) as f:
        class_names = [line.strip() for line in f if line.strip()]
    label_to_id = {name: idx for idx, name in enumerate(class_names)}
    id_to_label = {idx: name for idx, name in enumerate(class_names)}
    return class_names, label_to_id, id_to_label

def convert_xml_to_semantic_predictions(xml_path, class_agnostic_npz_path, output_dir, semantic_classes_path, device='cuda'):
    """
    Convert XML object names to semantic predictions using CLIP similarity.
    Uses instance_classes.txt for valid classes.
    """
    print(f"🚀 Converting XML to semantic predictions")
    print(f"   XML: {xml_path}")
    print(f"   Input NPZ: {class_agnostic_npz_path}")
    print(f"   Output Dir: {output_dir}")
    print(f"   Instance Classes: {semantic_classes_path}")
    print("=" * 80)

    # Load instance classes
    valid_classes, label_to_id, id_to_label = load_semantic_classes(semantic_classes_path)

    # Load CLIP model
    model = load_clip()

    # Extract object names from XML
    print("📄 Extracting object names from XML...")
    object_names, object_ids = extract_object_names_from_xml(xml_path)

    # Load class-agnostic predictions
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

    # Extract text features for object names
    print("🧠 Extracting CLIP text features for object names...")
    object_embeddings = extract_text_features_batched(model, object_names, batch_size=32)
    print(f"   Object embeddings shape: {object_embeddings.shape}")

    # Extract text features for instance class labels
    print("🧠 Extracting CLIP text features for instance class labels...")
    valid_class_embeddings = extract_text_features_batched(model, valid_classes, batch_size=32)
    print(f"   Class embeddings shape: {valid_class_embeddings.shape}")

    # Find best matches
    print("🎯 Finding best semantic matches...")
    matches = find_best_semantic_matches(
        object_embeddings,
        valid_class_embeddings,
        object_names,
        valid_classes,
        label_to_id,
    )

    # Extract label IDs and similarity scores
    label_ids = [match['label_id'] for match in matches]
    similarity_scores = [match['similarity'] for match in matches]

    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)

    # Save predictions in the evaluation format
    scene_id = os.path.splitext(os.path.basename(xml_path))[0]
    save_predictions_for_evaluation(
        scene_id,
        output_dir,
        pred_masks,
        label_ids,
        similarity_scores
    )

    # Print statistics
    unique_labels, counts = np.unique(label_ids, return_counts=True)
    print(f"\n📊 Label distribution:")
    for label, count in zip(unique_labels, counts):
        if label in id_to_label:
            class_name = id_to_label[label]
            print(f"   {class_name} ({label}): {count}")
        else:
            print(f"   UNKNOWN_ID ({label}): {count}")

    print(f"\n📈 CLIP Similarity Score Statistics (our confidence scores):")
    print(f"   Mean: {np.mean(similarity_scores):.3f}")
    print(f"   Min: {np.min(similarity_scores):.3f}")
    print(f"   Max: {np.max(similarity_scores):.3f}")
    print(f"   Std: {np.std(similarity_scores):.3f}")

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
    parser.add_argument('--scene_id', help='Single scene ID to process (alternative to --scene_list)')
    parser.add_argument('--scene_list', default="/home/vlm_search/scene-grounding/maskclustering/data/scannetpp/splits/scene_grounding.txt", help='Path to text file containing scene IDs (one per line)')
    parser.add_argument('--xml_path', help='Path to XML file (only used with --scene_id)')
    parser.add_argument('--class_agnostic_path', help='Path to class-agnostic .npz file (only used with --scene_id)')
    parser.add_argument('--output_dir', help='Path to save the prediction files for evaluation (if not provided, uses a default)')
    parser.add_argument('--semantic_classes_path', help='Path to instance_classes.txt')

    args = parser.parse_args()

    # Check that either scene_id or scene_list is provided, but not both
    if args.scene_id and args.scene_list:
        raise ValueError("Please provide either --scene_id or --scene_list, not both")
    if not args.scene_id and not args.scene_list:
        raise ValueError("Please provide either --scene_id or --scene_list")

    # Set default output directory
    if not args.output_dir:
        args.output_dir = f"/home/vlm_search/scene-grounding/maskclustering/data/prediction/scannetpp_vlm_caption/"

    if not args.semantic_classes_path:
        args.semantic_classes_path = f"/home/vlm_search/scene-grounding/maskclustering/data/scannetpp/metadata/semantic_classes.txt"

    # Get list of scene IDs to process
    if args.scene_list:
        print(f"📋 Reading scene IDs from: {args.scene_list}")
        with open(args.scene_list, 'r') as f:
            scene_ids = [line.strip() for line in f if line.strip()]
        print(f"📊 Found {len(scene_ids)} scenes to process")
    else:
        scene_ids = [args.scene_id]
        print(f"🎬 Processing single scene: {args.scene_id}")

    # Process each scene
    successful_conversions = 0
    failed_conversions = 0
    
    for i, scene_id in enumerate(scene_ids):
        print(f"\n{'='*80}")
        print(f"🎬 Processing scene {i+1}/{len(scene_ids)}: {scene_id}")
        print(f"{'='*80}")
        
        try:
            # Set paths for this scene
            xml_path = args.xml_path if args.xml_path else f"/home/vlm_search/scene-grounding/vlm_caption/outputs/{scene_id}.xml"
            class_agnostic_path = args.class_agnostic_path if args.class_agnostic_path else f"/home/vlm_search/scene-grounding/maskclustering/data/prediction/scannetpp_class_agnostic/{scene_id}.npz"
            
            # Convert predictions for this scene
            matches = convert_xml_to_semantic_predictions(
                xml_path,
                class_agnostic_path,
                args.output_dir,
                args.semantic_classes_path
            )
            
            print(f"✅ Successfully processed scene: {scene_id}")
            successful_conversions += 1
            
        except Exception as e:
            print(f"❌ Failed to process scene {scene_id}: {str(e)}")
            failed_conversions += 1
            continue

    # Print final summary
    print(f"\n{'='*80}")
    print(f"🎉 Batch conversion completed!")
    print(f"📊 Summary:")
    print(f"   ✅ Successful: {successful_conversions}")
    print(f"   ❌ Failed: {failed_conversions}")
    print(f"   📁 Output saved to: {args.output_dir}")
    print(f"{'='*80}")

if __name__ == "__main__":
    main()