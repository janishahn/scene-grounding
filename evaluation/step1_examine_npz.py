import numpy as np
import os

def examine_existing_npz(scene_id="95d525fbfd"):
    """Examine the structure of your existing class-agnostic .npz file."""
    
    npz_path = f'/home/vlm_search/scene-grounding/maskclustering/data/prediction/scannetpp_vlm_caption/95d525fbfd.npz'
    
    print(f"Examining: {npz_path}")
    print("=" * 60)
    
    if not os.path.exists(npz_path):
        print(f"❌ File not found: {npz_path}")
        return None
    
    # Load the data
    data = np.load(npz_path)
    
    print("📁 Keys in .npz file:")
    for key in data.keys():
        print(f"  - {key}")
    
    print("\n📊 Data structure:")
    for key in data.keys():
        array = data[key]
        print(f"  {key}:")
        print(f"    Shape: {array.shape}")
        print(f"    Dtype: {array.dtype}")
        print(f"    Min: {array.min()}, Max: {array.max()}")
        
        # Show sample values
        if len(array.shape) == 1 and len(array) <= 10:
            print(f"    Values: {array}")
        elif len(array.shape) == 1:
            print(f"    First 5: {array[:5]}")
        elif len(array.shape) == 2:
            print(f"    Sample from first instance: {array[:5, 0] if array.shape[1] > 0 else 'No instances'}")
    
    print("\n🔍 Analysis:")
    
    if 'pred_masks' in data:
        masks = data['pred_masks']
        num_points = masks.shape[0]
        num_instances = masks.shape[1]
        print(f"  📍 Number of 3D points: {num_points:,}")
        print(f"  🎯 Number of predicted instances: {num_instances}")
        
        # Analyze mask coverage
        points_covered = np.any(masks, axis=1).sum()
        coverage_percent = (points_covered / num_points) * 100
        print(f"  📈 Points covered by at least one mask: {points_covered:,} ({coverage_percent:.1f}%)")
        
        # Analyze instance sizes
        instance_sizes = np.sum(masks, axis=0)
        print(f"  📏 Instance sizes - Min: {instance_sizes.min()}, Max: {instance_sizes.max()}, Mean: {instance_sizes.mean():.1f}")
    
    if 'pred_score' in data:
        scores = data['pred_score']
        print(f"  🎯 Confidence scores - Min: {scores.min():.3f}, Max: {scores.max():.3f}, Mean: {scores.mean():.3f}")
        print(f"      Unique values: {np.unique(scores)}")
    
    if 'pred_classes' in data:
        classes = data['pred_classes']
        print(f"  🏷️  Current classes - Unique: {np.unique(classes)}")
    
    return data

# Run the examination
data = examine_existing_npz("95d525fbfd")