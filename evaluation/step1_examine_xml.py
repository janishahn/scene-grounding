import os
from pathlib import Path

def verify_step1_requirements(scene_id="95d525fbfd"):
    """Verify all required files exist for Step 1."""
    
    print(f"🔍 Verifying requirements for scene: {scene_id}")
    print("=" * 50)
    
    # Required files
    files_to_check = {
        "Class-agnostic predictions": f'/home/vlm_search/scene-grounding/maskclustering/data/prediction/scannetpp_class_agnostic/{scene_id}.npz',
        "XML object descriptions": f'/home/vlm_search/scene-grounding/vlm_caption/outputs/{scene_id}.xml',
        "Constants file": '/home/vlm_search/scene-grounding/maskclustering/evaluation/constants.py'
    }
    
    all_good = True
    
    for desc, filepath in files_to_check.items():
        if os.path.exists(filepath):
            size = os.path.getsize(filepath)
            print(f"✅ {desc}")
            print(f"   📁 {filepath}")
            print(f"   📏 Size: {size:,} bytes")
        else:
            print(f"❌ {desc}")
            print(f"   📁 {filepath}")
            print(f"   🚫 File not found!")
            all_good = False
        print()
    
    # Check directories exist
    dirs_to_check = {
        "Output directory (will create)": '/home/vlm_search/scene-grounding/maskclustering/data/prediction/scannetpp_semantic'
    }
    
    for desc, dirpath in dirs_to_check.items():
        if os.path.exists(dirpath):
            print(f"✅ {desc}")
            print(f"   📁 {dirpath}")
        else:
            print(f"⚠️  {desc}")
            print(f"   📁 {dirpath}")
            print(f"   💡 Will be created automatically")
        print()
    
    if all_good:
        print("🎉 All required files found! Ready to proceed to Step 2.")
    else:
        print("⚠️  Some required files are missing. Please check the paths above.")
    
    return all_good

# Run the verification
requirements_ok = verify_step1_requirements("95d525fbfd")