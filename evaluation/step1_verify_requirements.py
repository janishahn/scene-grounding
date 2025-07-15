import xml.etree.ElementTree as ET
import os

def examine_xml_structure(scene_id="95d525fbfd"):
    """Examine the XML file structure and object names."""
    
    xml_path = f'/home/vlm_search/scene-grounding/vlm_caption/outputs/{scene_id}.xml'
    
    print(f"🔍 Examining XML file: {xml_path}")
    print("=" * 60)
    
    if not os.path.exists(xml_path):
        print(f"❌ XML file not found!")
        return None
    
    # Parse XML
    tree = ET.parse(xml_path)
    root = tree.getroot()
    
    # Extract object information
    objects = []
    for obj in root.findall('object'):
        obj_id = obj.get('id')
        name_elem = obj.find('name')
        name = name_elem.text.strip() if name_elem is not None and name_elem.text else "NO_NAME"
        objects.append({'id': obj_id, 'name': name})
    
    print(f"📊 Found {len(objects)} objects in XML")
    print("\n🏷️  First 10 object names:")
    for i, obj in enumerate(objects[:10]):
        print(f"  {obj['id']}: '{obj['name']}'")
    
    # Analyze name patterns
    all_names = [obj['name'].lower() for obj in objects]
    unique_names = list(set(all_names))
    
    print(f"\n📈 Statistics:")
    print(f"  Total objects: {len(objects)}")
    print(f"  Unique names: {len(unique_names)}")
    print(f"  Objects without names: {sum(1 for name in all_names if name == 'no_name')}")
    
    print(f"\n🔤 Most common object types (first 15):")
    from collections import Counter
    name_counts = Counter(all_names)
    for name, count in name_counts.most_common(15):
        print(f"  '{name}': {count}")
    
    print(f"\n📝 Sample of unique names (first 20):")
    for name in sorted(unique_names)[:20]:
        print(f"  '{name}'")
    
    return objects

# Run the XML examination
xml_objects = examine_xml_structure("95d525fbfd")