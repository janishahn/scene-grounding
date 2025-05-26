import json
import logging
import yaml
from ollama import generate

def query_scene(captions_path: str):
    logging.info(f"Querying scene with captions from {captions_path}...")
    
    # Get the configs from query.yaml
    with open("llm_query/query.yaml", 'r') as f:
        query_config = yaml.safe_load(f)

    model_name = query_config.get("model", "")

    # Load captions and get the query from the user
    with open(captions_path, 'r') as f:
        captions = json.load(f)

    query = input("Please enter your query here: ")
    system_prompt = build_system_prompt(captions)

    logging.info("Querying LLM to find best object description for query...")
    # Call LLM with prompt via ollama
    combined_prompt = f"{system_prompt}\n\nUser Query: {query}"
    response = generate(
        model=model_name,
        prompt=combined_prompt
    )

    # Given the object id and the description pull the correct entry from the object dict
    
    img_path = ""
    return img_path

def build_system_prompt(captions: dict) -> str:
    system_prompt = """You are an expert object identification assistant. You will be provided with a JSON containing descriptions of objects found in a 3D scene. Each object has two types of captions:
    - "cropped": Description of the object in isolation
    - "original": Description of the object within its surrounding context

    Example object entry:
    ```json
    {
    "0": {
        "captions": {
        "cropped": {
            "text": "A solid white door with a brushed nickel door handle and plate. To the right of the door is a blue tiled wall, and to the left is a gray wall. A pink rectangular sticker is adhered to the door.",
            "img_path": "maskclustering/data/scannetpp/data/88cf747085/output/best_views/obj0001_f6510_m05_cropped.jpg"
        },
        "original": {
            "text": "A plain white door is the primary focus. It has a brushed metal handle and a rectangular, recessed door viewer. To the right of the door is a blue tiled wall with vertical chrome towel bars.",
            "img_path": "maskclustering/data/scannetpp/data/88cf747085/output/best_views/obj0001_f6510_m05.jpg"
        }
        }
    }
    }
    ```

    Your task is to:
    1. Analyze the user's query
    2. Compare it against both cropped and original captions for all objects
    3. Identify the object that best matches the query
    4. Return the complete object entry as JSON along with your reasoning

    Response format:
    ```json
    {
    "reasoning": "Brief explanation of why this object matches the query",
    "selected_object": {
        // Complete object entry from the input JSON
    }
    }
    ```

    Available objects:
    """ + json.dumps(captions, indent=2)
    
    return system_prompt