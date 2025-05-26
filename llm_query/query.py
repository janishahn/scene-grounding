import json
import logging
import yaml
import re
from ollama import generate
from torch import load

def query_scene(captions_path: str):
    """
    Query a scene using captions and natural language input to identify a specific object.
    This function processes scene captions, takes a user query, and uses an LLM to identify 
    the most relevant object in the scene. It then returns the path to an image highlighting
    that object.

    Args:
        captions_path (str): Path to the JSON file containing object captions.
    
    Returns:
        str or None: Path to the highlighted image of the identified object, or None if
                     no object could be identified from the LLM response.
    """
    
    logging.info(f"Querying scene with captions from {captions_path}...")
    # Get the configs from query.yaml
    with open("llm_query/query.yaml", 'r') as f:
        query_config = yaml.safe_load(f)

    model_name = query_config.get("model", "")

    # Load captions and get the query from the user
    with open(captions_path, 'r') as f:
        captions = json.load(f)
    # Reduce captions to only the cropped version
    lean_captions = {}
    for id, val in captions.items():
        lean_captions[id] = val['captions']['cropped']['text']

    # Get user input and build prompt
    query = input("Please enter your query here: ")
    system_prompt = build_system_prompt(lean_captions)
    combined_prompt = f"{system_prompt}\n\nThe user query, describing the object the user is looking for: {query}"

    # Call LLM with prompt via ollama
    logging.info("Querying LLM to find best object description for query...")
    response = generate(
        model=model_name,
        prompt=combined_prompt
    )

    logging.info("The LLM has returned the following response:")
    logging.info(response.response)

    # Extract object_id from the response with regex
    match = re.search(r'"object_id":\s*"(\w+)"', response.response)
    if match:
        # Load obj dict entry
        object_id = match.group(1)
        obj_dict_path = query_config.get('obj_dict_path', "")
        obj_dict = load(obj_dict_path, weights_only=False)
        # Extract img_path for the highlighted picture
        img_path = obj_dict[int(object_id)]['best_view']['highlighted_path']
        logging.error("No object_id found in the LLM response.")
        return None
    
    logging.info("Successfully identified object, returning image path.")
    return img_path

def build_system_prompt(captions: dict) -> str:
    system_prompt = """You are an expert object identification assistant. Your task is to find the object that best matches a user's query from a collection of described objects.

    You will receive a JSON containing object descriptions. Each object has two captions:
    - "cropped": Description of the object in isolation  
    - "original": Description of the object within its surrounding context

    You must analyze the user's query, compare it against ALL object descriptions, and return ONLY the object ID and reasoning.

    CRITICAL: Your response must be EXACTLY in this format (no additional text, explanations, or formatting):
   
    {
    "object_id": "X",
    "reasoning": "Brief explanation of why this object matches the query"
    }

    Available objects:
    """ + json.dumps(captions, indent=2)
    
    return system_prompt