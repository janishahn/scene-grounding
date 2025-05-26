import json
import logging
import ollama


PROMPT = """Something"""


def query_scene(captions_path: str):
    logging.info(f"Querying scene with captions from {captions_path}...")
    # 1. Load the captions 
    # 2. Create a prompt that specifies the task of the LLM: Take all descriptions and the query, and output the 
    # description + object_id that best fit the query.
    # 3. Call a small LLM via ollama with the propmt + descriptions
    # 3. Given the object_id, get the highlighted. best-view picture of the object and return it.
    
    # Load captions and get the query from the user
    with open(captions_path, 'r') as f:
        captions = json.load(f)

    query = input("Please enter your query here: ")
    prompt = build_prompt(query, captions)

    logging.info("Querying LLM to find best object description for query...")
    # Call LLM with prompt via ollama


    
    img_path = ""
    return img_path

def build_prompt(query: str, captions: dict) -> str:
    pass