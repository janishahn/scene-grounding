import torch
from transformers import pipeline
from typing import Union, List
from PIL import Image

def get_image_caption_pipeline(model_name: str, quantize: bool = False):
    """
    Create and return a HuggingFace pipeline for image captioning.
    
    Parameters
    ----------
    model_name : str
        The name or path of the model to load (e.g., 'Salesforce/blip2-opt-2.7b').
    quantize : bool, optional
        Whether to use 8-bit quantization to reduce memory usage.
        Default is False.
    
    Returns
    -------
    pipeline
        A HuggingFace pipeline object for image-to-text task.
    """
    return pipeline(
        "image-to-text",
        model=model_name,
        device_map="auto",
        torch_dtype=torch.float16,
        load_in_8bit=quantize,
    )

def generate_caption(
    pipeline,
    image: Union[Image.Image, List[Image.Image]],
    max_new_tokens: int = 50,
    num_beams: int = 5,
    min_length: int = 5,
    top_p: float = 0.9,
    repetition_penalty: float = 1.5,
    temperature: float = 1.0,
) -> Union[str, List[str]]:
    """
    Generate captions for the given images.
    
    Parameters
    ----------
    pipeline
        HuggingFace pipeline for image-to-text task.
    image : Union[Image.Image, List[Image.Image]]
        The image(s) to generate captions for.
    max_new_tokens : int, optional
        Maximum number of tokens to generate. Default is 50.
    num_beams : int, optional
        Number of beams for beam search. Default is 5.
    min_length : int, optional
        Minimum length of the generated caption. Default is 5.
    top_p : float, optional
        Top p sampling parameter. Default is 0.9.
    repetition_penalty : float, optional
        Penalty for repetition. Default is 1.5.
    temperature : float, optional
        Temperature for sampling. Default is 1.0.
    
    Returns
    -------
    Union[str, List[str]]
        The generated caption(s).
    """
    result = pipeline(
        image,
        max_new_tokens=max_new_tokens,
        num_beams=num_beams,
        min_length=min_length,
        top_p=top_p,
        repetition_penalty=repetition_penalty,
        temperature=temperature,
    )
    
    # Handle both single image and batch mode
    if isinstance(image, list):
        return [item[0]["generated_text"] for item in result]
    else:
        return result[0]["generated_text"]
