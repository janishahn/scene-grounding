import io
import base64
import logging
import os
import time
from typing import List, Union, Dict, Any
from PIL import Image
import torch
import ollama
import requests
from dotenv import load_dotenv

class VLMHandler:
    """
    Unified interface for image captioning backends.
    Supported backends: 'transformers', 'ollama'.
    """
    def __init__(self, model_name: str, backend: str = "transformers", quantize: bool = False):
        self.model_name = model_name
        self.backend = backend.lower()
        self.quantize = quantize

        if self.backend == "transformers":
            from transformers import pipeline, AutoTokenizer
            
            hf_kwargs = {
                "model": model_name,
                "device_map": "auto",
            }
            
            if quantize:
                from transformers import BitsAndBytesConfig, AutoImageProcessor
                
                quantization_config = BitsAndBytesConfig(
                    load_in_8bit=True,
                    llm_int8_threshold=6.0,
                    llm_int8_has_fp16_weight=False,
                )
                
                from transformers import AutoConfig
                
                config = AutoConfig.from_pretrained(model_name)
                model_type = type(config).__name__
                
                if "Blip" in model_type:
                    from transformers import AutoModelForVision2Seq
                    model = AutoModelForVision2Seq.from_pretrained(
                        model_name,
                        device_map="auto",
                        quantization_config=quantization_config,
                        torch_dtype=torch.float16
                    )
                else:
                    from transformers import AutoModel
                    model = AutoModel.from_pretrained(
                        model_name,
                        device_map="auto",
                        quantization_config=quantization_config,
                        torch_dtype=torch.float16
                    )
                
                image_processor = AutoImageProcessor.from_pretrained(model_name)
                tokenizer = AutoTokenizer.from_pretrained(model_name)
                
                self._captioner = pipeline(
                    "image-to-text", 
                    model=model, 
                    image_processor=image_processor, 
                    tokenizer=tokenizer
                )
            else:
                self._captioner = pipeline("image-to-text", **hf_kwargs)

        elif self.backend == "ollama":
            logging.getLogger("ollama").setLevel(logging.WARNING)
            logging.getLogger("httpx").setLevel(logging.WARNING)
        elif self.backend == "openrouter":
            logging.getLogger("httpx").setLevel(logging.WARNING)
        else:
            raise ValueError(f"Unsupported backend: {backend}")

    def _get_openrouter_headers(self) -> Dict[str, str]:
        load_dotenv()
        api_key = os.getenv("OPENROUTER_API_KEY")
        if not api_key:
            raise ValueError("OPENROUTER_API_KEY not found in .env file")
        
        return {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json"
        }

    def _prepare_openrouter_request_body(self, prompt: str, image_b64: str) -> Dict[str, Any]:
        return {
            "model": self.model_name,
            "messages": [
                {
                    "role": "user", 
                    "content": [
                        {"type": "text", "text": prompt},
                        {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{image_b64}"}}
                    ]
                }
            ],
            "temperature": 0.8,
            "max_tokens": 1024
        }

    def caption_image(self, image: Image.Image, prompt: str = None) -> str:
        """
        Generate a caption for a single image, returning a string.
        """
        if self.backend == "ollama":
            buf = io.BytesIO()
            image.save(buf, format="PNG")
            b64 = base64.b64encode(buf.getvalue()).decode()

            if not prompt:
                logging.warning("No captioning prompt passed, using default prompt.")
                prompt = "Describe this image in detail. **DO NOT OUTPUT ANYTHING OTHER THAN THE DESCRIPTION**"

            resp = ollama.generate(model=self.model_name, prompt=prompt, images=[b64], options={"max_tokens": 250, "temperature": 0.8})
            
            if isinstance(resp, dict):
                if "message" in resp and "content" in resp["message"]:
                    return resp["message"]["content"].strip()
                elif "response" in resp:
                    return resp["response"].strip()
                elif "content" in resp:
                    return resp["content"].strip()
            else:
                for attr in ["response", "content"]:
                    if hasattr(resp, attr):
                        return getattr(resp, attr).strip()
                        
                if hasattr(resp, "response") and hasattr(resp.response, "content"):
                    return resp.response.content.strip()
                
                return str(resp).strip()
        
        elif self.backend == "transformers":
            raw = self._captioner(image)
            
            if isinstance(raw, list) and raw and isinstance(raw[0], dict) and "generated_text" in raw[0]:
                return raw[0]["generated_text"].strip()
            elif isinstance(raw, dict) and "generated_text" in raw:
                return raw["generated_text"].strip()
            elif isinstance(raw, list) and raw and isinstance(raw[0], str):
                return raw[0].strip()
            else:
                return str(raw).strip()
            
        elif self.backend == "openrouter":
            buf = io.BytesIO()
            image.save(buf, format="PNG")
            b64 = base64.b64encode(buf.getvalue()).decode()

            if not prompt:
                logging.warning("No captioning prompt passed, using default prompt.")
                prompt = "Describe this image in detail. **DO NOT OUTPUT ANYTHING OTHER THAN THE DESCRIPTION**"

            headers = self._get_openrouter_headers()
            body = self._prepare_openrouter_request_body(prompt, b64)
            
            for attempt in range(3):
                try:
                    response = requests.post(
                        "https://openrouter.ai/api/v1/chat/completions",
                        headers=headers,
                        json=body,
                        timeout=30
                    )
                    
                    if response.status_code == 500 and attempt < 2:
                        logging.warning(f"Server error 500, retrying in 3 seconds (attempt {attempt + 1}/3)")
                        time.sleep(3)
                        continue
                    
                    response.raise_for_status()
                    response_json = response.json()
                    
                    if "choices" in response_json and response_json["choices"]:
                        return response_json["choices"][0]["message"]["content"].strip()
                    else:
                        return "Error: No content in response"
                        
                except requests.exceptions.RequestException as e:
                    if attempt < 2:
                        logging.warning(f"Request failed, retrying in 3 seconds (attempt {attempt + 1}/3): {e}")
                        time.sleep(3)
                        continue
                    logging.error(f"OpenRouter API request failed after 3 attempts: {e}")
                    return "Error: API request failed"

    def caption_batch(self, images: List[Image.Image]) -> List[str]:
        """
        Generate a caption for each image in `images` sequentially, returning List[str].
        """
        return [self.caption_image(img) for img in images]
