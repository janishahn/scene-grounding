import io
import base64
import logging
import os
import time
from typing import List, Union, Dict, Any, Optional
from PIL import Image
import torch
import ollama
import requests
from dotenv import load_dotenv

class VLMHandler:
    """
    Unified interface for image captioning backends.
    Supported backends: 'transformers', 'ollama', 'dam'.
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

        elif self.backend == "dam":
            # NVIDIA Describe Anything Model (DAM) – detailed mask-aware captioning
            import torch
            from transformers import AutoModel

            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

            dam_model = AutoModel.from_pretrained(
                model_name,
                trust_remote_code=True,
                torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
            ).to(device)

            # The DAM repository exposes a helper to build the captioning callable.
            # We retain the handle on the object for later usage in `caption_image`.
            self._dam = dam_model.init_dam(
                conv_mode="v1",
                prompt_mode="full+focal_crop",
            )

            # fall back values so that attribute always exists
            self._captioner = None

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

    def caption_image(self, image: Image.Image, prompt: str = None, mask: Optional[Image.Image] = None) -> str:
        """
        Generate a caption for a single image, returning a string.
        """
        if self.backend == "dam":
            # Detailed Localised Captioning requires a binary mask
            if mask is None:
                raise ValueError("Mask image must be provided when using the 'dam' backend.")

            if not prompt:
                prompt = "Describe the masked region in detail."

            # Ensure mandatory <image> token is included for DAM focal prompt
            if "<image>" not in prompt:
                prompt = "<image>\n" + prompt

            # DAM expects mask to be single-channel (L) PIL image with values {0,255}
            if mask.mode != "L":
                mask = mask.convert("L")

            try:
                res = self._dam.get_description(
                    image,
                    mask,
                    prompt,
                    streaming=False,
                    temperature=0.85,
                    top_p=0.95,
                    num_beams=3,
                    max_new_tokens=1024,
                    min_new_tokens=128,
                )
                
                if isinstance(res, str):
                    return res.strip()
                if isinstance(res, list):
                    return "".join(res).strip()
                # generator / iterator fallback
                return "".join(list(res)).strip()
            except Exception as e:
                logging.error(f"DAM captioning failed: {e}")
                return ""

        elif self.backend == "ollama":
            buf = io.BytesIO()
            image.save(buf, format="PNG")
            img_bytes = buf.getvalue()

            if not prompt:
                logging.warning("No captioning prompt passed, using default prompt.")
                prompt = "Describe this image in detail. **DO NOT OUTPUT ANYTHING OTHER THAN THE DESCRIPTION**"

            resp = ollama.generate(
                model=self.model_name,
                prompt=prompt,
                images=[img_bytes],
                options={"max_tokens": 1024, "temperature": 0.8}
            )
            
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
        
    def unload(self) -> bool:
        """
        Unload the model from memory.
        Returns True if unload was successful, False otherwise.
        """
        try:
            if self.backend == "ollama":
                # Use the Ollama API to explicitly unload the model
                import json
                import requests
                
                payload = {
                    "model": self.model_name,
                    "keep_alive": 0
                }
                
                response = requests.post(
                    "http://localhost:11434/api/generate",
                    json=payload
                )
                
                if response.status_code == 200:
                    resp_data = response.json()
                    if resp_data.get("done_reason") == "unload":
                        logging.info(f"Successfully unloaded model {self.model_name} from Ollama")
                        return True
                    else:
                        logging.warning(f"Unexpected response when unloading model: {resp_data}")
                        return False
                else:
                    logging.warning(f"Failed to unload model, status code: {response.status_code}")
                    return False
                    
            elif self.backend == "dam" or self.backend == "transformers":
                # For PyTorch models, delete the model and clear the cache
                if hasattr(self, '_dam') and self._dam is not None:
                    del self._dam
                    self._dam = None
                    
                if hasattr(self, '_captioner') and self._captioner is not None:
                    del self._captioner
                    self._captioner = None
                
                # Force garbage collection and clear CUDA cache
                import gc
                gc.collect()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                    
                logging.info(f"Cleared model from GPU memory")
                return True
                
            elif self.backend == "openrouter":
                # OpenRouter is API-based, so no explicit unloading needed
                logging.info("OpenRouter backend does not require explicit unloading")
                return True
                
            else:
                logging.warning(f"Unload not implemented for backend: {self.backend}")
                return False
                
        except Exception as e:
            logging.error(f"Error while unloading model: {e}")
            return False
