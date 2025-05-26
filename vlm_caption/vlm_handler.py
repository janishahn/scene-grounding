import io
import base64
import logging
from typing import List, Union
from PIL import Image
import torch
import ollama

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
        else:
            raise ValueError(f"Unsupported backend: {backend}")

    def caption_image(self, image: Image.Image) -> str:
        """
        Generate a caption for a single image, returning a string.
        """
        if self.backend == "ollama":
            buf = io.BytesIO()
            image.save(buf, format="PNG")
            b64 = base64.b64encode(buf.getvalue()).decode()
            prompt = f"Describe this image in detail. **DO NOT OUTPUT ANYTHING OTHER THAN THE DESCRIPTION**"
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

    def caption_batch(self, images: List[Image.Image]) -> List[str]:
        """
        Generate a caption for each image in `images` sequentially, returning List[str].
        """
        return [self.caption_image(img) for img in images]
