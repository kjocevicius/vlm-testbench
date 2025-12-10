"""
Triton Inference Server client utilities for VLM testbench.
"""

import numpy as np
import tritonclient.http as httpclient
from PIL import Image
from pathlib import Path
from typing import Union
import io


class TritonVLMClient:
    def __init__(self, url: str = "localhost:8000", verbose: bool = False):
        self.url = url
        self.verbose = verbose
        self.client = httpclient.InferenceServerClient(url=url, verbose=verbose)
        
    def is_server_ready(self) -> bool:
        return self.client.is_server_ready()
    
    def is_model_ready(self, model_name: str) -> bool:
        return self.client.is_model_ready(model_name)
    
    def get_server_metadata(self):
        return self.client.get_server_metadata()
    
    def get_model_metadata(self, model_name: str):
        return self.client.get_model_metadata(model_name)
    
    def infer(self, model_name: str, image: Union[str, Path, Image.Image], prompt: str) -> str:
        if isinstance(image, (str, Path)):
            image = Image.open(image)
        
        img_byte_arr = io.BytesIO()
        image.save(img_byte_arr, format='PNG')
        image_bytes = img_byte_arr.getvalue()
        
        image_data = np.frombuffer(image_bytes, dtype=np.uint8)
        prompt_data = np.array([prompt], dtype=object)
        
        inputs = [
            httpclient.InferInput("image", image_data.shape, "UINT8"),
            httpclient.InferInput("prompt", prompt_data.shape, "BYTES"),
        ]
        
        inputs[0].set_data_from_numpy(image_data)
        inputs[1].set_data_from_numpy(prompt_data)
        
        outputs = [
            httpclient.InferRequestedOutput("generated_text")
        ]
        
        response = self.client.infer(
            model_name=model_name,
            inputs=inputs,
            outputs=outputs
        )
        
        output_data = response.as_numpy("generated_text")
        return output_data[0].decode("utf-8")
    
    def list_models(self):
        return self.client.get_model_repository_index()


def describe_image_moondream(image: Union[str, Path, Image.Image], 
                            prompt: str = "Describe this image in detail.",
                            url: str = "localhost:8000") -> str:
    client = TritonVLMClient(url=url)
    return client.infer("moondream2", image, prompt)


def describe_image_kosmos2(image: Union[str, Path, Image.Image], 
                          prompt: str = "<grounding>Describe this image in detail.",
                          url: str = "localhost:8000") -> str:
    client = TritonVLMClient(url=url)
    return client.infer("kosmos2", image, prompt)


def describe_image_llava(image: Union[str, Path, Image.Image], 
                         prompt: str = "Describe this image in detail.",
                         url: str = "localhost:8000") -> str:
    client = TritonVLMClient(url=url)
    return client.infer("llava", image, prompt)
