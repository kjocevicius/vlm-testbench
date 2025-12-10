import json
import numpy as np
import torch
from PIL import Image
import io
import triton_python_backend_utils as pb_utils
from transformers import LlavaNextProcessor, LlavaNextForConditionalGeneration


class TritonPythonModel:
    def initialize(self, args):
        self.model_config = json.loads(args["model_config"])
        
        output0_config = pb_utils.get_output_config_by_name(self.model_config, "generated_text")
        self.output0_dtype = pb_utils.triton_string_to_numpy(output0_config["data_type"])
        
        if torch.cuda.is_available():
            self.device = torch.device("cuda")
            self.dtype = torch.float16
        elif torch.backends.mps.is_available():
            self.device = torch.device("mps")
            self.dtype = torch.float16
        else:
            self.device = torch.device("cpu")
            self.dtype = torch.float32
        
        model_id = "llava-hf/llava-v1.6-mistral-7b-hf"
        self.processor = LlavaNextProcessor.from_pretrained(model_id, use_fast=True)
        self.model = LlavaNextForConditionalGeneration.from_pretrained(
            model_id,
            torch_dtype=self.dtype,
            low_cpu_mem_usage=True
        ).to(self.device)
        
        print(f"LLaVA 1.6 Mistral model loaded on {self.device}")

    def execute(self, requests):
        responses = []
        
        for request in requests:
            image_bytes = pb_utils.get_input_tensor_by_name(request, "image").as_numpy()[0]
            prompt_bytes = pb_utils.get_input_tensor_by_name(request, "prompt").as_numpy()[0]
            
            prompt = prompt_bytes.decode("utf-8")
            image = Image.open(io.BytesIO(image_bytes))
            
            conversation = [
                {
                    "role": "user",
                    "content": [
                        {"type": "image"},
                        {"type": "text", "text": prompt},
                    ],
                },
            ]
            prompt_text = self.processor.apply_chat_template(conversation, add_generation_prompt=True)
            
            inputs = self.processor(images=image, text=prompt_text, return_tensors="pt").to(self.device)
            
            with torch.no_grad():
                output_ids = self.model.generate(**inputs, max_new_tokens=200)
            
            output = self.processor.decode(output_ids[0], skip_special_tokens=True)
            
            if "[/INST]" in output:
                output = output.split("[/INST]")[-1].strip()
            
            output_tensor = pb_utils.Tensor(
                "generated_text",
                np.array([output.encode("utf-8")], dtype=self.output0_dtype)
            )
            
            inference_response = pb_utils.InferenceResponse(output_tensors=[output_tensor])
            responses.append(inference_response)
        
        return responses

    def finalize(self):
        print("Cleaning up LLaVA model...")
