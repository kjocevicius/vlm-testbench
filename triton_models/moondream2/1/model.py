import json
import numpy as np
import torch
from PIL import Image
import io
import triton_python_backend_utils as pb_utils
from transformers import AutoTokenizer, AutoModelForCausalLM


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
        
        model_id = "vikhyatk/moondream2"
        self.model = AutoModelForCausalLM.from_pretrained(
            model_id,
            trust_remote_code=True,
            torch_dtype=self.dtype,
            low_cpu_mem_usage=True,
            revision="2025-01-09"
        ).to(self.device)
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_id, 
            trust_remote_code=True, 
            revision="2025-01-09"
        )
        
        print(f"Moondream2 model loaded on {self.device}")

    def execute(self, requests):
        responses = []
        
        for request in requests:
            image_bytes = pb_utils.get_input_tensor_by_name(request, "image").as_numpy()[0]
            prompt_bytes = pb_utils.get_input_tensor_by_name(request, "prompt").as_numpy()[0]
            
            prompt = prompt_bytes.decode("utf-8")
            image = Image.open(io.BytesIO(image_bytes))
            
            with torch.no_grad():
                enc_image = self.model.encode_image(image)
                output = self.model.answer_question(enc_image, prompt, self.tokenizer)
            
            output_tensor = pb_utils.Tensor(
                "generated_text",
                np.array([output.encode("utf-8")], dtype=self.output0_dtype)
            )
            
            inference_response = pb_utils.InferenceResponse(output_tensors=[output_tensor])
            responses.append(inference_response)
        
        return responses

    def finalize(self):
        print("Cleaning up Moondream2 model...")
