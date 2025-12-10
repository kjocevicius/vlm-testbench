import json
import numpy as np
import torch
from PIL import Image
import io
import triton_python_backend_utils as pb_utils
from transformers import AutoProcessor, AutoModelForVision2Seq


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
        
        model_id = "microsoft/kosmos-2-patch14-224"
        self.processor = AutoProcessor.from_pretrained(model_id)
        self.model = AutoModelForVision2Seq.from_pretrained(
            model_id,
            torch_dtype=self.dtype
        ).to(self.device)
        
        print(f"KOSMOS-2 model loaded on {self.device}")

    def execute(self, requests):
        responses = []
        
        for request in requests:
            image_bytes = pb_utils.get_input_tensor_by_name(request, "image").as_numpy()[0]
            prompt_bytes = pb_utils.get_input_tensor_by_name(request, "prompt").as_numpy()[0]
            
            prompt = prompt_bytes.decode("utf-8")
            image = Image.open(io.BytesIO(image_bytes))
            
            inputs = self.processor(text=prompt, images=image, return_tensors="pt").to(self.device)
            
            with torch.no_grad():
                generated_ids = self.model.generate(
                    pixel_values=inputs["pixel_values"],
                    input_ids=inputs["input_ids"],
                    attention_mask=inputs["attention_mask"],
                    image_embeds=None,
                    image_embeds_position_mask=inputs["image_embeds_position_mask"],
                    use_cache=True,
                    max_new_tokens=128
                )
            
            output = self.processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
            
            if prompt in output:
                output = output.replace(prompt, "").strip()
            
            output_tensor = pb_utils.Tensor(
                "generated_text",
                np.array([output.encode("utf-8")], dtype=self.output0_dtype)
            )
            
            inference_response = pb_utils.InferenceResponse(output_tensors=[output_tensor])
            responses.append(inference_response)
        
        return responses

    def finalize(self):
        print("Cleaning up KOSMOS-2 model...")
