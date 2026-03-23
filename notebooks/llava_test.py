# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:percent
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.19.1
#   kernelspec:
#     display_name: Python 3 (ipykernel)
#     language: python
#     name: python3
# ---

# %% [markdown]
# # LLaVA 1.6 Mistral Test (7B params)
#
# Highly capable VLM based on Mistral architecture with excellent performance.
#
# **Model:** `llava-hf/llava-v1.6-mistral-7b-hf`  
# **Size:** 7B parameters  
# **License:** Permissive  
# **Features:** Highly capable, excellent accuracy  
# **Requirements:** ~14GB disk, ~8GB RAM/VRAM
#

# %%
import sys
sys.path.append('..')

import torch
from transformers import LlavaNextProcessor, LlavaNextForConditionalGeneration
from PIL import Image
from vlm_utils import get_device_info, load_test_images, display_image, print_section, print_subsection

device = get_device_info()


# %% [markdown]
# ## Load Test Images
#

# %%
image_files = load_test_images()


# %% [markdown]
# ## Load LLaVA 1.6 Mistral Model
#

# %%
print("Loading LLaVA 1.6 Mistral...")
model_id = "llava-hf/llava-v1.6-mistral-7b-hf"

# Determine dtype based on device
use_float16 = torch.cuda.is_available() or torch.backends.mps.is_available()
model_dtype = torch.float16 if use_float16 else torch.float32

processor = LlavaNextProcessor.from_pretrained(model_id, use_fast=True)
model = LlavaNextForConditionalGeneration.from_pretrained(
    model_id,
    dtype=model_dtype,
    low_cpu_mem_usage=True
).to(device)
print("✓ LLaVA 1.6 Mistral loaded!")


# %% [markdown]
# ## Define Inference Function
#

# %%
def describe_image(image_path, prompt="Describe this image in detail."):
    """Generate description for an image using LLaVA 1.6 Mistral."""
    image = Image.open(image_path)
    
    # LLaVA 1.6 uses a specific conversation format
    conversation = [
        {
            "role": "user",
            "content": [
                {"type": "image"},
                {"type": "text", "text": prompt},
            ],
        },
    ]
    formatted_prompt = processor.apply_chat_template(conversation, add_generation_prompt=True)
    
    inputs = processor(images=image, text=formatted_prompt, return_tensors="pt").to(device)
    output = model.generate(**inputs, max_new_tokens=200)
    description = processor.decode(output[0], skip_special_tokens=True)
    
    # Extract just the assistant's response
    if "[/INST]" in description:
        description = description.split("[/INST]")[-1].strip()
    
    return description



# %% [markdown]
# ## Test on All Images
#

# %%
for image_path in image_files:
    print_section(f"Image: {image_path.name}")
    
    display_image(image_path)
    
    print_subsection("🦙 LLaVA 1.6 Mistral Description:")
    try:
        desc = describe_image(image_path)
        print(desc)
    except Exception as e:
        print(f"Error: {e}")


# %% [markdown]
# ## Custom Prompts
#
# Try asking specific questions about an image.
#

# %%
if image_files:
    test_image = image_files[0]
    
    custom_prompts = [
        "What objects can you see in this image?",
        "What colors are prominent in this image?",
        "What is the main subject of this image?"
    ]
    
    print_section(f"Custom Prompts - {test_image.name}")
    display_image(test_image)
    
    for prompt in custom_prompts:
        print_subsection(f"Q: {prompt}")
        answer = describe_image(test_image, prompt)
        print(f"A: {answer}")


# %%
