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
# # Moondream2 Test (~2B params)
#
# Small, efficient VLM that runs well on CPU. Perfect for quick testing and low-resource environments.
#
# **Model:** `vikhyatk/moondream2`  
# **Size:** ~2B parameters  
# **License:** Permissive  
# **Features:** Fast inference, CPU-friendly
#

# %%
import sys
sys.path.append('..')

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from PIL import Image
from vlm_utils import get_device_info, load_test_images, display_image, print_section, print_subsection

device = get_device_info()


# %% [markdown]
# ## Load Test Images
#

# %%
image_files = load_test_images()


# %% [markdown]
# ## Load Moondream2 Model
#

# %%
print("Loading moondream2...")
model_id = "vikhyatk/moondream2"
model = AutoModelForCausalLM.from_pretrained(
    model_id,
    trust_remote_code=True,
    torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
    low_cpu_mem_usage=True,
    revision="2025-01-09"
).to(device)
tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True, revision="2025-01-09")
print("✓ Moondream2 loaded!")


# %% [markdown]
# ## Define Inference Function
#

# %%
def describe_image(image_path, prompt="Describe this image in detail."):
    """Generate description for an image using Moondream2."""
    image = Image.open(image_path)
    enc_image = model.encode_image(image)
    output = model.answer_question(enc_image, prompt, tokenizer)
    return output



# %% [markdown]
# ## Test on All Images
#

# %%
for image_path in image_files:
    print_section(f"Image: {image_path.name}")
    
    display_image(image_path)
    
    print_subsection("🌙 Moondream2 Description:")
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
