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
# # KOSMOS-2 Test (~2B params)
#
# Multimodal model with grounding and zero-shot object detection capabilities.
#
# **Model:** `microsoft/kosmos-2-patch14-224`  
# **Size:** ~2B parameters  
# **License:** Permissive (MIT)  
# **Features:** Grounding, zero-shot object detection, referring expression comprehension  
# **Requirements:** ~4GB disk, ~2GB RAM/VRAM
#

# %%
import sys
sys.path.append('..')

import torch
from transformers import AutoProcessor, AutoModelForVision2Seq
from PIL import Image
from vlm_utils import get_device_info, load_test_images, display_image, print_section, print_subsection

device = get_device_info()


# %% [markdown]
# ## Load Test Images
#

# %%
image_files = load_test_images()


# %% [markdown]
# ## Load KOSMOS-2 Model
#

# %%
print("Loading KOSMOS-2...")
model_id = "microsoft/kosmos-2-patch14-224"

# Determine dtype based on device
use_float16 = torch.cuda.is_available() or torch.backends.mps.is_available()
model_dtype = torch.float16 if use_float16 else torch.float32

processor = AutoProcessor.from_pretrained(model_id)
model = AutoModelForVision2Seq.from_pretrained(
    model_id,
    torch_dtype=model_dtype
).to(device)
print("✓ KOSMOS-2 loaded!")


# %% [markdown]
# ## Define Inference Function
#

# %%
def describe_image(image_path, prompt="<grounding>Describe this image in detail."):
    """Generate description for an image using KOSMOS-2."""
    image = Image.open(image_path)
    inputs = processor(text=prompt, images=image, return_tensors="pt").to(device)
    
    generated_ids = model.generate(
        pixel_values=inputs["pixel_values"],
        input_ids=inputs["input_ids"],
        attention_mask=inputs["attention_mask"],
        image_embeds=None,
        image_embeds_position_mask=inputs["image_embeds_position_mask"],
        use_cache=True,
        max_new_tokens=128,
    )
    generated_text = processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
    
    # Process grounding output
    processed_text, entities = processor.post_process_generation(generated_text)
    return processed_text



# %% [markdown]
# ## Test on All Images
#

# %%
for image_path in image_files:
    print_section(f"Image: {image_path.name}")
    
    display_image(image_path)
    
    print_subsection("🔍 KOSMOS-2 Description:")
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
        "<grounding>What objects can you see in this image?",
        "<grounding>Describe the colors in this image.",
        "<grounding>What is the main subject?"
    ]
    
    print_section(f"Custom Prompts - {test_image.name}")
    display_image(test_image)
    
    for prompt in custom_prompts:
        print_subsection(f"Q: {prompt}")
        answer = describe_image(test_image, prompt)
        print(f"A: {answer}")


# %%
