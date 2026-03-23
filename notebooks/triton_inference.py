# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:percent
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.19.1
# ---

# %% [markdown]
# # Triton Inference Server - VLM Testing
#
# Test all VLM models via Triton Inference Server for optimized inference.
#
# **Prerequisites:**
# - Triton server must be running: `docker compose up -d`
# - Models will be loaded on first inference
#
# **Models Available:**
# - `moondream2` - Fast, efficient 2B model
# - `kosmos2` - Object grounding and detection
# - `llava` - Most capable 7B model
#

# %%
import sys
sys.path.append('..')

from triton_client import TritonVLMClient
from vlm_utils import load_test_images, display_image, print_section, print_subsection


# %% [markdown]
# ## Check Triton Server Status
#

# %%
client = TritonVLMClient(url="localhost:8000")

if client.is_server_ready():
    print("✓ Triton server is ready!")
    print("\nAvailable models:")
    models = client.list_models()
    for model in models:
        model_name = model['name']
        ready = "✓" if client.is_model_ready(model_name) else "✗"
        print(f"  {ready} {model_name}")
else:
    print("✗ Triton server is not ready. Make sure to run: docker compose up -d")


# %% [markdown]
# ## Load Test Images
#

# %%
image_files = load_test_images()


# %% [markdown]
# ## Test All Models
#

# %%
if image_files:
    test_image = image_files[0]
    prompt = "Describe this image in detail."
    
    print_section(f"Testing: {test_image.name}")
    display_image(test_image)
    print(f"\nPrompt: {prompt}")
    
    print_subsection("Moondream2 (2B) - Fast & Efficient")
    result = client.infer("moondream2", test_image, prompt)
    print(result)
    
    print_subsection("KOSMOS-2 (2B) - Grounding & Detection")
    result = client.infer("kosmos2", test_image, "<grounding>" + prompt)
    print(result)
    
    print_subsection("LLaVA 1.6 (7B) - Most Capable")
    result = client.infer("llava", test_image, prompt)
    print(result)


# %% [markdown]
# ## Custom Prompts
#
# Try your own prompts!
#

# %%
custom_prompt = "What objects are in this image?"
model_name = "moondream2"

if image_files:
    result = client.infer(model_name, image_files[0], custom_prompt)
    print(f"Model: {model_name}")
    print(f"Prompt: {custom_prompt}")
    print(f"\nResponse:\n{result}")


# %% [markdown]
# ## Benchmark Performance
#
# Compare inference times across models.
#

# %%
import time

if image_files:
    test_image = image_files[0]
    prompt = "What is in this image?"
    models = ["moondream2", "kosmos2", "llava"]
    
    print_section("Performance Benchmark (3 runs each)")
    
    for model in models:
        if not client.is_model_ready(model):
            print(f"Skipping {model} - not ready")
            continue
        
        times = []
        for i in range(3):
            start = time.time()
            client.infer(model, test_image, prompt)
            elapsed = time.time() - start
            times.append(elapsed)
        
        avg_time = sum(times) / len(times)
        print(f"{model:12s}: {avg_time:.2f}s avg (runs: {', '.join(f'{t:.2f}s' for t in times)})")


# %% [markdown]
# ## Process Multiple Images
#

# %%
model_name = "moondream2"
prompt = "What is the main subject of this image?"

for img_path in image_files:
    print_subsection(f"Image: {img_path.name}")
    display_image(img_path, max_width=400)
    result = client.infer(model_name, img_path, prompt)
    print(f"Response: {result}\n")

