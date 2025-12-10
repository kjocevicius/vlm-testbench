# VLM Testbench

Vision Language Model testing environment with Jupyter notebooks. All models run locally on your machine.

## Project Structure

```
vlm-testbench/
├── vlm_utils.py           # Shared utilities and helper functions
├── moondream_test.ipynb   # Moondream2 model testing
├── deepseek_test.ipynb    # DeepSeek-VL-Chat model testing
├── llava_test.ipynb       # LLaVA 1.6 Mistral model testing
└── test_images/           # Place your test images here
```

## Models

This testbench includes 3 diverse VLMs from the Hugging Face hub:

### 1. Moondream2 (~2B params)
- **Model:** `vikhyatk/moondream2`
- **Size:** ~2B parameters, ~4GB disk space
- **Best for:** Fast inference, CPU-friendly, quick testing
- **License:** Permissive

### 2. DeepSeek-VL-Chat (7B params)
- **Model:** `deepseek-ai/deepseek-vl-7b-chat`
- **Size:** 7B parameters, ~14GB disk space
- **Best for:** Chat-optimized responses, good accuracy
- **License:** Permissive
- **Requirements:** ~8GB RAM/VRAM
- **Note:** Currently falls back to CPU on Apple Silicon (MPS compatibility issues)

### 3. LLaVA 1.6 Mistral (7B params)
- **Model:** `llava-hf/llava-v1.6-mistral-7b-hf`
- **Size:** 7B parameters, ~14GB disk space
- **Best for:** Highly capable, excellent accuracy
- **License:** Permissive
- **Requirements:** ~8GB RAM/VRAM

All models are:
- Open source with permissive licenses
- Downloaded and cached locally in `~/.cache/huggingface/`
- Run entirely on your hardware (GPU recommended, CPU works)

## Requirements

- Python 3.10+
- ~30GB disk space for all models (or ~4GB for just Moondream2)
- 8GB+ RAM (16GB+ recommended for 7B models)
- GPU with 8GB+ VRAM recommended (but CPU works)

## Setup

1. Install dependencies:
```bash
uv sync
```

2. Launch Jupyter:
```bash
uv run jupyter notebook
```

3. Choose a notebook to test:
   - `moondream_test.ipynb` - Start here (smallest, fastest)
   - `deepseek_test.ipynb` - Chat-optimized model
   - `llava_test.ipynb` - Most capable model

## Usage

1. **Add images:** Place your test images in `test_images/` folder (JPG, PNG, JPEG)

2. **Open a notebook:** Start with `moondream_test.ipynb` for the fastest experience

3. **Run cells sequentially:** Models will download on first run (~5-10 minutes)

4. **Try custom prompts:** Each notebook includes examples for custom questions

5. **Compare models:** Run different notebooks on the same images to compare outputs

## Shared Utilities

The `vlm_utils.py` file contains helper functions used across all notebooks:
- `get_device_info()` - Detect and display device information
- `load_test_images()` - Load images from test folder
- `display_image()` - Display images in notebooks
- `print_section()` / `print_subsection()` - Formatted output

## Local Execution

All inference happens locally:
- ✅ No API keys required
- ✅ No data sent to external services
- ✅ Full control over model execution
- ✅ Works offline after initial model download
- ✅ Privacy-preserving
