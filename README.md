# VLM Testbench

Local Vision Language Model testing with Jupyter notebooks. Compare 3 diverse VLMs running entirely on your hardware.

## Models Tested

| Model | Size | Disk | RAM/VRAM | Best For | Notebook |
|-------|------|------|----------|----------|----------|
| **Moondream2** | 2B | 4GB | 2GB | Fast, CPU-friendly, general purpose | `moondream_test.ipynb` |
| **KOSMOS-2** | 2B | 4GB | 2GB | Object grounding, detection, spatial understanding | `kosmos2_test.ipynb` |
| **LLaVA 1.6 Mistral** | 7B | 14GB | 8GB | Most capable, detailed descriptions | `llava_test.ipynb` |

**Model Sources:**
- `vikhyatk/moondream2`
- `microsoft/kosmos-2-patch14-224`
- `llava-hf/llava-v1.6-mistral-7b-hf`

All models have permissive licenses and run locally (cached in `~/.cache/huggingface/`).

## Quick Start

```bash
# Install dependencies
uv sync

# Launch Jupyter
uv run jupyter notebook

# Open moondream_test.ipynb and run cells
```

## Requirements

- Python 3.10+
- 8GB+ RAM (16GB+ recommended)
- ~22GB disk space for all models (~4GB for just Moondream2)
- GPU recommended (Apple Silicon MPS, CUDA), CPU works

## Usage

1. Add test images to `test_images/` folder
2. Open a notebook (start with `moondream_test.ipynb`)
3. Run cells - models download on first run
4. Try custom prompts
5. Compare results across models
