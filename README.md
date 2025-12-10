# VLM Testbench

Local Vision Language Model testing with Jupyter notebooks. Compare 3 diverse VLMs running entirely on your hardware.

## Models Tested

| Model | Size | Disk | RAM/VRAM | Best For | Notebook |
|-------|------|------|----------|----------|----------|
| **Moondream2** | 2B | 4GB | 2GB | Fast, CPU-friendly, general purpose | `notebooks/moondream_test.ipynb` |
| **KOSMOS-2** | 2B | 4GB | 2GB | Object grounding, detection, spatial understanding | `notebooks/kosmos2_test.ipynb` |
| **LLaVA 1.6 Mistral** | 7B | 14GB | 8GB | Most capable, detailed descriptions | `notebooks/llava_test.ipynb` |

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

# Open notebooks/moondream_test.ipynb and run cells
```

## Requirements

- Python 3.10+
- 8GB+ RAM (16GB+ recommended)
- ~22GB disk space for all models (~4GB for just Moondream2)
- GPU recommended (Apple Silicon MPS, CUDA), CPU works

**Additional for Triton:**
- Docker and Docker Compose
- GPU: NVIDIA GPU with CUDA support (optional, CPU mode available)
- 4GB+ additional disk space for Triton container

## Usage

### Direct Model Testing

1. Add test images to `test_images/` folder
2. Open a notebook (start with `notebooks/moondream_test.ipynb`)
3. Run cells - models download on first run
4. Try custom prompts
5. Compare results across models

### Triton Inference Server

Run models via Triton Inference Server for optimized:

```bash
# Start Triton server (GPU)
docker compose up -d

# Start Triton server (CPU only)
docker compose -f docker-compose-cpu.yml up -d

# Check server status
docker compose logs -f

# Stop server
docker compose down
```

Then open `notebooks/triton_inference.ipynb` to test inference via Triton.

**Triton Features:**
- Concurrent model serving
- Dynamic batching
- Metrics and monitoring
- Production-ready deployment
- HTTP/gRPC endpoints (ports 8000/8001)

**Model Repository:** `triton_models/`
- `moondream2/` - Python backend for Moondream2
- `kosmos2/` - Python backend for KOSMOS-2
- `llava/` - Python backend for LLaVA 1.6
