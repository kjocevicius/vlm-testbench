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

### Image Embedding And Retrieval

This repo can also be used for image retrieval experiments with reusable embedding
models and DuckDB vector search.

```bash
# Build embeddings for the sample dataset with SigLIP2 and CLIP
uv run python scripts/embed_image_dataset.py --overwrite

# Explore nearest neighbors in the paired notebook/script
uv run jupyter notebook notebooks/duckdb_image_search.ipynb
```

New pieces:
- `embedding_eval/` - reusable image-embedding interface and model registry
- `datasets/demo_image_search/` - sample image dataset with `metadata.csv`
- `scripts/embed_image_dataset.py` - writes embeddings into `data/image_embeddings.duckdb`
- `notebooks/duckdb_image_search.py` - percent-format notebook for SQL similarity search

### Download A Pixabay Clothing Dataset

You can also build a small clothing-focused dataset from Pixabay search results,
then embed it into DuckDB for retrieval experiments.

#### 1. Set the API key

Either export it manually:

```bash
export PIXABAY_API_KEY=...
```

or, if you use `direnv`, add it to `.envrc` and allow it:

```bash
direnv allow
```

#### 2. Download images

Run the downloader:

```bash
uv run python scripts/download_pixabay_dataset.py --per-query 8
```

Current defaults are biased toward folded / flat-lay marketplace-style clothing
photos, for example:
- `t shirt folded`
- `t shirt flat lay`
- `dress folded`
- `dress flat lay`
- `jeans folded`
- `jacket folded`
- `hoodie folded`
- `sweater folded`

The downloader:
- writes images under `datasets/pixabay_clothes/images/`
- writes metadata to `datasets/pixabay_clothes/metadata.csv`
- downloads smaller Pixabay `webformat` images by default
- applies local tag filtering to reduce portraits / model photos

Useful flags:

```bash
# Keep more images per query
uv run python scripts/download_pixabay_dataset.py --per-query 12

# Download even smaller preview images
uv run python scripts/download_pixabay_dataset.py --download-size preview

# Try custom search phrases
uv run python scripts/download_pixabay_dataset.py --queries "jeans folded" "dress flat lay"
```

Important: `--overwrite` replaces metadata and redownloads matching filenames,
but it does not delete leftover image files with different names. For a true clean
rebuild, clear the generated dataset first:

```bash
find datasets/pixabay_clothes/images -type f -delete
rm -f datasets/pixabay_clothes/metadata.csv datasets/pixabay_clothes/metadata.jsonl
```

Then run the downloader again.

#### 3. Embed the downloaded dataset into DuckDB

```bash
uv run python scripts/embed_image_dataset.py \
  --dataset-root datasets/pixabay_clothes \
  --overwrite
```

This writes embeddings into:

```bash
data/image_embeddings.duckdb
```

By default it stores one row per `(image_id, model_name)` for:
- `siglip2`
- `clip`

#### 4. Explore results in the notebook

```bash
uv run jupyter notebook notebooks/duckdb_image_search.ipynb
```

The notebook reads available datasets from DuckDB and compares retrieval results
for multiple queries across:
- `SigLIP2`
- `SigLIP2 + Label`
- `CLIP`
- `CLIP + Label`

Notes:
- The script stores downloaded image files locally rather than hotlinking Pixabay URLs.
- `metadata.csv` keeps the first four fields compatible with the sample dataset shape:
  `image_id`, `image_filename`, `label`, `caption`.
- Extra attribution fields are included so notebooks can show source and contributor info.
- Pixabay’s API docs ask you to show users where results came from and discourage
  systematic mass downloads: [Pixabay API docs](https://pixabay.com/api/docs/).

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
