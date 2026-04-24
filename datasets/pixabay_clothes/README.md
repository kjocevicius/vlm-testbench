# Pixabay Clothes Dataset

This folder is meant to be populated by:

```bash
uv run python scripts/download_pixabay_dataset.py
```

Expected outputs:
- `images/` downloaded image files
- `metadata.csv` tabular metadata for scripts and notebooks
- `metadata.jsonl` full JSONL metadata with attribution fields

The downloader keeps Pixabay contributor and page fields so downstream notebooks can
show attribution when presenting results.
