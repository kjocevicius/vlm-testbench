"""Reusable image-embedding utilities for evaluation scripts and notebooks."""

from .datasets import DatasetImageRecord, load_dataset_records
from .duckdb_utils import create_embeddings_table, detect_similarity_expression
from .models import (
    ImageEmbeddingModel,
    build_model,
    get_model_config,
    list_model_configs,
)

__all__ = [
    "DatasetImageRecord",
    "ImageEmbeddingModel",
    "build_model",
    "create_embeddings_table",
    "detect_similarity_expression",
    "get_model_config",
    "list_model_configs",
    "load_dataset_records",
]
