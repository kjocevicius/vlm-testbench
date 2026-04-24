"""Universal image-embedding model interface and implementations."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import numpy as np
import torch
from PIL import Image
from transformers import AutoModel, AutoProcessor


def resolve_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def preferred_dtype(device: torch.device) -> torch.dtype:
    return torch.float16 if device.type == "cuda" else torch.float32


@dataclass(frozen=True)
class ModelConfig:
    name: str
    model_id: str
    description: str


MODEL_CONFIGS: dict[str, ModelConfig] = {
    "siglip2": ModelConfig(
        name="siglip2",
        model_id="google/siglip2-base-patch16-224",
        description="SigLIP2 base image encoder from Google",
    ),
    "clip": ModelConfig(
        name="clip",
        model_id="openai/clip-vit-base-patch32",
        description="CLIP ViT-B/32 image encoder from OpenAI",
    ),
}


class ImageEmbeddingModel(ABC):
    """Abstract image-embedding interface for reuse across scripts."""

    name: str
    model_id: str
    device: torch.device

    @abstractmethod
    def embed_images(
        self,
        image_paths: Sequence[str | Path],
        *,
        progress: bool = False,
    ) -> np.ndarray:
        """Return one normalized embedding row per image path."""

    def embed_image(self, image_path: str | Path) -> np.ndarray:
        return self.embed_images([image_path])[0]


class TransformersImageEmbeddingModel(ImageEmbeddingModel):
    def __init__(
        self,
        name: str,
        model_id: str,
        *,
        batch_size: int = 8,
        device: torch.device | None = None,
    ) -> None:
        self.name = name
        self.model_id = model_id
        self.batch_size = batch_size
        self.device = device or resolve_device()
        self.processor = AutoProcessor.from_pretrained(model_id)
        self.model = AutoModel.from_pretrained(
            model_id,
            torch_dtype=preferred_dtype(self.device),
        ).to(self.device)
        self.model.eval()

    def embed_images(
        self,
        image_paths: Sequence[str | Path],
        *,
        progress: bool = False,
    ) -> np.ndarray:
        if not image_paths:
            return np.empty((0, 0), dtype=np.float32)

        image_paths = [Path(path) for path in image_paths]
        embeddings: list[np.ndarray] = []
        total_batches = (len(image_paths) + self.batch_size - 1) // self.batch_size

        for batch_index, start in enumerate(range(0, len(image_paths), self.batch_size), start=1):
            batch_paths = image_paths[start : start + self.batch_size]
            if progress:
                print(
                    f"[{self.name}] batch {batch_index}/{total_batches} "
                    f"({len(batch_paths)} images)"
                )
            images = [Image.open(path).convert("RGB") for path in batch_paths]
            try:
                inputs = self.processor(images=images, return_tensors="pt")
            finally:
                for image in images:
                    image.close()

            inputs = {key: value.to(self.device) for key, value in inputs.items()}
            with torch.inference_mode():
                image_features = self.model.get_image_features(**inputs)
                image_features = torch.nn.functional.normalize(image_features, dim=-1)

            embeddings.append(image_features.detach().cpu().to(torch.float32).numpy())

        return np.concatenate(embeddings, axis=0)


def list_model_configs() -> list[ModelConfig]:
    return list(MODEL_CONFIGS.values())


def get_model_config(name: str) -> ModelConfig:
    try:
        return MODEL_CONFIGS[name]
    except KeyError as exc:
        available = ", ".join(sorted(MODEL_CONFIGS))
        raise ValueError(f"Unknown model '{name}'. Available models: {available}") from exc


def build_model(
    name: str, *, batch_size: int = 8, device: torch.device | None = None
) -> ImageEmbeddingModel:
    config = get_model_config(name)
    return TransformersImageEmbeddingModel(
        name=config.name,
        model_id=config.model_id,
        batch_size=batch_size,
        device=device,
    )
