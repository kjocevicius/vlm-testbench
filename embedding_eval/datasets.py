"""Dataset helpers for image retrieval experiments."""

from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable


@dataclass(frozen=True)
class DatasetImageRecord:
    image_id: str
    image_filename: str
    label: str
    caption: str
    split: str = "train"

    def resolve_path(self, dataset_root: Path) -> Path:
        return dataset_root / "images" / self.image_filename


def load_dataset_records(metadata_path: str | Path) -> list[DatasetImageRecord]:
    """Load dataset metadata from a CSV file."""
    metadata_file = Path(metadata_path)
    with metadata_file.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        required_columns = {"image_id", "image_filename", "label", "caption"}
        if reader.fieldnames is None or not required_columns.issubset(reader.fieldnames):
            missing = sorted(required_columns.difference(reader.fieldnames or []))
            raise ValueError(f"Metadata file is missing required columns: {missing}")

        records = [
            DatasetImageRecord(
                image_id=row["image_id"],
                image_filename=row["image_filename"],
                label=row["label"],
                caption=row["caption"],
                split=row.get("split", "train") or "train",
            )
            for row in reader
        ]

    if not records:
        raise ValueError(f"Metadata file {metadata_file} does not contain any records")

    return records


def iter_dataset_paths(
    dataset_root: str | Path, records: Iterable[DatasetImageRecord]
) -> list[Path]:
    root = Path(dataset_root)
    paths = [record.resolve_path(root) for record in records]
    missing = [path for path in paths if not path.exists()]
    if missing:
        sample = ", ".join(str(path) for path in missing[:3])
        raise FileNotFoundError(f"Missing dataset image files, for example: {sample}")
    return paths
