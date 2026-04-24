"""Embed a dataset of images into DuckDB for retrieval experiments."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

import duckdb

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from embedding_eval.datasets import iter_dataset_paths, load_dataset_records
from embedding_eval.duckdb_utils import create_embeddings_table
from embedding_eval.models import build_model


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dataset-root",
        type=Path,
        default=Path("datasets/demo_image_search"),
        help="Directory containing metadata.csv and an images/ folder",
    )
    parser.add_argument(
        "--metadata-path",
        type=Path,
        default=None,
        help="Optional explicit metadata CSV path",
    )
    parser.add_argument(
        "--db-path",
        type=Path,
        default=Path("data/image_embeddings.duckdb"),
        help="DuckDB file where embeddings will be stored",
    )
    parser.add_argument(
        "--models",
        nargs="+",
        default=["siglip2", "clip"],
        help="Model names to embed with",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=8,
        help="Number of images to embed at once",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Delete existing embeddings for the same dataset/model pairs before inserting",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    dataset_root = args.dataset_root
    metadata_path = args.metadata_path or dataset_root / "metadata.csv"
    dataset_name = dataset_root.name

    records = load_dataset_records(metadata_path)
    image_paths = iter_dataset_paths(dataset_root, records)
    args.db_path.parent.mkdir(parents=True, exist_ok=True)

    print(f"Dataset: {dataset_name}")
    print(f"Metadata: {metadata_path}")
    print(f"Images to embed: {len(records)}")
    print(f"Output DB: {args.db_path}")

    connection = duckdb.connect(str(args.db_path))
    create_embeddings_table(connection)

    total_models = len(args.models)
    for model_index, model_name in enumerate(args.models, start=1):
        print(
            f"\nModel {model_index}/{total_models}: "
            f"embedding {len(records)} images with '{model_name}'"
        )
        model = build_model(model_name, batch_size=args.batch_size)
        embeddings = model.embed_images(image_paths, progress=True)

        if args.overwrite:
            print(f"Deleting existing rows for dataset='{dataset_name}', model='{model.name}'")
            connection.execute(
                """
                DELETE FROM image_embeddings
                WHERE dataset_name = ? AND model_name = ?
                """,
                [dataset_name, model.name],
            )

        print(f"Inserting {len(records)} rows for '{model.name}'")
        rows = [
            (
                dataset_name,
                model.name,
                model.model_id,
                record.image_id,
                record.image_filename,
                str(image_path),
                record.label,
                record.caption,
                record.split,
                embedding.tolist(),
                int(len(embedding)),
            )
            for record, image_path, embedding in zip(records, image_paths, embeddings, strict=True)
        ]

        connection.executemany(
            """
            INSERT INTO image_embeddings (
                dataset_name,
                model_name,
                model_id,
                image_id,
                image_filename,
                image_path,
                label,
                caption,
                split,
                embedding,
                embedding_dim
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            rows,
        )
        print(f"Stored {len(rows)} embeddings for '{model.name}' in {args.db_path}")

    connection.close()
    print("\nEmbedding run complete.")


if __name__ == "__main__":
    main()
