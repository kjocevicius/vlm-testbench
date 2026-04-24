"""DuckDB helpers for storing and querying image embeddings."""

from __future__ import annotations

from typing import Final

import duckdb


EMBEDDINGS_TABLE_SQL: Final[str] = """
CREATE TABLE IF NOT EXISTS image_embeddings (
    dataset_name VARCHAR,
    model_name VARCHAR,
    model_id VARCHAR,
    image_id VARCHAR,
    image_filename VARCHAR,
    image_path VARCHAR,
    label VARCHAR,
    caption VARCHAR,
    split VARCHAR,
    embedding FLOAT[],
    embedding_dim INTEGER,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
)
"""


def create_embeddings_table(connection: duckdb.DuckDBPyConnection) -> None:
    connection.execute(EMBEDDINGS_TABLE_SQL)


def detect_similarity_expression(connection: duckdb.DuckDBPyConnection) -> str:
    """Return a compatible DuckDB similarity expression for two list vectors."""
    candidates = [
        ("list_cosine_similarity", "list_cosine_similarity(embedding, ?)"),
        ("array_cosine_similarity", "array_cosine_similarity(embedding, ?)"),
        ("list_cosine_distance", "1 - list_cosine_distance(embedding, ?)"),
        ("array_cosine_distance", "1 - array_cosine_distance(embedding, ?)"),
    ]

    for probe_name, expression in candidates:
        try:
            connection.execute(f"SELECT {probe_name}([1.0, 0.0], [1.0, 0.0])").fetchone()
            return expression
        except duckdb.Error:
            continue

    raise RuntimeError(
        "This DuckDB build does not expose cosine similarity functions for list vectors"
    )
