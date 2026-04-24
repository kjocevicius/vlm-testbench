"""Download a small Pixabay image dataset and emit retrieval-friendly metadata."""

from __future__ import annotations

import argparse
import csv
import json
import os
import re
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import requests


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


PIXABAY_API_URL = "https://pixabay.com/api/"
DEFAULT_QUERY_TERMS = [
    "t shirt folded",
    "t shirt flat lay",
    "t-shirt hanging",
    "dress folded",
    "dress flat lay",
    "jeans folded",
    "jacket folded",
    "hoodie folded",
    "sweater folded",
    "sweater hanging",
    "sneakers centered",
]
DEFAULT_PAGE_SIZE = 20
LABEL_KEYWORDS = [
    "t shirt",
    "dress",
    "jeans",
    "jacket",
    "hoodie",
    "sweater",
    "sneakers",
    "handbag",
]
DEFAULT_EXCLUDED_TAG_TERMS = [
    "woman",
    "women",
    "man",
    "men",
    "person",
    "people",
    "portrait",
    "model",
    "girl",
    "boy",
    "face",
    "selfie",
    "smile",
    "smiling",
]
DEFAULT_REQUIRED_TAG_TERMS = [
    "folded",
    "flat lay",
    "garment",
    "clothing",
    "shirt",
    "dress",
    "jeans",
    "jacket",
    "hoodie",
    "sweater",
]


@dataclass(frozen=True)
class PixabayRecord:
    image_id: str
    image_filename: str
    label: str
    caption: str
    split: str
    source: str
    query: str
    tags: str
    pixabay_id: int
    page_url: str
    download_url: str
    preview_url: str
    contributor_username: str
    contributor_id: int
    contributor_profile_url: str
    image_width: int
    image_height: int
    image_size: int
    likes: int
    downloads: int
    views: int


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path("datasets/pixabay_clothes"),
        help="Dataset directory that will contain images/ and metadata files",
    )
    parser.add_argument(
        "--queries",
        nargs="+",
        default=DEFAULT_QUERY_TERMS,
        help="Search queries to run against Pixabay",
    )
    parser.add_argument(
        "--per-query",
        type=int,
        default=10,
        help="Maximum number of images to keep per query",
    )
    parser.add_argument(
        "--candidate-multiplier",
        type=int,
        default=6,
        help="How many extra API candidates to fetch before local filtering",
    )
    parser.add_argument(
        "--image-type",
        default="photo",
        choices=["all", "photo", "illustration", "vector"],
        help="Pixabay image_type filter",
    )
    parser.add_argument(
        "--category",
        default="fashion",
        help="Pixabay category filter",
    )
    parser.add_argument(
        "--orientation",
        default="all",
        choices=["all", "horizontal", "vertical"],
        help="Pixabay orientation filter",
    )
    parser.add_argument(
        "--order",
        default="popular",
        choices=["popular", "latest"],
        help="Pixabay result ordering",
    )
    parser.add_argument(
        "--download-size",
        default="webformat",
        choices=["preview", "webformat", "large"],
        help="Which Pixabay image size to download",
    )
    parser.add_argument(
        "--lang",
        default="en",
        help="Pixabay language code",
    )
    parser.add_argument(
        "--safesearch",
        action="store_true",
        help="Enable Pixabay safe search filtering",
    )
    parser.add_argument(
        "--min-width",
        type=int,
        default=256,
        help="Minimum image width filter",
    )
    parser.add_argument(
        "--min-height",
        type=int,
        default=256,
        help="Minimum image height filter",
    )
    parser.add_argument(
        "--timeout-seconds",
        type=float,
        default=30.0,
        help="HTTP timeout for API and image downloads",
    )
    parser.add_argument(
        "--pause-seconds",
        type=float,
        default=0.25,
        help="Pause between query requests to stay polite",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing files and metadata in the output dataset",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Fetch metadata only without downloading image files",
    )
    parser.add_argument(
        "--excluded-tag-terms",
        nargs="*",
        default=DEFAULT_EXCLUDED_TAG_TERMS,
        help="Reject results whose tags contain any of these terms",
    )
    parser.add_argument(
        "--required-tag-terms",
        nargs="*",
        default=DEFAULT_REQUIRED_TAG_TERMS,
        help="Keep only results whose tags contain at least one of these terms",
    )
    return parser.parse_args()


def get_api_key() -> str:
    api_key = os.environ.get("PIXABAY_API_KEY")
    if not api_key:
        raise RuntimeError(
            "Missing PIXABAY_API_KEY. Create a Pixabay API key and export it before running."
        )
    return api_key


def slugify(text: str) -> str:
    lowered = text.strip().lower()
    normalized = re.sub(r"[^a-z0-9]+", "-", lowered)
    return normalized.strip("-") or "item"


def infer_label(query: str) -> str:
    normalized_query = query.strip().lower()
    for keyword in LABEL_KEYWORDS:
        if keyword in normalized_query:
            return slugify(keyword)
    return slugify(query)


def build_search_params(api_key: str, query: str, args: argparse.Namespace) -> dict[str, Any]:
    return {
        "key": api_key,
        "q": query,
        "lang": args.lang,
        "image_type": args.image_type,
        "orientation": args.orientation,
        "category": args.category,
        "min_width": args.min_width,
        "min_height": args.min_height,
        "safesearch": str(args.safesearch).lower(),
        "order": args.order,
        "page": 1,
        "per_page": min(max(args.per_query * args.candidate_multiplier, 3), 200),
    }


def choose_download_url(hit: dict[str, Any]) -> str:
    for key in ("largeImageURL", "webformatURL", "previewURL"):
        value = hit.get(key)
        if value:
            return str(value)
    raise ValueError(f"Pixabay hit {hit.get('id')} did not contain a downloadable URL")


def choose_download_url_for_size(hit: dict[str, Any], download_size: str) -> str:
    size_preferences = {
        "preview": ("previewURL", "webformatURL", "largeImageURL"),
        "webformat": ("webformatURL", "previewURL", "largeImageURL"),
        "large": ("largeImageURL", "webformatURL", "previewURL"),
    }
    for key in size_preferences[download_size]:
        value = hit.get(key)
        if value:
            return str(value)
    raise ValueError(f"Pixabay hit {hit.get('id')} did not contain a downloadable URL")


def choose_filename(hit: dict[str, Any], query: str, download_size: str) -> str:
    download_url = choose_download_url_for_size(hit, download_size)
    suffix = Path(download_url.split("?")[0]).suffix or ".jpg"
    return f"{slugify(query)}-{hit['id']}{suffix}"


def build_caption(hit: dict[str, Any], query: str) -> str:
    tags = str(hit.get("tags", "")).strip()
    if tags:
        return f"Pixabay result for '{query}' tagged with {tags}."
    return f"Pixabay result for '{query}'."


def search_pixabay(
    session: requests.Session,
    api_key: str,
    query: str,
    args: argparse.Namespace,
) -> list[dict[str, Any]]:
    response = session.get(
        PIXABAY_API_URL,
        params=build_search_params(api_key, query, args),
        timeout=args.timeout_seconds,
    )
    response.raise_for_status()
    payload = response.json()
    hits = payload.get("hits", [])
    if not isinstance(hits, list):
        raise RuntimeError(f"Unexpected Pixabay response for query '{query}'")
    return hits


def normalize_tags(hit: dict[str, Any]) -> str:
    return str(hit.get("tags", "")).strip().lower()


def should_keep_hit(hit: dict[str, Any], args: argparse.Namespace) -> bool:
    tags = normalize_tags(hit)
    if not tags:
        return False

    excluded_terms = [term.lower() for term in args.excluded_tag_terms]
    required_terms = [term.lower() for term in args.required_tag_terms]

    if any(term in tags for term in excluded_terms):
        return False

    if required_terms and not any(term in tags for term in required_terms):
        return False

    return True


def download_file(
    session: requests.Session, url: str, destination: Path, timeout_seconds: float
) -> None:
    response = session.get(url, timeout=timeout_seconds, stream=True)
    response.raise_for_status()
    with destination.open("wb") as handle:
        for chunk in response.iter_content(chunk_size=1024 * 128):
            if chunk:
                handle.write(chunk)


def build_record(hit: dict[str, Any], query: str, filename: str, download_size: str) -> PixabayRecord:
    contributor_username = str(hit.get("user", ""))
    contributor_id = int(hit.get("user_id", 0))
    return PixabayRecord(
        image_id=f"pixabay-{hit['id']}",
        image_filename=filename,
        label=infer_label(query),
        caption=build_caption(hit, query),
        split="train",
        source="pixabay",
        query=query,
        tags=str(hit.get("tags", "")),
        pixabay_id=int(hit["id"]),
        page_url=str(hit.get("pageURL", "")),
        download_url=choose_download_url_for_size(hit, download_size),
        preview_url=str(hit.get("previewURL", "")),
        contributor_username=contributor_username,
        contributor_id=contributor_id,
        contributor_profile_url=(
            f"https://pixabay.com/users/{contributor_username}-{contributor_id}/"
            if contributor_username and contributor_id
            else ""
        ),
        image_width=int(hit.get("imageWidth", 0)),
        image_height=int(hit.get("imageHeight", 0)),
        image_size=int(hit.get("imageSize", 0)),
        likes=int(hit.get("likes", 0)),
        downloads=int(hit.get("downloads", 0)),
        views=int(hit.get("views", 0)),
    )


def write_metadata(records: list[PixabayRecord], output_root: Path) -> None:
    metadata_csv = output_root / "metadata.csv"
    metadata_jsonl = output_root / "metadata.jsonl"

    fieldnames = list(asdict(records[0]).keys()) if records else list(PixabayRecord.__annotations__)

    with metadata_csv.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for record in records:
            writer.writerow(asdict(record))

    with metadata_jsonl.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(asdict(record), ensure_ascii=False) + "\n")


def ensure_output_dirs(output_root: Path, overwrite: bool) -> Path:
    images_dir = output_root / "images"
    images_dir.mkdir(parents=True, exist_ok=True)

    if overwrite:
        for metadata_path in (output_root / "metadata.csv", output_root / "metadata.jsonl"):
            if metadata_path.exists():
                metadata_path.unlink()

    return images_dir


def main() -> None:
    args = parse_args()
    api_key = get_api_key()
    output_root = args.output_root
    images_dir = ensure_output_dirs(output_root, overwrite=args.overwrite)

    print(f"Output dataset: {output_root}")
    print(f"Query count: {len(args.queries)}")
    print(f"Target images per query: {args.per_query}")
    print(f"Download size: {args.download_size}")

    session = requests.Session()
    seen_pixabay_ids: set[int] = set()
    records: list[PixabayRecord] = []

    total_queries = len(args.queries)
    for query_index, query in enumerate(args.queries, start=1):
        print(f"\nQuery {query_index}/{total_queries}: '{query}'")
        print("Searching Pixabay...")
        hits = search_pixabay(session, api_key, query, args)
        filtered_hits = [hit for hit in hits if should_keep_hit(hit, args)]
        print(
            f"Kept {len(filtered_hits)} / {len(hits)} candidates after local tag filtering"
        )

        kept = 0
        for candidate_index, hit in enumerate(filtered_hits, start=1):
            pixabay_id = int(hit["id"])
            if pixabay_id in seen_pixabay_ids:
                continue

            filename = choose_filename(hit, query, args.download_size)
            destination = images_dir / filename
            if destination.exists() and not args.overwrite:
                print(f"Skipping existing file {destination.name}")
                continue

            if not args.dry_run:
                print(
                    f"Downloading {kept + 1}/{args.per_query} "
                    f"(candidate {candidate_index}/{len(filtered_hits)}): {destination.name}"
                )
                download_file(
                    session,
                    choose_download_url_for_size(hit, args.download_size),
                    destination,
                    args.timeout_seconds,
                )

            record = build_record(hit, query, filename, args.download_size)
            records.append(record)
            seen_pixabay_ids.add(pixabay_id)
            kept += 1

            if kept >= args.per_query:
                break

        print(f"Kept {kept} unique images for '{query}'")
        time.sleep(args.pause_seconds)

    if not records:
        raise RuntimeError("No images were collected. Try relaxing filters or changing queries.")

    write_metadata(records, output_root)
    print(f"Wrote {len(records)} records to {output_root / 'metadata.csv'}")
    print("Remember Pixabay asks you to show source attribution when displaying results.")


if __name__ == "__main__":
    main()
