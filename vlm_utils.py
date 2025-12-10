"""
Shared utilities for VLM testbench.
"""

import torch
from pathlib import Path
from PIL import Image
from typing import List


def get_device_info():
    """Get device information and print details."""
    if torch.cuda.is_available():
        device = torch.device('cuda')
    elif torch.backends.mps.is_available():
        device = torch.device('mps')
    else:
        device = torch.device('cpu')
    
    print(f"Using device: {device}")
    print(f"PyTorch version: {torch.__version__}")
    
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.2f} GB")
    elif torch.backends.mps.is_available():
        print("Using Apple Silicon MPS (Metal Performance Shaders)")
    else:
        print("Running on CPU - models will be slower but still work")
    
    return device


def load_test_images(images_dir: str = "../test_images") -> List[Path]:
    """Load all test images from directory."""
    test_images_dir = Path(images_dir)
    image_files = (
        list(test_images_dir.glob("*.jpg")) +
        list(test_images_dir.glob("*.png")) +
        list(test_images_dir.glob("*.jpeg")) +
        list(test_images_dir.glob("*.JPG")) +
        list(test_images_dir.glob("*.PNG"))
    )
    
    if not image_files:
        print(f"No images found in {images_dir}/ folder. Please add some images to test.")
    else:
        print(f"Found {len(image_files)} image(s) to test:")
        for img_file in image_files:
            print(f"  - {img_file.name}")
    
    return image_files


def display_image(image_path: Path, max_width: int = 600):
    """Display an image with resize."""
    from IPython.display import display
    
    img = Image.open(image_path)
    aspect_ratio = img.height / img.width
    new_height = int(max_width * aspect_ratio)
    resized = img.resize((max_width, new_height))
    display(resized)


def print_section(title: str, width: int = 80):
    """Print a formatted section header."""
    print("\n" + "=" * width)
    print(title)
    print("=" * width)


def print_subsection(title: str, width: int = 80):
    """Print a formatted subsection header."""
    print(f"\n{title}")
    print("-" * width)
