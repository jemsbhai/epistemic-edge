"""
Experiment 00: Download Bonsai GGUF models from HuggingFace.

Downloads all three Bonsai model sizes for benchmarking.
Requires: pip install huggingface_hub

Usage:
    python 00_download_models.py              # Downloads all 3 sizes
    python 00_download_models.py --size 1.7B  # Download specific size
"""

import argparse
from pathlib import Path

MODELS = {
    "1.7B": {
        "repo": "prism-ml/Bonsai-1.7B-gguf",
        "filename": "Bonsai-1.7B.gguf",
        "size_mb": 248,
    },
    "4B": {
        "repo": "prism-ml/Bonsai-4B-gguf",
        "filename": "Bonsai-4B.gguf",
        "size_mb": 570,
    },
    "8B": {
        "repo": "prism-ml/Bonsai-8B-gguf",
        "filename": "Bonsai-8B.gguf",
        "size_mb": 1160,
    },
}

MODELS_DIR = Path(__file__).resolve().parent.parent / "models"


def download_model(size: str) -> Path:
    from huggingface_hub import hf_hub_download

    info = MODELS[size]
    print(f"Downloading Bonsai {size} (~{info['size_mb']} MB)...")
    local_path = hf_hub_download(
        repo_id=info["repo"],
        filename=info["filename"],
        local_dir=MODELS_DIR / f"bonsai-{size.lower()}",
    )
    print(f"  Saved to: {local_path}")
    return Path(local_path)


def main() -> None:
    parser = argparse.ArgumentParser(description="Download Bonsai GGUF models")
    parser.add_argument(
        "--size",
        choices=list(MODELS.keys()),
        default=None,
        help="Model size to download. Omit for all.",
    )
    args = parser.parse_args()

    MODELS_DIR.mkdir(parents=True, exist_ok=True)

    sizes = [args.size] if args.size else list(MODELS.keys())
    for size in sizes:
        download_model(size)

    print("\nAll downloads complete.")
    print(f"Models directory: {MODELS_DIR}")


if __name__ == "__main__":
    main()
