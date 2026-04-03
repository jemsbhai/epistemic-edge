"""
Experiment 00b: Download comparison GGUF models from HuggingFace.

Downloads standard quantized models for cross-model comparison against
PrismML Bonsai 1-bit models.

All models are Q4_K_M quantization for fair comparison (industry standard
4-bit quantization). These run on standard llama.cpp — no special kernels.

Comparison rationale:
  - Qwen3 8B:      Bonsai's parent architecture, isolates 1-bit vs 4-bit
  - Llama 3.2 3B:  Popular edge model, different architecture family
  - Phi-3.5 mini:  Microsoft edge-optimized, strong instruction following

Requires: pip install huggingface_hub

Usage:
    python 00_download_comparison_models.py              # All 3 models
    python 00_download_comparison_models.py --model qwen3-8b
    python 00_download_comparison_models.py --model llama-3.2-3b
    python 00_download_comparison_models.py --model phi-3.5-mini
"""

import argparse
from pathlib import Path

MODELS = {
    "qwen3-8b": {
        "repo": "Qwen/Qwen3-8B-GGUF",
        "filename": "Qwen3-8B-Q4_K_M.gguf",
        "size_gb": 5.03,
        "params": "8B",
        "quant": "Q4_K_M",
        "architecture": "qwen3",
        "notes": "Bonsai 8B parent architecture. Disable thinking mode with /no_think.",
    },
    "llama-3.2-3b": {
        "repo": "bartowski/Llama-3.2-3B-Instruct-GGUF",
        "filename": "Llama-3.2-3B-Instruct-Q4_K_M.gguf",
        "size_gb": 2.02,
        "params": "3B",
        "quant": "Q4_K_M",
        "architecture": "llama",
        "notes": "Meta edge model, optimized for multilingual dialogue and agentic tasks.",
    },
    "phi-3.5-mini": {
        "repo": "bartowski/Phi-3.5-mini-instruct-GGUF",
        "filename": "Phi-3.5-mini-instruct-Q4_K_M.gguf",
        "size_gb": 2.39,
        "params": "3.8B",
        "quant": "Q4_K_M",
        "architecture": "phi3",
        "notes": "Microsoft edge-optimized, 128K context, strong instruction following.",
    },
}

MODELS_DIR = Path(__file__).resolve().parent.parent / "models"


def download_model(model_id: str) -> Path:
    from huggingface_hub import hf_hub_download

    info = MODELS[model_id]
    print(f"\nDownloading {model_id} (~{info['size_gb']} GB)...")
    print(f"  Repo: {info['repo']}")
    print(f"  File: {info['filename']}")
    print(f"  Params: {info['params']}, Quant: {info['quant']}")

    local_dir = MODELS_DIR / model_id
    local_path = hf_hub_download(
        repo_id=info["repo"],
        filename=info["filename"],
        local_dir=local_dir,
    )
    print(f"  Saved to: {local_path}")
    return Path(local_path)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Download comparison GGUF models for cross-model benchmarking"
    )
    parser.add_argument(
        "--model",
        choices=list(MODELS.keys()),
        default=None,
        help="Specific model to download. Omit for all.",
    )
    args = parser.parse_args()

    MODELS_DIR.mkdir(parents=True, exist_ok=True)

    model_ids = [args.model] if args.model else list(MODELS.keys())

    total_gb = sum(MODELS[m]["size_gb"] for m in model_ids)
    print(f"Will download {len(model_ids)} model(s), ~{total_gb:.1f} GB total")

    for model_id in model_ids:
        download_model(model_id)

    print("\nAll downloads complete.")
    print(f"Models directory: {MODELS_DIR}")

    print("\nModel paths for llama-server:")
    for model_id in model_ids:
        info = MODELS[model_id]
        path = MODELS_DIR / model_id / info["filename"]
        print(f"  {model_id}: {path}")


if __name__ == "__main__":
    main()
