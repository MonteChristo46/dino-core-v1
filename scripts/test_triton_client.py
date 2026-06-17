"""Integration test for DinoCoreTritonClient against the docker-compose Triton server.

Usage:
    PYTHONPATH=. python scripts/test_triton_client.py
    PYTHONPATH=. python scripts/test_triton_client.py --threshold 0.5
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
from PIL import Image

from src.triton import DinoCoreTritonClient, DinoCoreTritonSettings


def test_client(url: str, category: str, image_size: int, threshold: float):
    ensemble_name = f"patchcore_ensemble_{category}"
    settings = DinoCoreTritonSettings(url=url)
    client = DinoCoreTritonClient(settings)

    # Check liveness
    print(f"Testing connection to Triton at {url}...")
    if not client.is_active(model_name=ensemble_name):
        print(f"FAIL: Model '{ensemble_name}' is not ready or server is down.")
        return False
    print(f"OK: Model '{ensemble_name}' is ready.")

    # Create dummy image
    dummy_image = np.random.randint(0, 256, (image_size, image_size, 3), dtype=np.uint8)
    pil_image = Image.fromarray(dummy_image)

    # Run inference
    print(f"\nRunning predict() on '{ensemble_name}' with image_size={image_size}...")
    result = client.predict(
        pil_image,
        model_name=ensemble_name,
        image_size=(image_size, image_size),
        threshold=threshold,
    )

    print(f"\nResult:")
    print(f"  anomaly_score: {result.anomaly_score:.6f}")
    print(f"  anomaly_map shape: {result.anomaly_map.shape}")
    print(f"  is_anomalous: {result.is_anomalous} (threshold={threshold})")

    expected_shape = (1, image_size, image_size)
    if result.anomaly_map.shape == expected_shape:
        print(f"  Mask shape OK: {result.anomaly_map.shape}")
    else:
        print(f"  WARNING: expected mask shape {expected_shape}, got {result.anomaly_map.shape}")
        return False

    print("\nAll checks passed.")
    return True


def main():
    parser = argparse.ArgumentParser(
        description="Test DinoCoreTritonClient against running Triton server"
    )
    parser.add_argument(
        "--url",
        default="localhost:8003",
        help="Triton HTTP URL (default: localhost:8000)",
    )
    parser.add_argument(
        "--category",
        default="mvtec_bottle",
        help="Category name (default: mvtec_bottle)",
    )
    parser.add_argument(
        "--image-size",
        type=int,
        default=384,
        help="Input image size (default: 384)",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=None,
        help="Anomaly threshold (optional, default: None = always non-anomalous)",
    )
    args = parser.parse_args()

    success = test_client(
        args.url, args.category, args.image_size, args.threshold
    )
    if not success:
        exit(1)


if __name__ == "__main__":
    main()
