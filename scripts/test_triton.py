"""Test script for Triton PatchCore ensemble deployment.

Sends a random image to the ensemble and verifies output shapes.

Usage:
    uv run python scripts/test_triton.py
    uv run python scripts/test_triton.py --category bottle
    uv run python scripts/test_triton.py --url localhost:8000
"""

import argparse

import numpy as np
import tritonclient.http as httpclient


def test_triton(url: str, category: str, image_size: int):
    ensemble_name = f"patchcore_ensemble_{category}"

    try:
        client = httpclient.InferenceServerClient(url=url)
    except Exception as e:
        print(f"Failed to connect to Triton at {url}: {e}")
        return False

    if not client.is_server_live():
        print("Triton server is not live")
        return False

    if not client.is_model_ready(ensemble_name):
        print(f"Model {ensemble_name} is not ready")
        print("Available models:", client.get_model_repository_index())
        return False

    # Dummy image: (1, 3, H, W)
    dummy_image = np.random.rand(1, 3, image_size, image_size).astype(
        np.float32
    )

    inputs = [
        httpclient.InferInput(
            "IMAGE", dummy_image.shape, "FP32"
        )
    ]
    inputs[0].set_data_from_numpy(dummy_image)

    outputs = [
        httpclient.InferRequestedOutput("SCORE"),
        httpclient.InferRequestedOutput("MASK"),
    ]

    print(f"Sending inference request to {ensemble_name}...")
    response = client.infer(ensemble_name, inputs, outputs=outputs)

    score = response.as_numpy("SCORE")
    mask = response.as_numpy("MASK")

    print(f"Success!")
    print(f"  Score: {score}")
    print(f"  Score shape: {score.shape}")
    print(f"  Mask shape: {mask.shape}")

    if mask.shape == (1, image_size, image_size):
        print("  Mask shape is correct!")
    else:
        print(f"  WARNING: Expected mask shape (1, {image_size}, {image_size})")

    if score.shape == (1,):
        print("  Score shape is correct!")
    else:
        print(f"  WARNING: Expected score shape (1,)")

    return True


def main():
    parser = argparse.ArgumentParser(
        description="Test Triton PatchCore ensemble"
    )
    parser.add_argument(
        "--url",
        default="localhost:8000",
        help="Triton HTTP URL (default: localhost:8000)",
    )
    parser.add_argument(
        "--category",
        default="bottle",
        help="Category name (default: bottle)",
    )
    parser.add_argument(
        "--image-size",
        type=int,
        default=384,
        help="Input image size (default: 384)",
    )
    args = parser.parse_args()

    success = test_triton(args.url, args.category, args.image_size)
    if not success:
        exit(1)


if __name__ == "__main__":
    main()
