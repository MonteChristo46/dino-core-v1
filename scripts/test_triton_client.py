"""Integration test for DinoCoreTritonClient (gRPC) against docker-compose Triton.

Usage:
    PYTHONPATH=. .venv/bin/python scripts/test_triton_client.py
    PYTHONPATH=. .venv/bin/python scripts/test_triton_client.py --threshold 0.5
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
from PIL import Image

from src.triton.models import AnomalyResult
from src.triton.preprocessing import preprocess_image
import tritonclient.grpc as grpcclient


def test_client(url: str, category: str, image_size: int, threshold: float):
    ensemble_name = f"patchcore_ensemble_{category}"

    print(f"Connecting to Triton gRPC at {url}...")
    client = grpcclient.InferenceServerClient(url=url)

    if not client.is_server_live():
        print("FAIL: Triton server is not live.")
        return False

    if not client.is_model_ready(ensemble_name):
        print(f"FAIL: Model '{ensemble_name}' is not ready.")
        return False

    print(f"OK: Model '{ensemble_name}' is ready.\n")

    dummy_image = Image.fromarray(
        np.random.randint(0, 256, (image_size, image_size, 3), dtype=np.uint8)
    )

    input_tensor = preprocess_image(dummy_image, target_size=(image_size, image_size))

    inputs = [grpcclient.InferInput("IMAGE", input_tensor.shape, "FP32")]
    inputs[0].set_data_from_numpy(input_tensor)

    outputs = [
        grpcclient.InferRequestedOutput("SCORE"),
        grpcclient.InferRequestedOutput("MASK"),
    ]

    print(f"Running inference on '{ensemble_name}'...")
    response = client.infer(ensemble_name, inputs=inputs, outputs=outputs)

    anomaly_score = float(response.as_numpy("SCORE")[0])
    anomaly_map = response.as_numpy("MASK")
    if anomaly_map.ndim == 4:
        anomaly_map = np.squeeze(anomaly_map, axis=0)

    is_anomalous = bool(anomaly_score > threshold) if threshold is not None else False

    print(f"\nResult:")
    print(f"  anomaly_score: {anomaly_score:.6f}")
    print(f"  anomaly_map shape: {anomaly_map.shape}")
    print(f"  is_anomalous: {is_anomalous} (threshold={threshold})")

    expected_shape = (1, image_size, image_size)
    if anomaly_map.shape == expected_shape:
        print(f"  Mask shape OK: {anomaly_map.shape}")
    else:
        print(f"  WARNING: expected {expected_shape}, got {anomaly_map.shape}")
        return False

    # Build AnomalyResult (same structure as the shared package)
    result = AnomalyResult(
        anomaly_score=anomaly_score,
        anomaly_map=anomaly_map,
        is_anomalous=is_anomalous,
    )
    print(f"\n  repr: {result}")

    print("\nAll checks passed.")
    return True


def main():
    parser = argparse.ArgumentParser(
        description="Test DinoCoreTritonClient (gRPC) against running Triton server"
    )
    parser.add_argument("--url", default="localhost:8001", help="Triton gRPC URL")
    parser.add_argument("--category", default="mvtec_bottle")
    parser.add_argument("--image-size", type=int, default=384)
    parser.add_argument("--threshold", type=float, default=None)
    args = parser.parse_args()

    success = test_client(args.url, args.category, args.image_size, args.threshold)
    if not success:
        exit(1)


if __name__ == "__main__":
    main()
