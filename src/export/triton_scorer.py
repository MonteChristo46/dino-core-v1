"""Self-contained Triton Python backend scorer for PatchCore.

This file is copied into each scorer model version directory and runs
independently — it does NOT import from the patchcore package. All logic
(PatchMaker, RescaleSegmentor, FAISS search) is implemented inline.
"""

import json
import os
import pickle

import faiss
import numpy as np
import torch
import torch.nn.functional as F
import scipy.ndimage as ndimage
import triton_python_backend_utils as pb_utils


class PatchMaker:
    """Replicates PatchMaker.patchify/unpatch/score for the scorer."""

    def __init__(self, patchsize, stride=None):
        self.patchsize = patchsize
        self.stride = stride if stride is not None else patchsize

    def unpatch_scores(self, x, batchsize):
        """Reshape flat patch scores back to (batch, num_patches, ...)."""
        return x.reshape(batchsize, -1, *x.shape[1:])

    def score(self, x):
        """Reduce patch scores to image-level by max-pooling."""
        was_numpy = False
        if isinstance(x, np.ndarray):
            was_numpy = True
            x = torch.from_numpy(x)
        while x.ndim > 1:
            x = torch.max(x, dim=-1).values
        if was_numpy:
            return x.numpy()
        return x


class RescaleSegmentor:
    """Upsamples patch scores to input image size with Gaussian smoothing."""

    def __init__(self, device, target_size):
        self.device = device
        self.target_size = target_size
        self.smoothing = 4

    def convert_to_segmentation(self, patch_scores):
        """Convert (B, H_grid, W_grid) scores to (B, H, W) masks."""
        with torch.no_grad():
            if isinstance(patch_scores, np.ndarray):
                patch_scores = torch.from_numpy(patch_scores)
            _scores = patch_scores.to(self.device)
            _scores = _scores.unsqueeze(1)
            _scores = F.interpolate(
                _scores,
                size=self.target_size,
                mode="bilinear",
                align_corners=False,
            )
            _scores = _scores.squeeze(1)
            patch_scores = _scores.cpu().numpy()

        return [
            ndimage.gaussian_filter(ps, sigma=self.smoothing)
            for ps in patch_scores
        ]


class TritonPythonModel:
    """Triton Python backend model for PatchCore anomaly scoring."""

    def initialize(self, args):
        """Load FAISS index and configuration from model version directory."""
        self.model_dir = args["model_repository"]
        self.version_dir = args["model_version"]
        self.full_path = os.path.join(self.model_dir, self.version_dir)

        # Load parameters
        params_path = os.path.join(self.full_path, "patchcore_params.pkl")
        if os.path.exists(params_path):
            with open(params_path, "rb") as f:
                self.params = pickle.load(f)
        else:
            self.params = {}

        # Device
        if torch.cuda.is_available():
            self.device = torch.device("cuda")
        elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            self.device = torch.device("mps")
        else:
            self.device = torch.device("cpu")

        # Load FAISS index
        index_path = os.path.join(
            self.full_path, "nn_scorer_search_index.faiss"
        )
        if not os.path.exists(index_path):
            raise FileNotFoundError(
                f"FAISS index not found at {index_path}. "
                "Ensure the trained model's FAISS index is copied to the "
                "scorer version directory."
            )

        faiss_on_gpu = self.params.get("faiss_on_gpu", False)
        faiss_num_workers = self.params.get("faiss_num_workers", 4)
        faiss.omp_set_num_threads(faiss_num_workers)

        if faiss_on_gpu and hasattr(faiss, "StandardGpuResources"):
            cpu_index = faiss.read_index(index_path)
            self.search_index = faiss.index_cpu_to_gpu(
                faiss.StandardGpuResources(), 0, cpu_index
            )
        else:
            self.search_index = faiss.read_index(index_path)

        # Configuration
        self.n_nearest_neighbours = self.params.get(
            "anomaly_scorer_num_nn", 9
        )
        self.patchsize = self.params.get("patchsize", 1)
        self.patchstride = self.params.get("patchstride", 1)
        input_shape = self.params.get("input_shape", [3, 384, 384])
        target_h, target_w = input_shape[-2], input_shape[-1]

        raw_score_min = self.params.get("raw_score_min", 0.0)
        raw_score_max = self.params.get("raw_score_max", 1.0)
        score_range = raw_score_max - raw_score_min
        if score_range > 0:
            self.score_min = raw_score_min
            self.score_range = score_range
            self.apply_normalization = True
        else:
            # TODO we must apply normalization!
            self.apply_normalization = False

        self.patch_maker = PatchMaker(self.patchsize, stride=self.patchstride)
        self.anomaly_segmentor = RescaleSegmentor(
            device=self.device, target_size=(target_h, target_w)
        )

        # Parse output dtypes from config
        self.model_config = json.loads(args["model_config"])
        score_cfg = pb_utils.get_output_config_by_name(
            self.model_config, "SCORE"
        )
        self.score_dtype = pb_utils.triton_string_to_numpy(
            score_cfg["data_type"]
        )
        mask_cfg = pb_utils.get_output_config_by_name(
            self.model_config, "MASK"
        )
        self.mask_dtype = pb_utils.triton_string_to_numpy(
            mask_cfg["data_type"]
        )

    def execute(self, requests):
        """Process inference requests."""
        responses = []
        for request in requests:
            features_tensor = pb_utils.get_input_tensor_by_name(
                request, "FEATURES"
            )
            patch_shapes_tensor = pb_utils.get_input_tensor_by_name(
                request, "PATCH_SHAPES"
            )

            features = features_tensor.as_numpy()
            patch_shapes = patch_shapes_tensor.as_numpy().tolist()

            # patch_shapes is (num_layers, 2) — use first layer's grid
            grid_h, grid_w = patch_shapes[0]
            num_patches = grid_h * grid_w
            batchsize = features.shape[0] // num_patches

            # FAISS nearest-neighbor search
            distances, _ = self.search_index.search(
                features, self.n_nearest_neighbours
            )
            # Mean distance across K nearest neighbors → per-patch score
            patch_scores = np.mean(distances, axis=-1)

            # Normalize patch scores to [0, 1] range using stored params
            if self.apply_normalization:
                patch_scores = (patch_scores - self.score_min) / self.score_range
                patch_scores = np.clip(patch_scores, 0, None)

            # Image-level score: max over all patches
            image_scores = self.patch_maker.unpatch_scores(
                patch_scores, batchsize=batchsize
            )
            image_scores = image_scores.reshape(
                *image_scores.shape[:2], -1
            )
            image_scores = self.patch_maker.score(image_scores)

            if isinstance(image_scores, torch.Tensor):
                image_scores = image_scores.cpu().numpy()

            # Anomaly mask: reshape to grid → upsample → smooth
            patch_scores_2d = self.patch_maker.unpatch_scores(
                patch_scores, batchsize=batchsize
            )
            patch_scores_2d = patch_scores_2d.reshape(
                batchsize, grid_h, grid_w
            )
            masks = self.anomaly_segmentor.convert_to_segmentation(
                patch_scores_2d
            )
            masks = np.array(masks)

            # Build response
            out_score = pb_utils.Tensor(
                "SCORE", image_scores.astype(self.score_dtype)
            )
            out_mask = pb_utils.Tensor(
                "MASK", masks.astype(self.mask_dtype)
            )

            response = pb_utils.InferenceResponse(
                output_tensors=[out_score, out_mask]
            )
            responses.append(response)

        return responses

    def finalize(self):
        """Cleanup on unload."""
        pass
