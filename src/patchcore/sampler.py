import abc
from typing import Union

import numpy as np
import torch
import tqdm


class IdentitySampler:
    def run(
        self, features: Union[torch.Tensor, np.ndarray]
    ) -> Union[torch.Tensor, np.ndarray]:
        return features


class BaseSampler(abc.ABC):
    def __init__(self, percentage: float):
        if not 0 < percentage < 1:
            raise ValueError("Percentage value not in (0, 1).")
        self.percentage = percentage

    @abc.abstractmethod
    def run(
        self, features: Union[torch.Tensor, np.ndarray]
    ) -> Union[torch.Tensor, np.ndarray]:
        pass

    def _store_type(self, features: Union[torch.Tensor, np.ndarray]) -> None:
        self.features_is_numpy = isinstance(features, np.ndarray)
        if not self.features_is_numpy:
            self.features_device = features.device

    def _restore_type(self, features: torch.Tensor) -> Union[torch.Tensor, np.ndarray]:
        if self.features_is_numpy:
            return features.cpu().numpy()
        return features.to(self.features_device)


class GreedyCoresetSampler(BaseSampler):
    # Chunk size (rows) for the random-projection feature reduction.
    # Projecting the whole [N, D] bank in one call temporarily holds N*D
    # floats on the GPU (~7 GiB for 1.8M x 1024); PyTorch's caching
    # allocator keeps that block reserved, and FAISS' raw cudaMalloc for
    # its GPU index then fails with "out of memory" even though the memory
    # is logically free.
    _REDUCE_CHUNK_ROWS = 65536

    def __init__(
        self,
        percentage: float,
        device: torch.device,
        dimension_to_project_features_to=128,
    ):
        """Greedy Coreset sampling base class."""
        super().__init__(percentage)

        self.device = device
        self.dimension_to_project_features_to = dimension_to_project_features_to

    def _reduce_features(self, features):
        if features.shape[1] == self.dimension_to_project_features_to:
            return features
        mapper = torch.nn.Linear(
            features.shape[1], self.dimension_to_project_features_to, bias=False
        )
        _ = mapper.to(self.device)
        # Chunked + no_grad: never materialize the full [N, D] bank (or an
        # autograd graph over it) on the GPU.
        reduced = torch.empty(
            (features.shape[0], self.dimension_to_project_features_to),
            dtype=features.dtype,
            device=self.device,
        )
        with torch.no_grad():
            for start in range(0, features.shape[0], self._REDUCE_CHUNK_ROWS):
                stop = start + self._REDUCE_CHUNK_ROWS
                reduced[start:stop] = mapper(features[start:stop].to(self.device))
        return reduced

    def run(
        self, features: Union[torch.Tensor, np.ndarray]
    ) -> Union[torch.Tensor, np.ndarray]:
        """Subsamples features using Greedy Coreset.

        Args:
            features: [N x D]
        """
        if self.percentage == 1:
            return features
        self._store_type(features)
        if isinstance(features, np.ndarray):
            features = torch.from_numpy(features)
        reduced_features = self._reduce_features(features)
        sample_indices = self._compute_greedy_coreset_indices(reduced_features)
        del reduced_features
        if getattr(self.device, "type", None) == "cuda":
            # Release the projection (and other cached blocks) back to the
            # driver before the caller allocates the FAISS GPU index.
            torch.cuda.empty_cache()
        features = features[sample_indices]
        return self._restore_type(features)

    @staticmethod
    def _compute_batchwise_differences(
        matrix_a: torch.Tensor, matrix_b: torch.Tensor
    ) -> torch.Tensor:
        """Computes batchwise Euclidean distances using PyTorch."""
        a_times_a = matrix_a.unsqueeze(1).bmm(matrix_a.unsqueeze(2)).reshape(-1, 1)
        b_times_b = matrix_b.unsqueeze(1).bmm(matrix_b.unsqueeze(2)).reshape(1, -1)
        a_times_b = matrix_a.mm(matrix_b.T)

        return (-2 * a_times_b + a_times_a + b_times_b).clamp(0, None).sqrt()

    def _compute_greedy_coreset_indices(self, features: torch.Tensor) -> np.ndarray:
        """Runs iterative greedy coreset selection.

        Args:
            features: [NxD] input feature bank to sample.
        """
        distance_matrix = self._compute_batchwise_differences(features, features)
        coreset_anchor_distances = torch.norm(distance_matrix, dim=1)

        coreset_indices = []
        num_coreset_samples = int(len(features) * self.percentage)

        for _ in range(num_coreset_samples):
            select_idx = torch.argmax(coreset_anchor_distances).item()
            coreset_indices.append(select_idx)

            coreset_select_distance = distance_matrix[
                :, select_idx : select_idx + 1  # noqa E203
            ]
            coreset_anchor_distances = torch.cat(
                [coreset_anchor_distances.unsqueeze(-1), coreset_select_distance], dim=1
            )
            coreset_anchor_distances = torch.min(coreset_anchor_distances, dim=1).values

        return np.array(coreset_indices)


class ApproximateGreedyCoresetSampler(GreedyCoresetSampler):
    def __init__(
        self,
        percentage: float,
        device: torch.device,
        number_of_starting_points: int = 10,
        dimension_to_project_features_to: int = 128,
    ):
        """Approximate Greedy Coreset sampling base class."""
        self.number_of_starting_points = number_of_starting_points
        super().__init__(percentage, device, dimension_to_project_features_to)

    def _compute_greedy_coreset_indices(self, features: torch.Tensor) -> np.ndarray:
        """Runs approximate iterative greedy coreset selection.

        This greedy coreset implementation does not require computation of the
        full N x N distance matrix and thus requires a lot less memory, however
        at the cost of increased sampling times.

        Args:
            features: [NxD] input feature bank to sample.
        """
        number_of_starting_points = np.clip(
            self.number_of_starting_points, None, len(features)
        )
        start_points = np.random.choice(
            len(features), number_of_starting_points, replace=False
        ).tolist()

        approximate_distance_matrix = self._compute_batchwise_differences(
            features, features[start_points]
        )
        # Running per-row distance to the nearest selected coreset point, as a
        # flat [N] tensor updated in place. The previous implementation rebuilt
        # the buffer via torch.cat and synced the device with .item() on every
        # pick, which serializes the GPU and dominates wall time for the
        # million-row banks produced by tiled PatchCore.
        anchor_distances = torch.mean(approximate_distance_matrix, dim=-1)
        num_coreset_samples = int(len(features) * self.percentage)
        coreset_indices = torch.empty(
            num_coreset_samples, dtype=torch.long, device=features.device
        )

        with torch.no_grad():
            # Per-row squared norms do not depend on the picks: hoist them out
            # of the loop so each iteration costs one [N x D] @ [D x 1] matmul
            # instead of re-reducing the full bank. Computed with the same bmm
            # formulation _compute_batchwise_differences uses, so the picked
            # indices stay bitwise-identical to the previous implementation.
            features_squared_norms = features.unsqueeze(1).bmm(
                features.unsqueeze(2)
            ).reshape(-1)
            for i in tqdm.tqdm(range(num_coreset_samples), desc="Subsampling..."):
                # Keep as 0-dim tensor: indexing with it avoids a host sync.
                select_idx = torch.argmax(anchor_distances)
                coreset_indices[i] = select_idx
                # dist^2 = ||a||^2 + ||b||^2 - 2<a, b>, in the same operand
                # order as _compute_batchwise_differences. b_times_b goes
                # through the same 1-row bmm as the old code: batched-bmm
                # results can differ by 1 ulp from a 1-row bmm, which would
                # flip argmax ties at million-row scale.
                selected = features[select_idx.reshape(1)]
                select_distances = features.mm(selected.t())
                select_distances.squeeze_(1).mul_(-2).add_(features_squared_norms)
                select_distances.add_(
                    selected.unsqueeze(1).bmm(selected.unsqueeze(2)).reshape(())
                )
                select_distances.clamp_(0).sqrt_()
                torch.minimum(anchor_distances, select_distances, out=anchor_distances)

        return coreset_indices.cpu().numpy()


class RandomSampler(BaseSampler):
    def __init__(self, percentage: float):
        super().__init__(percentage)

    def run(
        self, features: Union[torch.Tensor, np.ndarray]
    ) -> Union[torch.Tensor, np.ndarray]:
        """Randomly samples input feature collection.

        Args:
            features: [N x D]
        """
        num_random_samples = int(len(features) * self.percentage)
        subset_indices = np.random.choice(
            len(features), num_random_samples, replace=False
        )
        subset_indices = np.array(subset_indices)
        return features[subset_indices]
