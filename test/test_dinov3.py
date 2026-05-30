from types import SimpleNamespace

import torch

from patchcore import common
from patchcore.dinov3 import DinoV3Backbone


class _DummyDinoV3(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.config = SimpleNamespace(num_register_tokens=4)
        self.is_dinov3 = True

    def forward(self, images, output_hidden_states=False):
        _ = output_hidden_states
        batchsize = images.shape[0]
        hidden_states = tuple(
            torch.randn(batchsize, 1 + self.config.num_register_tokens + 196, 32)
            for _ in range(13)
        )
        return SimpleNamespace(hidden_states=hidden_states)


def test_get_patch_tokens_removes_special_tokens():
    feat = torch.randn(2, 201, 32)
    patch_tokens = DinoV3Backbone.get_patch_tokens(feat, num_register_tokens=4)
    assert patch_tokens.shape == (2, 196, 32)


def test_format_to_spatial_grid_output_shape():
    feat = torch.randn(2, 201, 32)
    spatial_feat = DinoV3Backbone.format_to_spatial_grid(feat, num_register_tokens=4)
    assert spatial_feat.shape == (2, 32, 14, 14)


def test_network_feature_aggregator_with_dinov3_hidden_states():
    aggregator = common.NetworkFeatureAggregator(
        _DummyDinoV3(), ["7", "8"], torch.device("cpu")
    )
    outputs = aggregator(torch.randn(2, 3, 224, 224))

    assert set(outputs.keys()) == {"7", "8"}
    assert outputs["7"].shape == (2, 32, 14, 14)
    assert outputs["8"].shape == (2, 32, 14, 14)
    assert aggregator.feature_dimensions([3, 224, 224]) == [32, 32]
