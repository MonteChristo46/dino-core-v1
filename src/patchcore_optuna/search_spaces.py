from dataclasses import dataclass

PATCH_SIZES: list[int] = [1, 3, 5, 7]
NUM_NN_CHOICES: list[int] = [5, 7, 9]
DEFAULT_CATEGORIES: list[str] = ["cable", "capsule", "transistor", "wood"]
CORESET_PCT: float = 0.1
BATCH_SIZE: int = 2
NUM_WORKERS: int = 8


@dataclass(frozen=True)
class BackboneConfig:
    name: str
    weights_dir: str
    num_layers: int
    pretrain_dim: int
    target_dim: int
    image_size: int
    resize: int
    layer_combos: list[str]


_LAYER_COMBOS_12: list[str] = [
    "4",
    "6",
    "8",
    "10",
    "12",
    "4,8",
    "6,12",
    "4,12",
    "8,10",
    "4,8,12",
    "2,6,10",
    "4,6,8",
    "8,10,12",
    "5,6",
    "6,7",
]

_LAYER_COMBOS_24: list[str] = [
    "8",
    "12",
    "16",
    "20",
    "24",
    "8,16",
    "12,24",
    "8,24",
    "16,20",
    "8,16,24",
    "4,12,20",
    "8,12,16",
    "16,20,24",
    "10,11",
    "12,13",
]

BACKBONE_CONFIGS: dict[str, BackboneConfig] = {
    "dinov3_vits": BackboneConfig(
        name="dinov3_vits",
        weights_dir="weights/dinov3_vits",
        num_layers=12,
        pretrain_dim=384,
        target_dim=384,
        image_size=384,
        resize=384,
        layer_combos=_LAYER_COMBOS_12,
    ),
    "dinov3_vitb": BackboneConfig(
        name="dinov3_vitb",
        weights_dir="weights/dinov3_vitb",
        num_layers=12,
        pretrain_dim=768,
        target_dim=768,
        image_size=384,
        resize=384,
        layer_combos=_LAYER_COMBOS_12,
    ),
    "dinov3_vitl": BackboneConfig(
        name="dinov3_vitl",
        weights_dir="weights/dinov3_vitl",
        num_layers=24,
        pretrain_dim=1024,
        target_dim=1024,
        image_size=384,
        resize=384,
        layer_combos=_LAYER_COMBOS_24,
    ),
    "dinov3_vithplus": BackboneConfig(
        name="dinov3_vithplus",
        weights_dir="weights/dinov3_vithplus",
        num_layers=24,
        pretrain_dim=1280,
        target_dim=1280,
        image_size=384,
        resize=384,
        layer_combos=_LAYER_COMBOS_24,
    ),
}


def get_backbone_config(name: str) -> BackboneConfig:
    if name not in BACKBONE_CONFIGS:
        raise ValueError(
            f"Unknown backbone '{name}'. Available: {list(BACKBONE_CONFIGS.keys())}"
        )
    return BACKBONE_CONFIGS[name]


def parse_layers(layers_str: str) -> tuple[str, ...]:
    return tuple(layers_str.split(","))
