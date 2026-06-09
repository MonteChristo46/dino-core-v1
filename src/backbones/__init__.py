import transformers
from .dinov3 import DinoV3Backbone
from .swin_transformer_v2 import SwinTransformerV2Backbone

DINOV3_WEIGHTS_DIRS = {
    "vits": "weights/dinov3_vits",
    "vitsplus": "weights/dinov3_vitsplus",
    "vitb": "weights/dinov3_vitb",
    "vitl": "weights/dinov3_vitl",
    "vithplus": "weights/dinov3_vithplus",
}

SWINV2_WEIGHTS_DIRS = {
    "base": "weights/swinv2_base_simMIM",
    "large": "weights/swinv2_large_simMIM",
}


def _resolve_dinov3_weights_dir(name: str) -> str:
    name_lower = name.lower()
    for key, path in DINOV3_WEIGHTS_DIRS.items():
        if key in name_lower:
            import os

            if os.path.exists(path):
                return path
    return "weights/dinov3"


def _resolve_swinv2_name(name: str) -> str:
    name_lower = name.lower()
    for key in SWINV2_WEIGHTS_DIRS:
        if key in name_lower:
            return f"swinv2_{key}"
    return "swinv2_base"


def load(name, token=None, **kwargs):
    if "dinov3" in name.lower():
        weights_dir = kwargs.get("weights_dir") or _resolve_dinov3_weights_dir(name)
        model = DinoV3Backbone(weights_dir=weights_dir)
        return model

    if "swinv2" in name.lower() or "swin" in name.lower():
        swin_name = _resolve_swinv2_name(name)
        pretrained = kwargs.get("pretrained", True)
        stages = kwargs.get("stages_to_extract", (1, 2, 3))
        model = SwinTransformerV2Backbone(
            name=swin_name, pretrained=pretrained, stages_to_extract=stages
        )
        return model

    model = transformers.AutoModel.from_pretrained(name, token=token)
    return model

    model = transformers.AutoModel.from_pretrained(name, token=token)
    return model

