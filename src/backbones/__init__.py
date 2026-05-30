import transformers
from .dinov3 import DinoV3Backbone

DINOV3_WEIGHTS_DIRS = {
    "vits": "weights/dinov3_vits",
    "vitsplus": "weights/dinov3_vitsplus",
    "vitb": "weights/dinov3_vitb",
    "vitl": "weights/dinov3_vitl",
    "vithplus": "weights/dinov3_vithplus",
}

def _resolve_dinov3_weights_dir(name: str) -> str:
    name_lower = name.lower()
    for key, path in DINOV3_WEIGHTS_DIRS.items():
        if key in name_lower:
            import os
            if os.path.exists(path):
                return path
    return "weights/dinov3"

def load(name, token=None, **kwargs):
    if "dinov3" in name.lower():
        weights_dir = kwargs.get("weights_dir") or _resolve_dinov3_weights_dir(name)
        model = DinoV3Backbone(weights_dir=weights_dir)
        return model

    model = transformers.AutoModel.from_pretrained(name, token=token)
    return model

