import os

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
    # Longest keys first so e.g. "vitsplus" does not match "vits". On a match we
    # return immediately even if the dir is missing — the constructor then raises
    # with the exact path, instead of silently loading a different model size.
    for key in sorted(DINOV3_WEIGHTS_DIRS, key=len, reverse=True):
        if key in name_lower:
            return DINOV3_WEIGHTS_DIRS[key]
    if os.path.exists("weights/dinov3"):
        return "weights/dinov3"
    raise FileNotFoundError(
        f"Could not resolve a DINOv3 weights directory for '{name}'. "
        f"Expected one of: {sorted(DINOV3_WEIGHTS_DIRS.values())}"
    )

def load(name, token=None, **kwargs):
    if "dinov3" in name.lower():
        weights_dir = kwargs.get("weights_dir") or _resolve_dinov3_weights_dir(name)
        model = DinoV3Backbone(weights_dir=weights_dir)
        return model

    model = transformers.AutoModel.from_pretrained(name, token=token)
    return model

