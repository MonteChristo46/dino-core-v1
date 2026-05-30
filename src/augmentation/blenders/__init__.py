from .base import BlendResult, Blender
from .feather import FeatherBlender
from .pyramid import PyramidBlender
from .seamless_clone import SeamlessCloneBlender

BLENDERS = {
    "pyramid": PyramidBlender,
    "seamless_clone": SeamlessCloneBlender,
    "mixed": SeamlessCloneBlender,
    "normal": SeamlessCloneBlender,
    "feather": FeatherBlender,
}


def create_blender(name: str, **kwargs) -> Blender:
    cls = BLENDERS.get(name)
    if cls is None:
        raise ValueError(f"Unknown blender: '{name}'. Available: {list(BLENDERS.keys())}")
    return cls(**kwargs)


__all__ = ["BlendResult", "Blender", "PyramidBlender", "SeamlessCloneBlender", "FeatherBlender", "create_blender"]
