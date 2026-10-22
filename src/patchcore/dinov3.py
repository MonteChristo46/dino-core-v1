"""Compatibility shim. The canonical DinoV3Backbone lives in src/backbones/dinov3.py.

Scripts run with either the repo root or src/ on sys.path, so both import
paths are tried: 'backbones.*' when src/ is on the path, 'src.backbones.*'
when the repo root is.
"""
try:
    from backbones.dinov3 import DinoV3Backbone
except ModuleNotFoundError as _e:
    if _e.name not in ("backbones", "src"):
        raise
    from src.backbones.dinov3 import DinoV3Backbone

__all__ = ["DinoV3Backbone"]
